from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, os, requests, ollama, math

app = Flask(__name__)

FEEDBACK_PATH = "feedback_memory.json"
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

def load_feedback():
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("correct", {})
                    data.setdefault("incorrect", {})
                    data.setdefault("last_run", {})
                    return data
        except Exception as e:
            print(f"⚠️ load_feedback: {e}")
    return {"correct": {}, "incorrect": {}, "last_run": {}}

def save_feedback():
    try:
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ save_feedback: {e}")

feedback_memory = load_feedback()

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else s

def _tokens(s: str) -> list[str]:
    if not isinstance(s, str): return []
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)  # camelCase -> camel Case
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s)    # keep alnum separators
    return [t for t in s.strip().lower().split() if t]

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 0.0
    return len(a & b) / max(1, len(a | b))

def _is_mostly_upper(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if not letters: return False
    upp = sum(c.isupper() for c in letters)
    return upp / len(letters) >= 0.8 and len(letters) >= 4

def _is_mostly_numeric(s: str) -> bool:
    digits = sum(c.isdigit() for c in s)
    return digits / max(1, len(s)) >= 0.4

def _looks_like_id(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False
    if " " not in t and len(t) >= 8:
        alnum_ratio = sum(c.isalnum() for c in t) / len(t)
        if alnum_ratio >= 0.9 and any(c.isdigit() for c in t):
            return True
    if re.fullmatch(r"[A-Fa-f0-9]{8,}", t):
        return True
    if re.fullmatch(r"[A-Za-z0-9_-]{12,}", t):
        return True
    return False

def _is_headerish(s: str) -> bool:
    if not isinstance(s, str): return False
    w = s.strip()
    if not w: return False
    if _looks_like_id(w): return False
    if any(ch.isdigit() for ch in w): return False
    if "_" in w or "#" in w: return False
    parts = w.split()
    if not (1 <= len(parts) <= 3): return False
    if any(len(p) > 28 for p in parts): return False
    if _is_mostly_upper(s): return False
    if _is_mostly_numeric(s): return False
    if not any(c.isalpha() for c in s): return False
    return True

ID_BASE62 = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")

def _looks_like_hex_id(s: str) -> bool:
    w = s.strip()
    return bool(re.fullmatch(r"[0-9a-fA-F]{10,}", w))  # e.g., 683f60c962e2...

def _looks_like_base62_id(s: str) -> bool:
    w = s.strip()
    # long, mostly base62-ish, no spaces -> likely a node/id
    return len(w) >= 10 and " " not in w and all(c in ID_BASE62 for c in w)

def _valid_figma_label(t: str) -> bool:
    if not isinstance(t, str): return False
    s = t.strip()
    if not s: return False
    # drop obvious IDs / codes / refs
    if _looks_like_hex_id(s) or _looks_like_base62_id(s): return False
    # shape constraints for column headers
    if "_" in s or "#" in s: return False
    if "-" in s: return False                           # avoids “Dashboard - My Opportunities”
    if any(ch.isdigit() for ch in s): return False      # drop “Q3 2024”, etc.
    if len(s) > 24: return False                        # too long for a column header
    if len(s.split()) > 3: return False                 # very long phrases
    return _is_headerish(s)




def build_blocklist() -> set:
    """
    Normalized set of headers/patterns to never output again,
    sourced from feedback_memory['incorrect'].
    """
    blocked = set()
    inc = feedback_memory.get("incorrect", {}) or {}
    for hdr, pats in inc.items():
        blocked.add(_norm(hdr))
        if isinstance(pats, list):
            for p in pats:
                if isinstance(p, str):
                    blocked.add(_norm(p))
    return blocked

def force_decode(raw):
    try:
        rounds = 0
        while isinstance(raw, str) and rounds < 6:
            rounds += 1
            try:
                raw = json.loads(raw)
                continue
            except Exception:
                break
        return raw
    except Exception:
        return raw

def get_payload(req):
    payload = req.get_json(silent=True)
    if isinstance(payload, dict) and payload:
        return payload
    # (You can add form/files/raw fallbacks here if you need them again)
    return None

def extract_figma_text(figma_json: dict) -> list[str]:
    """
    Collect ONLY text from nodes with type == 'TEXT' (characters).
    No 'name' keys or component metadata. Then keep only header-ish shapes.
    """
    out = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                val = node.get("characters", "")
                if isinstance(val, str):
                    s = val.strip()
                    if s and not is_numeric(s) and _is_headerish(s):
                        out.append(s)
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    # de-dup preserve order
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

# ============== RHS field universe (generic) ==============
def extract_all_keys(data, prefix=""):
    keys = set()
    if isinstance(data, dict):
        for k, v in data.items():
            full = f"{prefix}.{k}" if prefix else k
            keys.add(full)
            keys.update(extract_all_keys(v, full))
    elif isinstance(data, list):
        for item in data:
            keys.update(extract_all_keys(item, prefix))
    return keys

def field_leaf(path: str) -> str:
    """
    Generic leaf: prefer token after last '.properties.'; otherwise last dotted token.
    Ignore custom extension suffixes like 'x-...'.
    """
    if not isinstance(path, str) or not path: return ""
    parts = path.split(".")
    # prefer schema-ish properties
    try:
        last_prop = max(i for i, t in enumerate(parts) if t == "properties")
        j = last_prop + 1
        if j < len(parts):
            leaf = parts[j]
            if leaf == "items" and j + 1 < len(parts):
                leaf = parts[j + 1]
        else:
            leaf = parts[-1]
    except ValueError:
        leaf = parts[-1]
    # strip extension-like tokens
    if leaf.startswith("x-"):
        # walk back to previous non x- token
        for tok in reversed(parts):
            if not tok.startswith("x-"):
                leaf = tok
                break
    return leaf

def collect_rhs_paths_and_leaves(data):
    paths = sorted(list(extract_all_keys(data)))
    meta = []
    for p in paths:
        meta.append({"path": p, "leaf": field_leaf(p)})
    return meta

def build_faiss_on_leaves(rhs_meta):
    docs = []
    for m in rhs_meta:
        # FAISS over leaf names, but store full path in metadata
        docs.append(Document(page_content=m["leaf"], metadata={"path": m["path"]}))
    return FAISS.from_documents(docs, OllamaEmbeddings(model=OLLAMA_MODEL))

# ============== LLM prompt (minimal change) ==============
def make_prompt_from_figma(labels: list[str]) -> str:
    """
    Keep your existing behavior: choose ONLY from provided labels.
    We do not bias with any hard-coded header lists.
    """
    blob = "\n".join(f"- {t}" for t in labels)
    incorrect = set(p for pats in feedback_memory["incorrect"].values() for p in pats)
    correct   = set(p for pats in feedback_memory["correct"].values()   for p in pats)
    avoid = ("\nAvoid patterns like:\n" + "\n".join(f"- {p}" for p in incorrect)) if incorrect else ""
    prefer = ("\nPrefer patterns like:\n" + "\n".join(f"- {p}" for p in correct)) if correct else ""

    return f"""
Extract TABLE COLUMN HEADERS from the candidate label list.

Rules (follow ALL strictly):
- Use ONLY labels that appear in the list (do not invent).
- Keep them short (1–3 words), human-readable, column-like (not actions/menus/status).
- ALWAYS RETURN an output - never have any empty headers!
- DO NOT select labels that contain an underscore "_" or a hash "#".
- Avoid generic technical/container words such as: components, schemas, properties, paths, tags, servers, definitions, refs.


{avoid}
{prefer}

Return STRICT JSON: keys = your normalized headers, values = the EXACT matched label.

Candidate labels:
{blob}
""".strip()

# ============== Header gating via RHS affinity (pattern-only) ==============
def has_rhs_affinity(header: str, rhs_meta: list[dict], min_overlap: float = 0.34) -> bool:
    """
    Keep a header only if it has plausible affinity to at least one RHS leaf:
    - exact leaf match (case-insensitive), or
    - substring either way (header in leaf or leaf in header), or
    - Jaccard token overlap >= min_overlap (default 0.34).
    - ALWAYS RETURN an output - never have any empty headers
    - Avoid generic technical/container words such as: components, schemas, properties, paths, tags, servers, definitions, refs.
    - If a candidate violates these rules, skip it and choose another that fits.



    """
    h = header.strip()
    htoks = set(_tokens(h))
    hnorm = _norm(h)
    if not htoks and not hnorm:
        return False
    for m in rhs_meta:
        leaf = m["leaf"] or ""
        ln = _norm(leaf)
        ltoks = set(_tokens(leaf))
        if not leaf:
            continue
        if ln == hnorm:
            return True
        if hnorm and (hnorm in ln or ln in hnorm):
            return True
        if htoks and ltoks and _jaccard(htoks, ltoks) >= min_overlap:
            return True
    return False

def rank_candidates_for(header: str, rhs_meta: list[dict], field_index, k: int = 3):
    h = header.strip()
    hnorm = _norm(h)
    htoks = set(_tokens(h))
    scored = []

    # lexical passes over leaves (win first)
    for m in rhs_meta:
        leaf = m["leaf"] or ""
        ln = _norm(leaf)
        ltoks = set(_tokens(leaf))
        score = None
        if ln == hnorm:
            score = 0.0                          # exact = best
        elif hnorm and (hnorm in ln or ln in hnorm):
            score = 0.25                         # substring
        else:
            j = _jaccard(htoks, ltoks)
            if j > 0:
                score = 1.0 - min(0.99, j)       # better overlap -> lower score
        if score is not None:
            scored.append((score, m["path"], leaf))

    scored.sort(key=lambda x: x[0])
    out = [{"field": p, "field_short": leaf, "score": float(s)} for (s, p, leaf) in scored[:k]]

    # FAISS backfill if needed
    if len(out) < k:
        try:
            res = field_index.similarity_search_with_score(header, k=k*2)
            for doc, dist in res:
                pth = doc.metadata.get("path", "")
                leaf = doc.page_content
                tup = {"field": pth, "field_short": leaf, "score": float(dist) + 0.5}  # keep after lexical hits
                if all(pth != x["field"] for x in out):
                    out.append(tup)
                if len(out) >= k:
                    break
        except Exception:
            pass

    out.sort(key=lambda x: x.get("score", 1e9))
    return out[:k]

def get_ollama_info():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return {"ollama_host": OLLAMA_URL, "models": [m.get("name") for m in r.json().get("models", [])]}
        return {"ollama_host": OLLAMA_URL, "error": f"status {r.status_code}"}
    except Exception as e:
        return {"ollama_host": OLLAMA_URL, "error": str(e)}

@app.get("/api/ollama_info")
def api_ollama_info():
    return jsonify(get_ollama_info())

# ============== API: find_fields ==============
@app.post("/api/find_fields")
def api_find_fields():
    try:
        raw = get_payload(request)
        if not isinstance(raw, dict):
            return jsonify({"error": "Request must include figma_json and data_json"}), 400
        if "figma_json" not in raw or "data_json" not in raw:
            return jsonify({"error": "Missing 'figma_json' or 'data_json' keys"}), 400

        figma_json = force_decode(raw["figma_json"])
        data_json  = force_decode(raw["data_json"])

        # Build RHS universe (paths + leaves) and FAISS over leaves (for ranking/matching only)
        rhs_meta = collect_rhs_paths_and_leaves(data_json)
        field_index = build_faiss_on_leaves(rhs_meta)

        # Figma labels (TEXT nodes only), shape-filtered
        figma_labels = extract_figma_text(figma_json)
        figma_labels = [t for t in figma_labels if _valid_figma_label(t)]

                # --- EMERGENCY FIGMA FALLBACK: if TEXT-node harvest is empty, sweep all strings ---
        if not figma_labels:
            def _harvest_strings(node, bag):
                if isinstance(node, dict):
                    for v in node.values():
                        if isinstance(v, str):
                            s = v.strip()
                            # keep short-ish, header-ish strings only
                            if 1 <= len(s) <= 40 and _is_headerish(s):
                                bag.append(s)
                        elif isinstance(v, (dict, list)):
                            _harvest_strings(v, bag)
                elif isinstance(node, list):
                    for it in node:
                        _harvest_strings(it, bag)

            _tmp = []
            _harvest_strings(figma_json, _tmp)

            # de-dup and basic hygiene: no underscores / hashes / generic tech terms
            _BAD_TERMS = {
                "components", "schemas", "properties", "responses",
                "schema", "paths", "tags", "servers", "definitions", "refs"
            }
            seen = set()
            figma_labels = []
            for s in _tmp:
                if s in seen:
                    continue
                seen.add(s)
                if "_" in s or "#" in s:
                    continue
                if _norm(s) in _BAD_TERMS:
                    continue
                figma_labels.append(s)
        # --- END EMERGENCY FIGMA FALLBACK ---


        # Drop labels previously marked incorrect (candidate filter)
       # Drop labels previously marked incorrect
        blocked_norm = build_blocklist()
        figma_labels = [lbl for lbl in figma_labels if _norm(lbl) not in blocked_norm]
  # ---- Figma-only sanitization (keeps us away from IDs, long codes, UI chrome) ----
        _BAD_TERMS = {
            "components","schemas","properties","responses","schema",
            "paths","tags","servers","definitions","refs","required","type"
        }

        def _looks_like_id(s: str) -> bool:
            if not isinstance(s, str): 
                return False
            t = s.strip()
            # long hex/base62-ish tokens or UUID-ish bits
            if re.fullmatch(r"[A-Za-z0-9_-]{10,}", t): 
                return True
            if re.search(r"\b[0-9a-f]{8}\b", t, re.I): 
                return True
            return False
        
        def _valid_figma_label(t: str) -> bool:
            if not isinstance(t, str) or not t.strip():
                return False
            s = t.strip()
            if _looks_like_id(s):
                return False
            # drop anything with numbers (usually not column headers), underscores, or hashes
            if any(ch.isdigit() for ch in s): 
                return False
            if "_" in s or "#" in s: 
                return False
            # drop generic container/tech terms
            if _norm(s) in _BAD_TERMS: 
                return False
            return _is_headerish(s)
        
        figma_labels = [t for t in figma_labels if _valid_figma_label(t)]


        # Helper filters (keep these figma-only)
        _BAD_TERMS = {"components", "schemas", "properties", "responses", "schema", "paths", "tags", "servers", "definitions", "refs"}

        def _valid_header(s: str) -> bool:
            if not isinstance(s, str) or not s.strip():
                return False
            t = s.strip()
            if "_" in t or "#" in t:
                return False
            if _norm(t) in _BAD_TERMS:
                return False
            return True

        # LLM selection from candidate labels (STRICTLY from figma_labels)
        headers = []
        if figma_labels:
            prompt = make_prompt_from_figma(figma_labels)
            try:
                out = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
                content = (out.get("message") or {}).get("content", "") or ""
            except Exception:
                content = ""

            parsed = {}
            if content:
                try:
                    parsed = json.loads(content) or {}
                except Exception:
                    m = re.search(r"\{[\s\S]*?\}", content)
                    if m:
                        try:
                            parsed = json.loads(m.group())
                        except Exception:
                            parsed = {}

            if isinstance(parsed, dict):
                norm_fig = {_norm(x): x for x in figma_labels}
                for _, v in parsed.items():
                    if isinstance(v, str):
                        nv = _norm(v)
                        if nv in norm_fig:
                            cand = norm_fig[nv]
                            if _valid_figma_label(cand):
                                headers.append(cand)


        # Fallback (figma-only): rank figma labels by (affinity first, then shape)
        if not headers:
            def _shape_score(s: str) -> float:
                parts = s.strip().split()
                L = sum(len(p) for p in parts)
                alpha = sum(c.isalpha() for c in s)
                shout = 1.0 if _is_mostly_upper(s) else 0.0
                # lower is better
                return L - 2.0*shout - 0.5*abs(len(parts) - 2) - 0.25*alpha

            fig_sorted = sorted(
                (x for x in figma_labels if _valid_header(x)),
                key=lambda x: (
                    0 if has_rhs_affinity(x, rhs_meta, min_overlap=0.20) else 1,
                    _shape_score(x)
                )
            )

            pick, seen_local = [], set()
            for x in fig_sorted:
                if _norm(x) in blocked_norm:
                    continue
                if x in seen_local:
                    continue
                seen_local.add(x)
                if has_rhs_affinity(x, rhs_meta, min_overlap=0.20):
                    pick.append(x)
                if len(pick) >= 8:
                    break

            # If affinity didn’t yield 8 yet, top up from remaining FIGMA labels (no RHS sourcing)
            if len(pick) < 8:
                # first, allow weaker affinity
                for x in fig_sorted:
                    if x in pick:
                        continue
                    if has_rhs_affinity(x, rhs_meta, min_overlap=0.10):
                        pick.append(x)
                    if len(pick) >= 8:
                        break

            if len(pick) < 8:
                # finally, fill by best shape only (still figma-only, valid, not blocked)
                for x in fig_sorted:
                    if x in pick:
                        continue
                    pick.append(x)
                    if len(pick) >= 8:
                        break

            # If still nothing (extreme edge), salvage 1 figma label
            if not pick and figma_labels:
                pick = [next((z for z in figma_labels if _valid_header(z)), figma_labels[0])]

            headers = pick

        # Gate headers by RHS affinity (keep quality) but do not let it drop below 8
        gated = [h for h in headers if has_rhs_affinity(h, rhs_meta)]
        if len(gated) < 8:
            # relax gating using remaining figma-only candidates
            remaining = [x for x in figma_labels if x not in gated and _valid_header(x)]
            # prefer weak affinity first
            weak_aff = [x for x in remaining if has_rhs_affinity(x, rhs_meta, min_overlap=0.10)]
            for x in weak_aff:
                if x not in gated:
                    gated.append(x)
                if len(gated) >= 8:
                    break
            # then shape-only if needed
            if len(gated) < 8:
                def _shape_score2(s: str) -> float:
                    parts = s.strip().split()
                    L = sum(len(p) for p in parts)
                    alpha = sum(c.isalpha() for c in s)
                    shout = 1.0 if _is_mostly_upper(s) else 0.0
                    return L - 2.0*shout - 0.5*abs(len(parts) - 2) - 0.25*alpha
                for x in sorted(remaining, key=_shape_score2):
                    if x not in gated:
                        gated.append(x)
                    if len(gated) >= 8:
                        break

        # de-dup, final figma-only validity + blocklist
        # ------------- consolidate, ensure 8 figma-only headers -------------
# keep only valid, non-blocked, dedup
        # ------------- consolidate, ensure 8 figma-only headers -------------
        # keep only valid, non-blocked, de-dup
        seen = set()
        headers = [h for h in (gated if 'gated' in locals() else headers)
                   if _valid_figma_label(h) and _norm(h) not in blocked_norm and not (h in seen or seen.add(h))]

        # root-word dedupe to avoid many "Account ...", "Opportunity ..." variants
        def _bucket_key(h: str) -> str:
            toks = _tokens(h)
            return toks[0] if toks else _norm(h)

        kept_roots, deduped = set(), []
        for h in headers:
            r = _bucket_key(h)
            if r in kept_roots:
                continue
            kept_roots.add(r)
            deduped.append(h)
        headers = deduped

        # If we have fewer than 8, top up strictly from remaining FIGMA labels (no RHS sourcing)
        def _shape_score(s: str) -> float:
            parts = s.strip().split()
            L = sum(len(p) for p in parts)
            alpha = sum(c.isalpha() for c in s)
            shout = 1.0 if _is_mostly_upper(s) else 0.0
            # lower is better
            return L - 2.0*shout - 0.5*abs(len(parts) - 2) - 0.25*alpha

        if len(headers) < 8:
            # candidate pool: figma-only, valid, not blocked, not already chosen
            pool = [x for x in figma_labels
                    if x not in headers and _valid_figma_label(x) and _norm(x) not in blocked_norm]

            # (a) prefer weak RHS-affinity first (helps quality but doesn't block us)
            pool_aff = sorted(pool, key=lambda x: (
                0 if has_rhs_affinity(x, rhs_meta, min_overlap=0.10) else 1,
                _shape_score(x)
            ))
            for x in pool_aff:
                r = _bucket_key(x)
                if r in kept_roots:
                    continue
                headers.append(x)
                kept_roots.add(r)
                if len(headers) >= 8:
                    break

            # (b) still short? fill purely by shape (still figma-only & new root words)
            if len(headers) < 8:
                pool_shape = sorted([x for x in pool if _bucket_key(x) not in kept_roots], key=_shape_score)
                for x in pool_shape:
                    headers.append(x)
                    kept_roots.add(_bucket_key(x))
                    if len(headers) >= 8:
                        break

        # absolute last resort: if still empty, pick up to 8 best-shaped figma labels
        if not headers:
            fallback = sorted(figma_labels, key=_shape_score)
            headers = [x for x in fallback if _valid_figma_label(x)][:8]

        # keep global cap
        headers = headers[:15]
        # ------------- end ensure 8 -------------

# ------------- end ensure 8 -------------

        # -------- end Figma-only header selection --------------------------------
                # -------- FINAL GUARANTEE: if still empty, take top 8 from figma_json --------
               # -------- FINAL GUARANTEE: if still empty, take top 8 from figma_json --------
        if not headers:
            _BAD_TERMS = {"components", "schemas", "properties", "responses", "schema", "paths", "tags", "servers", "definitions", "refs"}

            # If figma_labels is empty, permissively harvest strings from the whole figma JSON
            if not figma_labels:
                def _harvest_strings(node, bag):
                    if isinstance(node, dict):
                        for v in node.values():
                            if isinstance(v, str):
                                sv = v.strip()
                                if 1 <= len(sv) <= 40:
                                    bag.append(sv)
                            elif isinstance(v, (dict, list)):
                                _harvest_strings(v, bag)
                    elif isinstance(node, list):
                        for it in node:
                            _harvest_strings(it, bag)

                tmp = []
                _harvest_strings(figma_json, tmp)
                # de-dup
                seen_tmp = set()
                figma_labels = []
                for s in tmp:
                    if s and s not in seen_tmp:
                        seen_tmp.add(s)
                        figma_labels.append(s)

            def _valid_figma_label_local(t: str) -> bool:
                if not isinstance(t, str) or not t.strip():
                    return False
                s = t.strip()
                if any(ch.isdigit() for ch in s):
                    return False
                if "_" in s or "#" in s:
                    return False
                if _norm(s) in _BAD_TERMS:
                    return False
                return _is_headerish(s)

            def _pure_figma_score(s: str) -> float:
                """Lower is better: short, 1–2 words, more alpha, not shouty."""
                s2 = s.strip()
                parts = s2.split()
                L = sum(len(p) for p in parts)
                alpha = sum(c.isalpha() for c in s2)
                shout = 1.0 if _is_mostly_upper(s2) else 0.0
                word_bonus = -0.75 if 1 <= len(parts) <= 2 else 0.0
                return L - 2.0*shout - 0.5*abs(len(parts) - 2) - 0.25*alpha + word_bonus

            # candidate pool from (possibly re-built) figma_labels only
            pool = [t for t in figma_labels if _valid_figma_label_local(t) and _norm(t) not in blocked_norm]
            pool_sorted = sorted(pool, key=_pure_figma_score)

            # dedupe by root word while filling up to 8
            def _bucket_key(h: str) -> str:
                toks = _tokens(h)
                return toks[0] if toks else _norm(h)

            headers = []
            seen_roots = set()
            for cand in pool_sorted:
                r = _bucket_key(cand)
                if r in seen_roots:
                    continue
                seen_roots.add(r)
                headers.append(cand)
                if len(headers) >= 8:
                    break

            # if still short, fill without root dedupe
            if len(headers) < 8:
                for cand in pool_sorted:
                    if cand in headers:
                        continue
                    headers.append(cand)
                    if len(headers) >= 8:
                        break

            # absolute last-resort: any strings we can find in figma (already de-duped above)
            if not headers and figma_labels:
                headers = figma_labels[:8]
        # -------- END FINAL GUARANTEE --------


        # Build matches (lexical first, FAISS as backstop)
        matches = {}
        for h in headers:
            matches[h] = rank_candidates_for(h, rhs_meta, field_index, k=3)

        # persist context for feedback
        feedback_memory["last_run"] = {}
        for h in headers:
            feedback_memory["correct"].setdefault(h, []).append(h)
            feedback_memory["last_run"][h] = {
                "matched_ui_label": h,
                "figma_text": figma_labels,
                "top_rhs_candidates": matches.get(h, [])
            }
        save_feedback()

        if request.args.get("debug") in {"1","true"}:
            return jsonify({
                "headers_extracted": headers,
                "matches": matches,
                "debug": {
                    "figma_label_count": len(figma_labels),
                    "figma_sample": figma_labels[:25],
                    "rhs_paths_count": len(rhs_meta),
                    "rhs_leaf_sample": [m["leaf"] for m in rhs_meta[:25]]
                }
            })

        return jsonify({"headers_extracted": headers, "matches": matches})

    except Exception as e:
        return jsonify({"error": "Find fields failed", "details": str(e)}), 500


# ============== API: feedback ==============
@app.post("/api/feedback")
def api_feedback():
    try:
        body = request.get_json(force=True)
        header = body.get("header")
        status = body.get("status")  # "correct" | "incorrect"
        if not header or status not in {"correct", "incorrect"}:
            return jsonify({"error": "Invalid feedback format"}), 400

        patterns = feedback_memory["correct"].get(header, [])

        if status == "correct":
            feedback_memory["correct"].setdefault(header, [])
            for p in patterns:
                if p not in feedback_memory["correct"][header]:
                    feedback_memory["correct"][header].append(p)
        else:
            # 🔴 NEW: move header & its patterns to incorrect (permanent block)
            feedback_memory["incorrect"].setdefault(header, [])
            if header not in feedback_memory["incorrect"][header]:
                feedback_memory["incorrect"][header].append(header)
            for p in patterns:
                if p not in feedback_memory["incorrect"][header]:
                    feedback_memory["incorrect"][header].append(p)
            feedback_memory["correct"].pop(header, None)

        ctx = (feedback_memory.get("last_run") or {}).get(header, {})
        matched_ui = ctx.get("matched_ui_label")
        rhs_cands  = ctx.get("top_rhs_candidates", [])
        figma_sample = "\n".join((ctx.get("figma_text") or [])[:40])

        explain_prompt = f"""
Explain briefly and neutrally (3–5 sentences) how this header could be selected:
focus on shape (short, human-like phrase), overlap with RHS leaf names, and general similarity.

Header: {header}
Matched UI label: {matched_ui}
Top RHS candidates: {json.dumps(rhs_cands, ensure_ascii=False)}
Sample UI labels:
{figma_sample}
""".strip()

        try:
            exp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": explain_prompt}])
            explanation = exp["message"]["content"]
        except Exception as oe:
            explanation = f"(Explanation unavailable: {oe})"

        save_feedback()

        return jsonify({
            "header": header,
            "status": status,
            "patterns_used": patterns,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": "Feedback failed", "details": str(e)}), 500

# ============== Root/Runner ==============
@app.get("/")
def home():
    return jsonify({
        "message": "POST /api/find_fields with {figma_json, data_json}. "
                   "Labels come from TEXT nodes only; headers gated by RHS affinity (no hard-coding). "
                   "Matches are ranked lexically on RHS leaf names with FAISS as backstop. "
                   "POST /api/feedback with {header, status}. Add ?debug=1 to inspect."
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    args = parser.parse_args()
    print(f"✅ API running at http://localhost:{args.port}")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        print("ℹ️  Ollama:", OLLAMA_URL, "ok" if r.status_code == 200 else f"status {r.status_code}")
    except Exception as e:
        print("ℹ️  Ollama:", OLLAMA_URL, f"error: {e}")
    app.run(host="0.0.0.0", port=args.port, debug=True)
