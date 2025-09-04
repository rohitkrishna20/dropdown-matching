# app.py
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, os, requests, ollama

app = Flask(__name__)

FEEDBACK_PATH = "feedback_memory.json"
OLLAMA_URL   = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# ============== persistence ==============
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

# ============== text utils (generic) ==============
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else s

def _tokens(s: str) -> list[str]:
    if not isinstance(s, str): return []
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)  # camelCase → camel Case
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s)
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

def _is_headerish(s: str) -> bool:
    """Shape-only heuristic: short, human-ish, not shouty, not numeric."""
    if not isinstance(s, str): return False
    w = s.strip()
    if not w: return False
    parts = w.split()
    if not (1 <= len(parts) <= 3): return False
    if any(len(p) > 28 for p in parts): return False
    if _is_mostly_upper(w): return False
    if _is_mostly_numeric(w): return False
    if not any(c.isalpha() for c in w): return False
    return True

# ============== tolerant JSON intake (no extra deps) ==============
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
    return None

# ============== Figma extraction (TEXT nodes only) ==============
def extract_figma_text(figma_json: dict) -> list[str]:
    """
    Collect ONLY text from nodes with type == 'TEXT' (characters).
    Then keep only header-ish shapes (short human phrases).
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
    If leaf starts with 'x-', back up to the previous non 'x-' token if any.
    """
    if not isinstance(path, str) or not path: return ""
    parts = path.split(".")
    leaf = parts[-1]
    try:
        last_prop = max(i for i, t in enumerate(parts) if t == "properties")
        j = last_prop + 1
        if j < len(parts):
            leaf = parts[j]
            if leaf == "items" and j + 1 < len(parts):
                leaf = parts[j + 1]
    except ValueError:
        pass
    if leaf.startswith("x-"):
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
        docs.append(Document(page_content=m["leaf"], metadata={"path": m["path"]}))
    return FAISS.from_documents(docs, OllamaEmbeddings(model=OLLAMA_MODEL))

# ============== Blocklist from feedback (NEVER again) ==============
def build_blocklist() -> set:
    """
    Create a normalized set of headers to never output again, based on feedback_memory['incorrect'].
    Includes both the 'header' keys and any stored patterns for that header.
    """
    blocked = set()
    inc = feedback_memory.get("incorrect", {}) or {}
    for hdr, patterns in inc.items():
        blocked.add(_norm(hdr))
        if isinstance(patterns, list):
            for p in patterns:
                if isinstance(p, str):
                    blocked.add(_norm(p))
    return blocked

# ============== LLM prompt (generic; with NEVER list) ==============
def make_prompt_from_figma(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)

    incorrect = set(p for pats in (feedback_memory.get("incorrect") or {}).values() for p in pats)
    correct   = set(p for pats in (feedback_memory.get("correct")   or {}).values() for p in pats)
    avoid  = ("\nAvoid patterns like:\n" + "\n".join(f"- {p}" for p in incorrect)) if incorrect else ""
    prefer = ("\nPrefer patterns like:\n" + "\n".join(f"- {p}" for p in correct))   if correct   else ""

    # Build visible NEVER list (only show those that are actually present among candidates)
    blocked_norm = build_blocklist()
    never_output = ""
    if blocked_norm:
        blocked_in_labels = [t for t in labels if _norm(t) in blocked_norm]
        if blocked_in_labels:
            never_output = "\nNEVER output these labels:\n" + "\n".join(f"- {t}" for t in blocked_in_labels)

    return f"""
Extract TABLE COLUMN HEADERS from the candidate label list below.
- Use ONLY labels that appear in the list (do not invent).
- Keep them short (1–3 words), human-readable, column-like (not actions/menus/status).
{avoid}
{prefer}
{never_output}

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
    - Jaccard token overlap >= min_overlap.
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

# ============== Candidate ranking (lexical first, then FAISS) ==============
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

# ============== Soft FAISS fallback for headers (prevents empty) ==============
def pick_headers_with_faiss(figma_labels: list[str], field_index, blocked_norm: set, limit: int = 10):
    """
    If we couldn't get headers from the model/affinity gating, pick figma labels
    that are most similar to *any* RHS leaf (via FAISS over leaves). This is model-agnostic:
    use the best (lowest) distance as the label's score.
    """
    scored = []
    for label in figma_labels:
        if _norm(label) in blocked_norm:
            continue
        try:
            res = field_index.similarity_search_with_score(label, k=1)
            if res:
                _, dist = res[0]
                scored.append((float(dist), label))
        except Exception:
            # If FAISS fails, we just skip scoring for this label
            continue
    scored.sort(key=lambda x: x[0])
    headers = [lbl for _, lbl in scored[:limit]]
    # final de-dup
    seen = set()
    return [h for h in headers if not (h in seen or seen.add(h))]

# ============== Ollama info (unchanged) ==============
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

        # Build RHS universe (paths + leaves) and FAISS over leaves
        rhs_meta    = collect_rhs_paths_and_leaves(data_json)
        field_index = build_faiss_on_leaves(rhs_meta)

        # Figma labels (TEXT nodes only), shape-filtered
        figma_labels_all = extract_figma_text(figma_json)

        # Apply "never again" blocklist to candidate labels
        blocked_norm = build_blocklist()
        figma_labels = [t for t in figma_labels_all if _norm(t) not in blocked_norm]

        # LLM selection from candidate labels
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
                        try: parsed = json.loads(m.group())
                        except Exception: parsed = {}

            if isinstance(parsed, dict):
                norm_fig = {_norm(x): x for x in figma_labels}
                for _, v in parsed.items():
                    if isinstance(v, str):
                        nv = _norm(v)
                        if nv in norm_fig and nv not in blocked_norm:
                            headers.append(norm_fig[nv])

        # If model produced nothing usable, take the first few figma labels (still blocklisted)
        if not headers:
            headers = [s for s in figma_labels][:12]

        # Gate headers by RHS affinity
        gated = [h for h in headers if has_rhs_affinity(h, rhs_meta)]

        # If gating killed everything (Dashboard case), use soft FAISS fallback:
        # pick figma labels most similar to RHS leaves, then gate lightly again.
        if not gated:
            fallback_headers = pick_headers_with_faiss(figma_labels, field_index, blocked_norm, limit=10)
            # light gating with lower overlap threshold
            gated = [h for h in fallback_headers if has_rhs_affinity(h, rhs_meta, min_overlap=0.2)]

        # If still empty, keep a tiny set of figma labels (last resort), still respecting blocklist
        if not gated:
            gated = figma_labels[:5]

        # De-dup, cap, and apply blocklist once more
        seen = set()
        headers = [h for h in gated if _norm(h) not in blocked_norm and not (h in seen or seen.add(h))][:15]

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
                "figma_text": figma_labels_all,
                "top_rhs_candidates": matches.get(h, [])
            }
        save_feedback()

        if request.args.get("debug") in {"1","true"}:
            return jsonify({
                "headers_extracted": headers,
                "matches": matches,
                "debug": {
                    "figma_label_count": len(figma_labels_all),
                    "figma_sample": figma_labels_all[:25],
                    "rhs_paths_count": len(rhs_meta),
                    "rhs_leaf_sample": [m["leaf"] for m in rhs_meta[:25]],
                    "blocked_norm": sorted(list(build_blocklist()))[:25]
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
            feedback_memory["incorrect"].setdefault(header, [])
            # record the header itself as a pattern too (helps blocklist)
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
focus on short human phrasing, overlap with RHS leaf names, and general similarity.

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
                   "Headers come from TEXT nodes only; blocklist applied; soft FAISS fallback prevents empty results. "
                   "POST /api/feedback with {header, status} to store feedback and block future outputs of 'incorrect' headers. "
                   "GET /api/ollama_info to see Ollama host/models. Add ?debug=1 to inspect."
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
