# app.py
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, os, requests, ollama

app = Flask(__name__)

FEEDBACK_PATH = "feedback_memory.json"
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
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

# ============== basic utils ==============
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else s

def _split_tokens(s: str) -> list[str]:
    """split on camelCase, snake_case, kebab-case, and whitespace; lowercased"""
    if not isinstance(s, str): return []
    s = s.strip()
    # split camelCase boundaries by inserting spaces before capitals (except first)
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)
    s = re.sub(r"[_\-\/\.]", " ", s)
    toks = [t.lower() for t in re.split(r"\s+", s) if t.strip()]
    return toks

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    if inter == 0: return 0.0
    union = len(a | b)
    return inter / union

# ============== Figma extraction (broader + header-aware) ==============
COMMON_HEADER_TOKENS = {
    "name","id","account","owner","status","stage","type","amount","value",
    "date","created","updated","email","phone","city","state","country",
    "priority","category","product","quantity","price","score","rank","title",
    "department","role","contact","company","region","segment","source"
}
BANNED_TITLES = {"dashboard", "oracle l2q", "overview", "welcome", "home", "recent activity"}

def extract_figma_text(figma_json: dict) -> list[str]:
    out = []
    TEXTY_KEYS = {"characters", "label", "title", "placeholder", "text"}

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def looks_like_header(s: str) -> bool:
        w = s.strip()
        if not w or is_numeric(w): return False
        parts = w.split()
        if len(parts) == 0 or len(parts) > 4: return False  # avoid long section titles
        if _norm(w) in BANNED_TITLES: return False
        return True

    def maybe_add(t: str):
        if not isinstance(t, str): return
        s = t.strip()
        if s and looks_like_header(s):
            out.append(s)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                maybe_add(str(node.get("characters", "")))
            for k in TEXTY_KEYS:
                if k in node:
                    maybe_add(str(node.get(k)))
            # light use of 'name' if header-ish
            if "name" in node:
                maybe_add(str(node.get("name")))
            cp = node.get("componentProperties")
            if isinstance(cp, dict):
                for v in cp.values():
                    if isinstance(v, dict) and v.get("type") == "TEXT":
                        maybe_add(str(v.get("value", "")))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    uniq, seen = [], set()
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def score_figma_headers(figma_labels: list[str], rhs_leafs: list[str]) -> list[str]:
    """Rank labels by how column-like they are and how well they overlap RHS leaves."""
    rhs_vocab = set()
    for leaf in rhs_leafs:
        rhs_vocab.update(_split_tokens(leaf))
    ranked = []
    for lab in figma_labels:
        toks = set(_split_tokens(lab))
        score = 0.0
        # classic column tokens
        if toks & COMMON_HEADER_TOKENS: score += 2.0
        # overlap with RHS leaf vocabulary
        score += 3.0 * jaccard(toks, rhs_vocab)
        # short/clean labels
        words = lab.split()
        if 1 <= len(words) <= 3: score += 0.5
        # penalize obvious section titles
        if _norm(lab) in BANNED_TITLES: score -= 3.0
        ranked.append((score, lab))
    ranked.sort(key=lambda x: x[0], reverse=True)
    # keep the top dozen unique
    seen = set()
    result = []
    for _, lab in ranked:
        if lab not in seen:
            seen.add(lab)
            result.append(lab)
        if len(result) >= 12: break
    return result

# ============== RHS (data_json) field collection ==============
def collect_candidate_fields(data):
    """
    Return dotted field paths that look like actual data columns.
    - Ignore OpenAPI roots: paths, servers, tags
    - Keep deep 'leaf-like' keys (primitives) and components.schemas.*.properties.* entries
    """
    IGNORE_ROOTS = {"paths", "servers", "tags"}
    MIN_DEPTH = 3

    out = set()

    def is_primitive(x):
        return isinstance(x, (str, int, float, bool)) or x is None

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if prefix == "" and k in IGNORE_ROOTS:
                    continue
                new_prefix = f"{prefix}.{k}" if prefix else k
                # Treat schema properties as candidate fields
                if new_prefix.startswith("components.schemas.") and ".properties." in new_prefix:
                    out.add(new_prefix)
                if is_primitive(v):
                    if new_prefix.count(".") + 1 >= MIN_DEPTH:
                        out.add(new_prefix)
                else:
                    walk(v, new_prefix)
        elif isinstance(node, list):
            for item in node:
                walk(item, prefix)

    walk(data)
    return sorted(out)

def build_faiss_over_fieldnames(candidate_paths: list[str]):
    docs = []
    for p in candidate_paths:
        leaf = p.split(".")[-1]
        docs.append(Document(page_content=leaf, metadata={"path": p}))
    return FAISS.from_documents(docs, OllamaEmbeddings(model=OLLAMA_MODEL))

# ============== Deterministic ranking for RHS matches ==============
COMMON_SUFFIXES = ["name","id","date","amount","status","owner","type","value","email","phone"]

def rank_rhs_candidates(header: str, rhs_paths: list[str], k: int = 5):
    """
    Deterministic ranking before FAISS:
    1) exact leaf == header (case-insensitive)
    2) leaf contains header or header contains leaf
    3) header token overlap with leaf tokens
    Then fill with FAISS results (deduped).
    """
    header_norm = _norm(header)
    header_toks = set(_split_tokens(header))
    # (path, leaf, score)
    scored = []

    for p in rhs_paths:
        leaf = p.split(".")[-1]
        leaf_norm = _norm(leaf)
        leaf_toks = set(_split_tokens(leaf))

        score = 0.0
        # exact leaf
        if leaf_norm == header_norm:
            score += 1000.0
        # contains / endswith patterns (e.g., header 'name' vs 'accountName')
        if header_norm in leaf_norm or leaf_norm in header_norm:
            score += 25.0
        # prefer common suffixes match like *Name, *Owner, *Date when header is that token
        for suf in COMMON_SUFFIXES:
            if header_norm == suf and leaf_norm.endswith(suf):
                score += 10.0
        # token overlap
        score += 5.0 * jaccard(header_toks, leaf_toks)

        if score > 0.0:
            scored.append((score, p, leaf))

    scored.sort(key=lambda x: x[0], reverse=True)
    # return top prelim (dedup by path)
    prelim = []
    seen = set()
    for s, p, leaf in scored:
        if p not in seen:
            seen.add(p)
            prelim.append({"field": p, "leaf": leaf, "score": s})

    return prelim[:k]

# ---------- tolerant JSON intake ----------
def _sanitize_jsonish(s: str) -> str:
    if not isinstance(s, str): return s
    t = s.strip()
    if len(t) >= 2 and t[0] == "'" and t[-1] == "'": t = t[1:-1]
    t = re.sub(r",\s*(\}|\])", r"\1", t)
    t = (t.replace(": None", ": null").replace(": True", ": true").replace(": False", ": false")
           .replace(":None", ": null").replace(":True", ": true").replace(":False", ": false"))
    return t

def _pyish_to_json(s: str) -> str:
    if not isinstance(s, str): return s
    t = s
    if re.search(r'"\s*:\s*', t):  # likely already JSON
        return t
    if not re.match(r"^\s*[\{\[]", t) or "'" not in t:
        return t
    def replacer(m):
        chunk = m.group(0)
        return re.sub(r"(?<!\\)'", '"', chunk)
    return re.sub(r"[\{\[][^\0]*[\}\]]", replacer, t, count=1)

def force_decode(raw):
    try:
        rounds = 0
        while isinstance(raw, str) and rounds < 6:
            rounds += 1
            s = _sanitize_jsonish(raw)
            try:
                raw = json.loads(s); continue
            except Exception: pass
            try:
                s2 = s.encode("utf-8").decode("unicode_escape")
                if s2 != s:
                    try:
                        raw = json.loads(s2); continue
                    except Exception: s = s2
            except Exception: pass
            s3 = _pyish_to_json(s)
            if s3 != s:
                try:
                    raw = json.loads(s3); continue
                except Exception: pass
            if re.match(r'^\s*"(?:\\.|[^"])*"\s*$', s):  # quoted blob
                raw = s.strip()[1:-1]; continue
            break
        return raw
    except Exception as e:
        raise ValueError(f"Failed to decode JSON: {e}")

def get_payload(req):
    payload = req.get_json(silent=True)
    if isinstance(payload, dict) and payload:
        return payload
    if req.form:
        cand = {}
        if "figma_json" in req.form: cand["figma_json"] = req.form.get("figma_json")
        if "data_json"  in req.form: cand["data_json"]  = req.form.get("data_json")
        if cand: return cand
    if req.files:
        cand = {}
        if "figma_json" in req.files:
            cand["figma_json"] = req.files["figma_json"].read().decode("utf-8", errors="ignore")
        if "data_json" in req.files:
            cand["data_json"] = req.files["data_json"].read().decode("utf-8", errors="ignore")
        if cand: return cand
    text = req.get_data(as_text=True) or ""
    def grab_value(t: str, key: str):
        m = re.search(rf'(["\']){re.escape(key)}\1\s*:', t)
        if not m: return None
        i = m.end()
        while i < len(t) and t[i] in " \t\r\n": i += 1
        if i >= len(t): return None
        ch = t[i]
        if ch in ('"', "'"):
            q = ch; i += 1; out=[]; esc=False
            while i < len(t):
                c = t[i]
                if esc: esc=False
                elif c == "\\": esc=True
                elif c == q: break
                else: out.append(c)
                i += 1
            return q + "".join(out) + q
        if ch in "{[":
            stack=[ch]; j=i+1; in_str=False; q=""; esc=False
            while j < len(t) and stack:
                c=t[j]
                if in_str:
                    if esc: esc=False
                    elif c == "\\": esc=True
                    elif c == q: in_str=False
                else:
                    if c in ('"', "'"): in_str=True; q=c
                    elif c in "{[": stack.append(c)
                    elif c in "}]": stack.pop()
                j += 1
            return t[i:j]
        j=i
        while j < len(t) and t[j] not in ",\n\r}": j += 1
        return t[i:j].strip()
    f_raw = grab_value(text, "figma_json")
    d_raw = grab_value(text, "data_json")
    if f_raw is not None and d_raw is not None:
        return {"figma_json": f_raw, "data_json": d_raw}
    blob = re.search(r"\{[\s\S]*\}", text)
    if blob:
        s = _sanitize_jsonish(blob.group(0))
        s = _pyish_to_json(s)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict): return obj
        except Exception:
            try:
                s2 = s.encode("utf-8").decode("unicode_escape")
                obj = json.loads(s2)
                if isinstance(obj, dict): return obj
            except Exception:
                pass
    return None

# ============== Ollama info ==============
def get_ollama_info():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return {"ollama_host": OLLAMA_URL,
                    "models": [m.get("name") for m in r.json().get("models", [])]}
        return {"ollama_host": OLLAMA_URL, "error": f"status {r.status_code}"}
    except Exception as e:
        return {"ollama_host": OLLAMA_URL, "error": str(e)}

@app.get("/api/ollama_info")
def api_ollama_info():
    return jsonify(get_ollama_info())

# ============== LLM prompt (FIGMA -> headers only) ==============
def make_prompt_from_figma(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    incorrect = set(p for pats in feedback_memory["incorrect"].values() for p in pats)
    correct   = set(p for pats in feedback_memory["correct"].values()   for p in pats)
    avoid = ("\nAvoid patterns like:\n" + "\n".join(f"- {p}" for p in incorrect)) if incorrect else ""
    prefer = ("\nPrefer patterns like:\n" + "\n".join(f"- {p}" for p in correct)) if correct else ""

    return f"""
You are extracting TABLE COLUMN HEADERS from Figma UI text below.

RULES:
- Use ONLY labels that appear in the list (do not invent anything).
- Headers should be compact (1–3 words), column-like, not actions/status/menus.

{avoid}
{prefer}

OUTPUT: JSON object where keys are your proposed (normalized) headers
and values are the EXACT UI label you matched from the list.

Figma UI labels:
{blob}
""".strip()

# ============== API routes ==============
@app.post("/api/find_fields")
def api_find_fields():
    try:
        raw = get_payload(request)
        if not isinstance(raw, dict):
            return jsonify({"error": "Request must include figma_json and data_json"}), 400
        if "figma_json" not in raw or "data_json" not in raw:
            return jsonify({"error": "Missing 'figma_json' or 'data_json' keys"}), 400

        # decode inputs
        figma_json = force_decode(raw["figma_json"])
        data_json  = force_decode(raw["data_json"])

        # RHS candidates
        rhs_paths = collect_candidate_fields(data_json)
        rhs_leafs = [p.split(".")[-1] for p in rhs_paths]
        field_index = build_faiss_over_fieldnames(rhs_paths)

        # 1) FIGMA labels
        figma_labels = extract_figma_text(figma_json)

        # Call model safely (optional, we still have fallback)
        content, parsed = "", {}
        try:
            prompt = make_prompt_from_figma(figma_labels)
            model_out = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
            content = (model_out.get("message") or {}).get("content", "") or ""
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
        except Exception:
            parsed = {}

        # keep headers that map to an actual Figma label; use the *value* as the header
        norm_figma = {_norm(x) for x in figma_labels}
        headers_from_model = []
        if isinstance(parsed, dict):
            for _, v in parsed.items():
                if isinstance(v, str) and _norm(v) in norm_figma:
                    headers_from_model.append(v)

        # fallback + ranking by overlap with RHS leaves
        ranked_headers = score_figma_headers(figma_labels, rhs_leafs)
        # combine: model-suggested first (if any), then ranked fallback (dedup preserve order)
        seen = set()
        headers = []
        for h in headers_from_model + ranked_headers:
            if h not in seen:
                seen.add(h)
                headers.append(h)
            if len(headers) >= 12:
                break

        # 2) Deterministic matching first, FAISS to fill
        matches = {}
        for header in headers:
            prelim = rank_rhs_candidates(header, rhs_paths, k=5)

            # fill with FAISS (dedup by path)
            try:
                res = field_index.similarity_search_with_score(header, k=8)
                for doc, score in res:
                    full = doc.metadata.get("path", doc.page_content)
                    if not any(m["field"] == full for m in prelim):
                        prelim.append({"field": full, "leaf": doc.page_content, "score": float(score)})
            except Exception:
                res = field_index.similarity_search(header, k=8)
                for doc in res:
                    full = doc.metadata.get("path", doc.page_content)
                    if not any(m["field"] == full for m in prelim):
                        prelim.append({"field": full, "leaf": doc.page_content})

            # keep top 5 by score (if score missing, push to end)
            prelim.sort(key=lambda x: x.get("score", -1e9), reverse=True)
            matches[header] = prelim[:5]

        # save last_run for explanations
        feedback_memory["last_run"] = {}
        for h in headers:
            feedback_memory["correct"].setdefault(h, []).append(h)
            feedback_memory["last_run"][h] = {
                "matched_ui_label": h,
                "figma_text": figma_labels,
                "top_rhs_candidates": matches.get(h, [])
            }
        save_feedback()

        # optional debug
        if request.args.get("debug") in {"1", "true"}:
            return jsonify({
                "headers_extracted": headers,
                "matches": matches,
                "debug": {
                    "figma_label_count": len(figma_labels),
                    "figma_sample": figma_labels[:15],
                    "rhs_paths_count": len(rhs_paths),
                    "rhs_leafs_sample": rhs_leafs[:20],
                    "model_raw_len": len(content),
                    "parsed_keys": list(parsed.keys())[:10]
                }
            })

        return jsonify({"headers_extracted": headers, "matches": matches})

    except Exception as e:
        return jsonify({"error": "Find fields failed", "details": str(e)}), 500

# ============== Feedback route ==============
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
            for p in patterns:
                if p not in feedback_memory["incorrect"][header]:
                    feedback_memory["incorrect"][header].append(p)
            feedback_memory["correct"].pop(header, None)

        ctx = (feedback_memory.get("last_run") or {}).get(header, {})
        matched_ui = ctx.get("matched_ui_label")
        rhs_cands  = ctx.get("top_rhs_candidates", [])
        figma_sample = "\n".join((ctx.get("figma_text") or [])[:40])

        if status == "correct":
            explain_prompt = f"""
Briefly explain (3–5 sentences) the patterns used to select this header from Figma UI text.
Focus on the matched UI label and column-like cues. Do not critique.

Header: {header}
Matched UI label: {matched_ui}
Top RHS candidates: {json.dumps(rhs_cands, ensure_ascii=False)}
Sample Figma text:
{figma_sample}
""".strip()
        else:
            explain_prompt = f"""
Explain (3–5 sentences) how the system might have chosen this header from the Figma UI text,
and why that could be misleading given the context.

Header: {header}
Matched UI label: {matched_ui}
Top RHS candidates: {json.dumps(rhs_cands, ensure_ascii=False)}
Sample Figma text:
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

# ============== Root ==============
@app.get("/")
def home():
    return jsonify({
        "message": "POST /api/find_fields with {figma_json, data_json}. "
                   "Headers come ONLY from Figma; we deterministically rank top-5 RHS fields per header. "
                   "POST /api/feedback with {header, status} to store feedback and get an explanation. "
                   "GET /api/ollama_info to see Ollama host/models. Add ?debug=1 to /api/find_fields to inspect."
    })

# ============== Runner ==============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "5000")),
        help="Port to run the API server on"
    )
    args = parser.parse_args()

    print(f"✅ API running at http://localhost:{args.port}")
    print("ℹ️  Ollama:", get_ollama_info())
    app.run(host="0.0.0.0", port=args.port, debug=True)
