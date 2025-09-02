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

# ============== helpers ==============
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else s

# --- Figma text extraction (broader + header-ish) ---
def extract_figma_text(figma_json: dict) -> list[str]:
    """
    Collect visible-ish UI strings.
    Primary: nodes with type == 'TEXT' and 'characters'
    Also scan common text-y keys: 'characters', 'label', 'title', 'placeholder', 'text'
    Carefully allow 'name' only if it looks like a short header.
    """
    out = []
    BAD_NAMES = {"button", "icon", "avatar", "badge", "chip", "color strip", "rectangle"}
    TEXTY_KEYS = {"characters", "label", "title", "placeholder", "text"}

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def looks_like_header(t: str) -> bool:
        w = t.strip()
        if not w or is_numeric(w): return False
        parts = w.split()
        # allow 1–3 words; avoid very long labels (section titles)
        if len(parts) == 0 or len(parts) > 4: return False
        if w.lower() in BAD_NAMES: return False
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

# --- prefer column-like tokens in fallback ---
COMMON_HEADER_TOKENS = {
    "name","id","account","owner","status","stage","type","amount","value",
    "date","created","updated","email","phone","city","state","country",
    "priority","category","product","quantity","price","score","rank"
}
def headerish(s: str) -> bool:
    if not isinstance(s, str): return False
    w = s.strip()
    if not w: return False
    parts = w.split()
    if len(parts) > 4:
        return False
    tokens = {t.lower().strip(",:;()[]{}") for t in parts}
    return bool(tokens & COMMON_HEADER_TOKENS) or (1 <= len(parts) <= 3 and w.istitle())

# --- RHS field collection: skip OpenAPI scaffolding; keep deep/leaf-like fields ---
def collect_candidate_fields(data):
    """
    Return a list of dotted field paths that look like actual data columns.
    - Ignore OpenAPI roots: paths, servers, tags (they aren't columns)
    - Keep deep 'leaf-like' keys (primitives) and components.schemas.*.properties.* entries
    - Require a minimum depth to avoid generic roots
    """
    IGNORE_ROOTS = {"paths", "servers", "tags"}
    MIN_DEPTH = 3  # require at least a.b.c

    out = set()

    def is_primitive(x):
        return isinstance(x, (str, int, float, bool)) or x is None

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if prefix == "" and k in IGNORE_ROOTS:
                    # ignore top-level OpenAPI scaffolding
                    continue

                new_prefix = f"{prefix}.{k}" if prefix else k

                # Special: components.schemas.*.properties.* are likely "real" fields
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
    """
    Build FAISS over the *leaf token* of each path (better semantic match),
    but preserve full dotted path in metadata for returning results.
    """
    docs = []
    for p in candidate_paths:
        leaf = p.split(".")[-1]
        docs.append(Document(page_content=leaf, metadata={"path": p}))
    return FAISS.from_documents(docs, OllamaEmbeddings(model=OLLAMA_MODEL))

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

        # RHS candidates (deep/leaf-like) and FAISS over leaf tokens
        rhs_fields_paths = collect_candidate_fields(data_json)
        field_index = build_faiss_over_fieldnames(rhs_fields_paths)

        # 1) FIGMA labels
        figma_labels = extract_figma_text(figma_json)

        # Call model safely
        prompt = make_prompt_from_figma(figma_labels)
        content = ""
        try:
            model_out = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
            content = (model_out.get("message") or {}).get("content", "") or ""
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

        # keep headers that map to an actual Figma label; use the *value* as the header
        norm_figma = {_norm(x) for x in figma_labels}
        headers = []
        if isinstance(parsed, dict):
            for _, v in parsed.items():
                if isinstance(v, str) and _norm(v) in norm_figma:
                    headers.append(v)  # use matched UI label directly as header

        # fallback if model gave nothing usable: pick header-ish labels from Figma
        if not headers:
            candidates = [s for s in figma_labels if headerish(s)]
            if not candidates:
                candidates = figma_labels
            headers = candidates[:12]

        # de-dup preserve order
        seen = set()
        headers = [h for h in headers if not (h in seen or seen.add(h))]

        # 2) top-5 matches per header from RHS candidate paths (return full dotted path)
        matches = {}
        for header in headers:
            try:
                res = field_index.similarity_search_with_score(header, k=5)
                matches[header] = [
                    {
                        "field": r[0].metadata.get("path", r[0].page_content),  # full path
                        "leaf": r[0].page_content,  # leaf token used for match
                        "score": float(r[1])
                    }
                    for r in res
                ]
            except Exception:
                res = field_index.similarity_search(header, k=5)
                matches[header] = [
                    {
                        "field": doc.metadata.get("path", doc.page_content),
                        "leaf": doc.page_content
                    }
                    for doc in res
                ]

        # save last_run for explanations
        feedback_memory["last_run"] = {}
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
                "model_raw_len": len(content),
                "parsed_keys": list(parsed.keys())[:10]
            }
        })

    return jsonify({"headers_extracted": headers, "matches": matches})

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
                   "Headers come ONLY from Figma; for each header we return top-5 RHS fields from data_json. "
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
