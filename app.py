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

# ============== helpers ==============
def extract_figma_text(figma_json: dict) -> list[str]:
    """Collect visible UI strings from Figma nodes of type TEXT."""
    out = []
    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = str(node.get("characters", "")).strip()
                if txt and not is_numeric(txt) and txt.lower() != "text":
                    out.append(txt)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(figma_json)
    # de-dup while preserving order
    return list(dict.fromkeys([t for t in out if t]))

def extract_all_keys(data, prefix=""):
    """Flatten nested dict/list into dotted key paths (field names only)."""
    keys = set()
    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            keys.update(extract_all_keys(v, full_key))
    elif isinstance(data, list):
        for item in data:
            keys.update(extract_all_keys(item, prefix))
    return keys

def build_faiss_over_fieldnames(field_names: list[str]):
    docs = [Document(page_content=fn) for fn in field_names]
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
    # 1) JSON body
    payload = req.get_json(silent=True)
    if isinstance(payload, dict) and payload:
        return payload
    # 2) form-data / x-www-form-urlencoded
    if req.form:
        cand = {}
        if "figma_json" in req.form: cand["figma_json"] = req.form.get("figma_json")
        if "data_json"  in req.form: cand["data_json"]  = req.form.get("data_json")
        if cand: return cand
    # 3) files
    if req.files:
        cand = {}
        if "figma_json" in req.files:
            cand["figma_json"] = req.files["figma_json"].read().decode("utf-8", errors="ignore")
        if "data_json" in req.files:
            cand["data_json"] = req.files["data_json"].read().decode("utf-8", errors="ignore")
        if cand: return cand
    # 4) raw text (ugly cURL/pastes)
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
                if esc: esc=False if c != "\\" else True
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
    # allow user feedback to bias/avoid patterns
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

OUTPUT: JSON object where keys are your extracted headers and values are the EXACT UI label you matched.

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

        # decode inputs (accept stringified, double-escaped, etc.)
        figma_json = force_decode(raw["figma_json"])
        data_json  = force_decode(raw["data_json"])

        # collect RHS items and its field-name universe
        if isinstance(data_json, dict) and "items" in data_json:
            rhs_items = data_json["items"]
        elif isinstance(data_json, list):
            rhs_items = data_json
        else:
            rhs_items = [data_json]

        rhs_fields = sorted(list(extract_all_keys(rhs_items)))    # allowed match targets
        # Build FAISS index over RHS FIELD NAMES (not values)
        field_index = build_faiss_over_fieldnames(rhs_fields)

        # 1) HEADERS ONLY FROM FIGMA
        figma_labels = extract_figma_text(figma_json)
        prompt = make_prompt_from_figma(figma_labels)
        model_out = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        content = model_out["message"]["content"]

        try:
            parsed = json.loads(content) or {}
        except Exception:
            m = re.search(r"\{[\s\S]*?\}", content)
            parsed = json.loads(m.group()) if m else {}

        # keep only headers that actually appear in the Figma text list
        headers = [h for h in parsed.keys() if isinstance(h, str) and parsed[h] in figma_labels]

        # 2) For each header, return top-3 most similar RHS field names
        matches = {}
        for header in headers:
            try:
                res = field_index.similarity_search_with_score(header, k=3)
                matches[header] = [{"field": r[0].page_content, "score": float(r[1])} for r in res]
            except Exception:
                res = field_index.similarity_search(header, k=3)
                matches[header] = [{"field": r.page_content} for r in res]

        # record last run for explanations
        feedback_memory["last_run"] = {}
        for h in headers:
            feedback_memory["correct"].setdefault(h, []).append(parsed[h])  # store matched UI label
            feedback_memory["last_run"][h] = {
                "matched_ui_label": parsed[h],
                "figma_text": figma_labels,
                "top_rhs_candidates": matches.get(h, [])
            }
        save_feedback()

        return jsonify({
            "headers_extracted": headers,   # from Figma only
            "matches": matches              # top-3 RHS field-name candidates for each header
        })

    except Exception as e:
        return jsonify({"error": "Find fields failed", "details": str(e)}), 500

@app.post("/api/feedback")
def api_feedback():
    try:
        body = request.get_json(force=True)
        header = body.get("header")
        status = body.get("status")  # "correct" | "incorrect"
        if not header or status not in {"correct", "incorrect"}:
            return jsonify({"error": "Invalid feedback format"}), 400

        # update memories
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

        # explanation request (always neutral; if incorrect, explain likely confusion)
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
            "patterns_used": patterns,     # prior matched UI labels we stored
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": "Feedback failed", "details": str(e)}), 500

@app.get("/")
def home():
    return jsonify({
        "message": "POST /api/find_fields with {figma_json, data_json}. "
                   "Headers come ONLY from Figma; for each header we return top-3 RHS fields from data_json. "
                   "POST /api/feedback with {header, status} to store feedback and get an explanation. "
                   "GET /api/ollama_info to see Ollama host/models."
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "5000")),   # 👈 default now 5000
        help="Port to run the API server on"
    )
    args = parser.parse_args()

    print(f"✅ API running at http://localhost:{args.port}")
    print("ℹ️  Ollama:", get_ollama_info())
    app.run(host="0.0.0.0", port=args.port, debug=True)