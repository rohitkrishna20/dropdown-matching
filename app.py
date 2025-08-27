from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, ollama, os, requests

app = Flask(__name__)

FEEDBACK_PATH = "feedback_memory.json"

# ========= persistence =========
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

# ========= helpers =========
def extract_figma_text(figma_json: dict) -> list[str]:
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
    return list(dict.fromkeys(out))  # de-dupe

# === CHANGE #1: stricter prompt (limit to candidate labels only) ===
def make_prompt(labels: list[str]) -> str:
    # Only allow choosing from the extracted UI labels
    allowed = [t.strip() for t in labels if isinstance(t, str) and t.strip()]
    blob = "\n".join(f"- {t}" for t in allowed)

    incorrect_patterns = set(p for pats in feedback_memory["incorrect"].values() for p in pats)
    correct_patterns   = set(p for pats in feedback_memory["correct"].values()   for p in pats)
    avoid_headers      = set(feedback_memory["incorrect"].keys())

    avoid_lines = []
    if incorrect_patterns:
        avoid_lines.append("Additional patterns to avoid:")
        avoid_lines += [f"- {p}" for p in incorrect_patterns]
    if avoid_headers:
        avoid_lines.append("Header names to avoid entirely (do not output these exact headers):")
        avoid_lines += [f"- {h}" for h in avoid_headers]
    avoid_section = ("\n" + "\n".join(avoid_lines)) if avoid_lines else ""
    include_section = ("\nPrioritize patterns similar to:\n" + "\n".join(f"- {p}" for p in correct_patterns)) if correct_patterns else ""

    return f"""
You are extracting table COLUMN HEADERS from a Figma-based UI.

CANDIDATE LABELS (you MUST map from ONLY this list; do not invent new labels):
{blob}

❌ DO NOT include:
- Anything not present in the candidate label list
- Status fields, vague terms, repeated labels, action buttons, section names, alerts
- Data values from rows, company/customer names, email/contact words, 'status' words
{avoid_section}

✅ DO INCLUDE:
- Labels that appear once per column in a table
- Short (1–3 words), clear, descriptive
{include_section}

OUTPUT: STRICT JSON. Keys = your normalized headers. Values = the **exact label from the candidate list** that you matched.

Example:
{{
  "Name": "Full Name",
  "Location": "City"
}}
""".strip()

def extract_all_keys(data, prefix=""):
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

def build_faiss_index(rhs_data: list[dict]):
    fields = set()
    for row in rhs_data:
        if isinstance(row, dict):
            fields.update(extract_all_keys(row))
    docs = [Document(page_content=field) for field in fields]
    return FAISS.from_documents(docs, OllamaEmbeddings(model="llama3.2"))

# ---------- tolerant JSON intake (no ast/codecs) ----------
def _sanitize_jsonish(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip()
    if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
        t = t[1:-1]
    t = re.sub(r",\s*(\}|\])", r"\1", t)
    t = t.replace(": None", ": null").replace(": True", ": true").replace(": False", ": false")
    t = t.replace(":None", ": null").replace(":True", ": true").replace(":False", ": false")
    return t

def _pyish_to_json(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s
    if re.search(r'"\s*:\s*', t):
        return t
    if not re.match(r"^\s*[\{\[]", t) or "'" not in t:
        return t
    def replacer(m):
        chunk = m.group(0)
        chunk = re.sub(r"(?<!\\)'", '"', chunk)
        return chunk
    t = re.sub(r"[\{\[][^\0]*[\}\]]", replacer, t, count=1)
    return t

def force_decode(raw):
    try:
        rounds = 0
        while isinstance(raw, str) and rounds < 6:
            rounds += 1
            s = _sanitize_jsonish(raw)
            try:
                raw = json.loads(s)
                continue
            except Exception:
                pass
            try:
                s2 = s.encode("utf-8").decode("unicode_escape")
                if s2 != s:
                    try:
                        raw = json.loads(s2)
                        continue
                    except Exception:
                        s = s2
            except Exception:
                pass
            s3 = _pyish_to_json(s)
            if s3 != s:
                try:
                    raw = json.loads(s3)
                    continue
                except Exception:
                    pass
            if re.match(r'^\s*"(?:\\.|[^"])*"\s*$', s):
                raw = s.strip()[1:-1]
                continue
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
                if esc: out.append(c); esc=False
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
        while j < len(t) and t[j] not in ",\n\r}":
            j += 1
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

# ========= Ollama info (CHANGE #2: add endpoint to show host & models) =========
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

def get_ollama_info():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return {
                "ollama_host": OLLAMA_URL,
                "models": [m.get("name") for m in r.json().get("models", [])]
            }
        return {"ollama_host": OLLAMA_URL, "error": f"status {r.status_code}"}
    except Exception as e:
        return {"ollama_host": OLLAMA_URL, "error": str(e)}

@app.get("/api/ollama_info")
def api_ollama_info():
    return jsonify(get_ollama_info())

# ========= API =========
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

        if isinstance(data_json, dict) and "items" in data_json:
            rhs_items = data_json["items"]
        elif isinstance(data_json, list):
            rhs_items = data_json
        else:
            rhs_items = [data_json]

        # Build prompt and call LLM
        figma_text = extract_figma_text(figma_json)
        prompt = make_prompt(figma_text)
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw_response = response["message"]["content"]

        try:
            parsed_headers = json.loads(raw_response)
        except Exception:
            match = re.search(r"\{[\s\S]*?\}", raw_response)
            parsed_headers = json.loads(match.group()) if match else {}

        # === NEW: drop any pairs whose matched label isn't in the candidate list ===
        allowed_set = set(figma_text)
        parsed_headers = {
            h: m for h, m in (parsed_headers or {}).items()
            if isinstance(m, str) and m in allowed_set
        }

        headers = list(parsed_headers.keys())

        # Build FAISS index and find matches
        index = build_faiss_index(rhs_items)
        matches = {}
        for header in headers:
            try:
                res = index.similarity_search_with_score(header, k=5)
                matches[header] = [{"field": r[0].page_content, "score": float(r[1])} for r in res]
            except Exception:
                res = index.similarity_search(header, k=5)
                matches[header] = [{"field": r.page_content} for r in res]

        # Save pattern used per header + context for later explanation
        feedback_memory.setdefault("last_run", {})
        for header, pattern in parsed_headers.items():
            feedback_memory["correct"].setdefault(header, []).append(pattern)
            feedback_memory["last_run"][header] = {
                "matched_label": pattern,
                "figma_text": figma_text,
                "prompt_used": prompt,
                "faiss_matches": matches.get(header, [])
            }
        save_feedback()

        # Optional debug via query/body
        debug_flag = False
        try:
            body_json = request.get_json(silent=True) or {}
            debug_flag = bool(body_json.get("debug"))
        except Exception:
            pass
        if request.args.get("debug") in {"1","true","yes"}:
            debug_flag = True

        resp = {"headers_extracted": headers, "matches": matches}
        if debug_flag:
            resp["debug"] = {
                "ollama": get_ollama_info(),
                "figma_text_count": len(figma_text),
                "figma_text_sample": figma_text[:30],
                "prompt_chars": len(prompt),
                "raw_model_output": raw_response[:1500],
                "parsed_header_map": parsed_headers
            }
        return jsonify(resp)

    except Exception as e:
        return jsonify({"error": "Find fields failed", "details": str(e)}), 500

@app.post("/api/feedback")
def api_feedback():
    try:
        raw = request.get_json(force=True)
        header = raw.get("header")
        status = raw.get("status")

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

        # status-aware explanation
        explanation = None
        ctx = (feedback_memory.get("last_run") or {}).get(header, {})
        matched_label = ctx.get("matched_label")
        faiss_matches = ctx.get("faiss_matches", [])
        figma_sample = "\n".join((ctx.get("figma_text") or [])[:40])

        if status == "correct":
            explain_prompt = f"""
Provide a concise, neutral explanation (3–5 sentences) of the *patterns used* to extract this header.
Do NOT critique. Focus on: the matched UI label, column-like cues, and alignment with JSON candidates.

Header: {header}
Matched UI label: {matched_label}
Top JSON field candidates: {json.dumps(faiss_matches, ensure_ascii=False)}
Sample UI text:
{figma_sample}
""".strip()
        else:
            explain_prompt = f"""
Explain (3–5 sentences) the patterns that led to extracting "{header}" even though it was marked incorrect.
Mention the decisive UI cues and how JSON candidates aligned.

Matched UI label: {matched_label}
Top JSON field candidates: {json.dumps(faiss_matches, ensure_ascii=False)}
Sample UI text:
{figma_sample}
""".strip()

        try:
            exp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": explain_prompt}])
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

@app.get("/")
def home():
    return jsonify({
        "message": "POST /api/find_fields with figma_json and data_json (any JSON-ish format accepted). "
                   "POST /api/feedback with {header, status}. If status='incorrect' or 'correct', you get an explanation. "
                   "GET /api/ollama_info to see which Ollama host/models are active."
    })

if __name__ == "__main__":
    print("✅ API running at http://localhost:5000")
    print("ℹ️  Ollama:", get_ollama_info())
    app.run(debug=True)