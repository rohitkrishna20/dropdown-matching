from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, ollama, os, ast   # ← ast already used below

app = Flask(__name__)

FEEDBACK_PATH = "feedback_memory.json"

# ─────── Feedback memory: load/save ───────
def load_feedback():
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "correct" in data and "incorrect" in data:
                    return data
        except Exception as e:
            print(f"⚠️ Could not load feedback file: {e}")
    return {"correct": {}, "incorrect": {}}

def save_feedback():
    try:
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save feedback file: {e}")

feedback_memory = load_feedback()

# ─────── Extract all UI text from Figma JSON ───────
def extract_figma_text(figma_json: dict) -> list[str]:
    out = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def walk(node: dict):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = node.get("characters", "").strip()
                if txt and not is_numeric(txt) and txt.lower() != "text":
                    out.append(txt)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    return list(dict.fromkeys(out))  # de-dupe

# ─────── Build prompt to extract table headers ───────
def make_prompt(labels: list[str], explain: bool = False) -> str:
    blob = "\n".join(f"- {t}" for t in labels)

    incorrect_patterns = set()
    for patterns in feedback_memory["incorrect"].values():
        incorrect_patterns.update(patterns)

    correct_patterns = set()
    for patterns in feedback_memory["correct"].values():
        correct_patterns.update(patterns)

    avoid_section = "\nAdditional patterns to avoid:\n" + "\n".join(f"- {p}" for p in incorrect_patterns) if incorrect_patterns else ""
    include_section = "\nPrioritize patterns similar to:\n" + "\n".join(f"- {p}" for p in correct_patterns) if correct_patterns else ""

    # When explain=True, ask for a richer object with a short description of the pattern used.
    if explain:
        return f"""
You are extracting column headers from a raw Figma-based UI. Focus only on **structured table column headers**.

❌ DO NOT include:
- Status fields or indicators
- Vague terms (like general words for time, value, details, etc.)
- Repeated labels or empty strings
- Anything related to email or contact method
- Company or customer names
- Process stages or pipeline labels
- Words that include "status"
- Data values from inside table rows
- UI section names, menus, or action buttons
- Alerts or warnings
- Dashboard widgets, activity counters
- Any duplicates or empty entries
{avoid_section}

✅ DO INCLUDE:
- Labels that appear once per column in a table
- Compact and clearly descriptive field names
- Short phrases (1–3 words)
- Likely to appear in the top row of a table
- Structured data field categories (not individual values)
- Not vague, status-based, or action-based
{include_section}

Return **strict JSON** where:
- Each key is the extracted column header.
- Each value is an object with:
  - "matched_label": the exact UI phrase you matched (verbatim from the text list).
  - "pattern_description": a brief, 1–2 sentence description of the pattern/logic you used to decide this header (no chain-of-thought, just a concise explanation).

Example:
{{
  "Name": {{ "matched_label": "Full Name", "pattern_description": "Picked 'Full Name' because it appears as a single label likely used once per row to identify a person." }},
  "Location": {{ "matched_label": "City", "pattern_description": "City is a compact geographic field label commonly used as a column header." }}
}}

Raw UI text:
{blob}
""".strip()

    # Original behavior (non-explain): simple mapping header -> matched label
    return f"""
You are extracting column headers from a raw Figma-based UI. Focus only on **structured table column headers**.

❌ DO NOT include:
- Status fields or indicators
- Vague terms (like general words for time, value, details, etc.)
- Repeated labels or empty strings
- Anything related to email or contact method
- Company or customer names
- Process stages or pipeline labels
- Words that include "status"
- Data values from inside table rows
- UI section names, menus, or action buttons
- Alerts or warnings
- Dashboard widgets, activity counters
- Any duplicates or empty entries
{avoid_section}

✅ DO INCLUDE:
- Labels that appear once per column in a table
- Compact and clearly descriptive field names
- Short phrases (1–3 words)
- Likely to appear in the top row of a table
- Structured data field categories (not individual values)
- Not vague, status-based, or action-based
{include_section}

Return a JSON where:
- The key is the extracted column header
- The value is the **exact label or phrase** you matched it to

Example:
{{
  "Name": "Full Name",
  "Location": "City"
}}

Raw UI text:
{blob}
""".strip()

# ─────── Extract All Keys Recursively ───────
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

# ─────── Build FAISS vector index from field names ───────
def build_faiss_index(rhs_data: list[dict]):
    fields = set()
    for row in rhs_data:
        if isinstance(row, dict):
            fields.update(extract_all_keys(row))
    docs = [Document(page_content=field) for field in fields]
    return FAISS.from_documents(docs, OllamaEmbeddings(model="llama3.2"))

# ─────── Light sanitizer for broken JSON-ish strings ───────
def _sanitize_jsonish(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip()
    if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
        t = t[1:-1]
    if re.match(r"\s*\{.*\}\s*$", t, flags=re.S) and "'" in t and '"' not in t:
        t = re.sub(r"'", '"', t)
    t = re.sub(r",\s*(\}|])", r"\1", t)
    return t

# ─────── Decode stringified JSON safely ───────
def force_decode(raw):
    try:
        while isinstance(raw, str):
            s = _sanitize_jsonish(raw)
            try:
                raw = json.loads(s)
                continue
            except Exception:
                pass
            try:
                raw = ast.literal_eval(s)
                continue
            except Exception:
                pass
            break
        return raw
    except Exception as e:
        raise ValueError(f"Failed to decode JSON: {e}")

# ─────── Robust payload extractor for ugly Postman/cURL pastes ───────
def get_payload(req):
    payload = req.get_json(silent=True)
    if isinstance(payload, dict) and payload:
        return payload

    if req.form:
        cand = {}
        if "figma_json" in req.form:
            cand["figma_json"] = req.form["figma_json"]
        if "data_json" in req.form:
            cand["data_json"] = req.form["data_json"]
        if "explain" in req.form:
            cand["explain"] = req.form["explain"]
        if cand:
            return cand

    raw_txt = req.get_data(as_text=True) or ""

    def _extract_value(text: str, key: str):
        m = re.search(rf'"{re.escape(key)}"\s*:', text) or re.search(rf"'{re.escape(key)}'\s*:", text)
        if not m:
            return None
        i = m.end()
        while i < len(text) and text[i] in " \t\r\n":
            i += 1
        if i >= len(text):
            return None
        ch = text[i]
        if ch in ('"', "'"):
            quote = ch; i += 1; out=[]; esc=False
            while i < len(text):
                c = text[i]
                if esc:
                    out.append(c); esc=False
                elif c == "\\":
                    esc=True
                elif c == quote:
                    break
                else:
                    out.append(c)
                i += 1
            return quote + "".join(out) + quote
        if ch in "{[":
            stack=[ch]; j=i+1; in_str=False; str_q=""; esc=False
            while j < len(text) and stack:
                c=text[j]
                if in_str:
                    if esc: esc=False
                    elif c == "\\": esc=True
                    elif c == str_q: in_str=False
                else:
                    if c in ('"', "'"): in_str=True; str_q=c
                    elif c in "{[": stack.append(c)
                    elif c in "}]": stack.pop()
                j += 1
            return text[i:j]
        j=i
        while j < len(text) and text[j] not in ",\n\r}":
            j += 1
        return text[i:j].strip()

    figma_raw = _extract_value(raw_txt, "figma_json")
    data_raw  = _extract_value(raw_txt, "data_json")
    explain_raw = _extract_value(raw_txt, "explain")

    if figma_raw is not None and data_raw is not None:
        out = {"figma_json": figma_raw, "data_json": data_raw}
        if explain_raw is not None:
            out["explain"] = explain_raw
        return out

    m = re.search(r"\{[\s\S]*\}", raw_txt)
    if m:
        s = _sanitize_jsonish(m.group(0))
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                pass

    return None

# ─────── Main API Endpoint ───────
@app.post("/api/find_fields")
def api_find_fields():
    try:
        raw = get_payload(request)
        if not isinstance(raw, dict):
            return jsonify({"error": "Request must include figma_json and data_json"}), 400
        if "figma_json" not in raw or "data_json" not in raw:
            return jsonify({"error": "Missing 'figma_json' or 'data_json' keys"}), 400

        # explain flag: accept true/false/1/0/"yes"/"no"
        explain_val = str(raw.get("explain", "")).strip().lower()
        explain = explain_val in {"1", "true", "yes"}

        figma_json = force_decode(raw["figma_json"])
        data_json  = force_decode(raw["data_json"])

        if isinstance(data_json, dict) and "items" in data_json:
            rhs_items = data_json["items"]
        elif isinstance(data_json, list):
            rhs_items = data_json
        else:
            rhs_items = [data_json]

        figma_text = extract_figma_text(figma_json)
        prompt = make_prompt(figma_text, explain=explain)

        # Call LLM
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw_response = response["message"]["content"]

        try:
            parsed = json.loads(raw_response)
        except Exception:
            match = re.search(r"\{[\s\S]*?\}", raw_response)
            parsed = json.loads(match.group()) if match else {}

        # parsed can be {header: "pattern"} or {header: {matched_label, pattern_description}}
        headers = list(parsed.keys()) if isinstance(parsed, dict) else []

        # Build FAISS index and match
        index = build_faiss_index(rhs_items)
        matches = {}
        for header in headers:
            results = index.similarity_search(header, k=5)
            matches[header] = [{"field": r.page_content} for r in results]

        # Persist patterns for feedback and assemble provenance
        provenance = {}
        for header in headers:
            val = parsed.get(header)
            if isinstance(val, dict):
                matched_label = val.get("matched_label")
                pattern_desc  = val.get("pattern_description")
            else:
                matched_label = val
                pattern_desc  = None

            if matched_label:
                feedback_memory["correct"].setdefault(header, []).append(matched_label)

            provenance[header] = {
                "matched_label": matched_label,
                "pattern_description": pattern_desc
            }

        save_feedback()

        return jsonify({
            "headers_extracted": headers,
            "matches": matches,
            "provenance": provenance
        })

    except Exception as e:
        return jsonify({"error": "Find fields failed", "details": str(e)}), 500

# ─────── Feedback Endpoint ───────
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

        save_feedback()

        return jsonify({"header": header, "status": status, "patterns_used": patterns})

    except Exception as e:
        return jsonify({"error": "Feedback failed", "details": str(e)}), 500

# ─────── Root route ───────
@app.get("/")
def home():
    return jsonify({
        "message": "POST to /api/find_fields with figma_json and data_json (objects or stringified JSON). "
                   "Optional: add \"explain\": true to receive pattern descriptions. "
                   "POST to /api/feedback with {header, status}."
    })

if __name__ == "__main__":
    print("✅ API running at http://localhost:5000/api/find_fields")
    app.run(debug=True)