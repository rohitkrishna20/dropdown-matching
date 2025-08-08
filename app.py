from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, ollama, os

app = Flask(__name__)

FEEDBACK_PATH = "feedback_memory.json"

# ─────── Feedback memory (load if present) ───────
def load_feedback():
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "correct" in data and "incorrect" in data:
                    return data
        except Exception:
            pass
    return {"correct": {}, "incorrect": {}}

def save_feedback():
    try:
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # Don’t crash the request if disk write fails
        print(f"Warning: failed to save feedback: {e}")

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
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)

    incorrect_patterns = set()
    for patterns in feedback_memory["incorrect"].values():
        incorrect_patterns.update(patterns)

    correct_patterns = set()
    for patterns in feedback_memory["correct"].values():
        correct_patterns.update(patterns)

    avoid_section = "\nAdditional patterns to avoid:\n" + "\n".join(f"- {p}" for p in incorrect_patterns) if incorrect_patterns else ""
    include_section = "\nPrioritize patterns similar to:\n" + "\n".join(f"- {p}" for p in correct_patterns) if correct_patterns else ""

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

# ─────── Decode stringified JSON safely ───────
def force_decode(raw):
    try:
        while isinstance(raw, str):
            raw = json.loads(raw)
        return raw
    except Exception as e:
        raise ValueError(f"Failed to decode JSON: {e}")

# ─────── Main API Endpoint ───────
@app.post("/api/find_fields")
def api_find_fields():
    try:
        raw = request.get_json(force=True)
        if isinstance(raw, str):
            raw = json.loads(raw)

        if not isinstance(raw, dict):
            return jsonify({"error": "Request must be a JSON object"}), 400
        if "figma_json" not in raw or "data_json" not in raw:
            return jsonify({"error": "Missing 'figma_json' or 'data_json' keys"}), 400

        figma_json = force_decode(raw["figma_json"])
        data_json = force_decode(raw["data_json"])

        if isinstance(data_json, dict) and "items" in data_json:
            rhs_items = data_json["items"]
        elif isinstance(data_json, list):
            rhs_items = data_json
        else:
            rhs_items = [data_json]

        figma_text = extract_figma_text(figma_json)
        prompt = make_prompt(figma_text)

        # Requires local Ollama to be running and model pulled
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw_response = response["message"]["content"]

        try:
            parsed_headers = json.loads(raw_response)
        except:
            match = re.search(r"\{[\s\S]*?\}", raw_response)
            parsed_headers = json.loads(match.group()) if match else {}

        headers = list(parsed_headers.keys())

        # Save pattern used per header for feedback
        for header, pattern in parsed_headers.items():
            feedback_memory["correct"].setdefault(header, []).append(pattern)

        index = build_faiss_index(rhs_items)
        matches = {}
        for header in headers:
            results = index.similarity_search(header, k=5)
            matches[header] = [{"field": r.page_content} for r in results]

        return jsonify({
            "headers_extracted": headers,
            "matches": matches
        })

    except Exception as e:
        return jsonify({
            "error": "Find fields failed",
            "details": str(e)
        }), 500

# ─────── Feedback Endpoint ───────
@app.post("/api/feedback")
def api_feedback():
    try:
        raw = request.get_json(force=True)
        header = raw.get("header")
        status = raw.get("status")

        if not header or status not in {"correct", "incorrect"}:
            return jsonify({"error": "Invalid feedback format"}), 400

        # Get the most recent pattern(s) used for the header (if available)
        patterns = feedback_memory["correct"].get(header, [])

        # Update memory
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

        return jsonify({
            "header": header,
            "status": status,
            "patterns_used": patterns
        })

    except Exception as e:
        return jsonify({
            "error": "Feedback failed",
            "details": str(e)
        }), 500

# ─────── Root route ───────
@app.get("/")
def home():
    return jsonify({
        "message": "POST to /api/find_fields with figma_json and data_json as raw stringified JSON values"
    })

if __name__ == "__main__":
    print("✅ API running at http://localhost:5000/api/find_fields")
    app.run(debug=True)  # ← no colon