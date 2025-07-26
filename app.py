from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, ollama

app = Flask(__name__)

# ─────── Extract all Figma UI text ───────
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
    return list(dict.fromkeys(out))  # de-dupe, preserve order

# ─────── Header Extraction Prompt ───────
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
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

✅ DO INCLUDE:
- Labels that appear once per column in a table
- Compact and clearly descriptive field names
- Short phrases (1–3 words)
- Likely to appear in the top row of a table
- Structured data field categories (not individual values)
- Not vague, status-based, or action-based

Return a JSON like this:
{{
  "header1": "...",
  "header2": "...",
  ...
}}
Raw UI text:
{blob}
""".strip()

# ─────── Build FAISS index from dataset fields ───────
def build_faiss_index(rhs_data: list[dict]):
    fields = set()
    for row in rhs_data:
        if isinstance(row, dict):
            for k in row:
                if k and isinstance(k, str):
                    fields.add(k.strip())
    docs = [Document(page_content=field) for field in fields]
    return FAISS.from_documents(docs, OllamaEmbeddings(model="llama3.2"))

# ─────── Decode any nested JSON string until it's dict or list ───────
def force_decode(raw):
    try:
        while isinstance(raw, str):
            raw = json.loads(raw)
        return raw
    except Exception as e:
        raise ValueError(f"Failed to decode JSON: {e}")

# ─────── Main endpoint ───────
@app.post("/api/find_fields")
def api_find_fields():
    try:
        body = request.get_json(force=True)

        # Fix double-wrapped string body issue
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception as e:
                return jsonify({"error": "Failed to parse outer request body", "details": str(e)}), 400

        # Access keys manually, no .get()
        if not isinstance(body, dict):
            return jsonify({"error": "Request must be a JSON object"}), 400

        if "figma_json" not in body or "data_json" not in body:
            return jsonify({"error": "Missing 'figma_json' or 'data_json' keys"}), 400

        figma_str = body["figma_json"]
        data_str = body["data_json"]

        if not isinstance(figma_str, str) or not isinstance(data_str, str):
            return jsonify({"error": "figma_json and data_json must be stringified JSON"}), 400

        # Decode both
        figma_json = force_decode(figma_str)
        data_json = force_decode(data_str)

        print("✅ Type of data_json:", type(data_json))
        print("✅ Keys in data_json:", data_json.keys() if isinstance(data_json, dict) else "Not a dict")

        # Extract right-hand data entries
        if isinstance(data_json, dict) and "items" in data_json:
            rhs_items = data_json["items"]
        elif isinstance(data_json, dict):
            rhs_items = [data_json]
        elif isinstance(data_json, list):
            rhs_items = data_json
        else:
            raise ValueError("Invalid data_json format: must be a dict or list")

        # Extract UI labels
        figma_text = extract_figma_text(figma_json)
        prompt = make_prompt(figma_text)
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw_response = response["message"]["content"]

        try:
            parsed_headers = json.loads(raw_response)
        except:
            match = re.search(r"\{[\s\S]*?\}", raw_response)
            parsed_headers = json.loads(match.group()) if match else {}

        headers = list(parsed_headers.keys())

        # Semantic match via FAISS
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

@app.get("/")
def home():
    return jsonify({
        "message": "POST to /api/find_fields with figma_json and data_json as stringified JSON values"
    })

if __name__ == "__main__":
    print("✅ API running at http://localhost:5000/api/find_fields")
    app.run(debug=True)