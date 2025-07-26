from flask import Flask, request, jsonify
import json, re, ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

app = Flask(__name__)

# ─────────── Extract raw text from Figma JSON ───────────
def extract_figma_text(figma_json: dict) -> list[str]:
    out = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = node.get("characters", "").strip()
                if txt and not is_numeric(txt) and txt.lower() != "text":
                    out.append(txt)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    return list(dict.fromkeys(out))  # remove duplicates, preserve order

# ─────────── Prompt Template (NO HARDCODED HEADERS) ───────────
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

# ─────────── Build FAISS index from data schema ───────────
def build_faiss_index(data: list[dict]):
    fields = set()
    for item in data:
        if isinstance(item, dict):
            for key in item:
                if key and isinstance(key, str):
                    fields.add(key.strip())
    docs = [Document(page_content=field) for field in fields]
    return FAISS.from_documents(docs, OllamaEmbeddings(model="llama3.2"))

# ─────────── /api/find_fields ───────────
@app.post("/api/find_fields")
def api_find_fields():
    try:
        req_data = request.get_json(force=True)

        figma_json_raw = req_data.get("figma_json", "")
        rhs_json_raw = req_data.get("data_json", "")

        figma_dict = json.loads(figma_json_raw)
        rhs_dict = json.loads(rhs_json_raw)
        rhs_items = rhs_dict.get("items") if isinstance(rhs_dict, dict) and "items" in rhs_dict else rhs_dict

        ui_text = extract_figma_text(figma_dict)
        prompt = make_prompt(ui_text)

        # Call LM Studio via Ollama
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw_output = response["message"]["content"]

        try:
            headers_json = json.loads(raw_output)
        except:
            match = re.search(r"\{[\s\S]*?\}", raw_output)
            headers_json = json.loads(match.group()) if match else {}

        all_headers = list(headers_json.values())

        # Build index and match all headers
        index = build_faiss_index(rhs_items)
        match_results = {}

        for header in all_headers:
            if not header.strip():
                continue
            results = index.similarity_search(header, k=5)
            match_results[header] = [{"field": r.page_content} for r in results]

        return jsonify({
            "extracted_headers": all_headers,
            "field_matches": match_results
        })

    except Exception as e:
        return jsonify({
            "error": "Failed to process",
            "details": str(e)
        }), 500

@app.get("/")
def home():
    return jsonify({"message": "Use POST /api/find_fields with raw JSON"})

if __name__ == "__main__":
    print("✅ Running Matching App")
    app.run(debug=True)