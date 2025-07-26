from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, ollama

app = Flask(__name__)

# ───── Text Extraction ─────
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
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    return list(dict.fromkeys(out))

# ───── Prompt Engineering ─────
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are extracting column headers from a raw Figma-based UI. Focus only on **structured table column headers**.

❌ DO NOT include:
- Status fields or indicators
- Vague terms
- Repeated labels or empty strings
- Anything related to email or contact method
- Company or customer names
- Process stages or pipeline labels
- Alerts, buttons, widgets

✅ DO INCLUDE:
- Column labels from a table
- Clearly descriptive short phrases
- Structured field categories (not values)

Return JSON:
{{
  "header1": "...",
  ...
  "header10": "..."
}}

Raw UI text:
{blob}
""".strip()

# ───── FAISS Index Builder ─────
def build_faiss_index(rhs_data: list[dict]):
    fields = set()
    for row in rhs_data:
        if isinstance(row, dict):
            for k in row:
                if k and isinstance(k, str):
                    fields.add(k.strip())
    docs = [Document(page_content=field) for field in fields]
    return FAISS.from_documents(docs, OllamaEmbeddings(model="llama3.2"))

# ───── API Endpoint ─────
@app.post("/api/find_fields")
def api_find_fields():
    try:
        data = request.get_json(force=True)

        # These are raw strings inside nested dicts
        figma_str = data.get("figma_json")
        data_str = data.get("data_json")

        if not isinstance(figma_str, str) or not isinstance(data_str, str):
            return jsonify({"error": "figma_json and data_json must be stringified JSON"}), 400

        # Convert raw strings to real JSON
        figma_json = json.loads(figma_str)
        data_json = json.loads(data_str)

        # Step 1: Extract Figma headers
        figma_text = extract_figma_text(figma_json)
        prompt = make_prompt(figma_text)
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw_output = response["message"]["content"]

        try:
            parsed = json.loads(raw_output)
        except:
            match = re.search(r"\{[\s\S]*?\}", raw_output)
            parsed = json.loads(match.group()) if match else {}

        headers = list(parsed.keys())

        # Step 2: Build FAISS from data_json
        rhs_items = data_json.get("items") if isinstance(data_json, dict) and "items" in data_json else data_json
        index = build_faiss_index(rhs_items)

        # Step 3: Match top 5 per header
        matches = {}
        for header in headers:
            results = index.similarity_search(header, k=5)
            matches[header] = [{"field": r.page_content} for r in results]

        return jsonify({
            "headers_extracted": headers,
            "matches": matches
        })

    except Exception as e:
        return jsonify({"error": "Find fields failed", "details": str(e)}), 500

@app.get("/")
def home():
    return jsonify({"message": "POST to /api/find_fields with figma_json and data_json as stringified JSON strings inside wrapper"})

if __name__ == "__main__":
    app.run(debug=True)