from flask import Flask, request, jsonify
from pathlib import Path
import json, re, ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)

# ──────── Load Figma UI JSON ────────
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

def extract_figma_text(figma_json: dict) -> list[str]:
    out = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def walk(node: dict):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = node.get("characters", "").strip()
                if txt and not is_numeric(txt):
                    out.append(txt)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    return list(dict.fromkeys(out))  # de-dupe

ui_text = extract_figma_text(lhs_data)

def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing raw UI text from a sales dashboard built in Figma. Extract the **10 best column headers** from this text.

Follow these strict rules:
- ✅ Must be structured field names used to label columns
- ✅ Must be unique
- ❌ Never include "Status", "Date", or "Value"
- ❌ Exclude generic or vague labels, pipeline stages, or alert messages
- ❌ Exclude values like “Web”, “E-Mail”, “Due to closure”
- Return only 10 distinct, non-empty column headers in strict JSON format.
- Never return empty strings or placeholders.

Return only:
{{
  "header1": "...",
  "header2": "...",
  ...
  "header10": "..."
}}

Raw Text:
{blob}
""".strip()

@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)
    try:
        resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        # Extract headers from model response
        headers = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)
        headers = [h.strip() for h in headers if h.strip()]  # Remove blanks

        # Remove duplicates while preserving order
        seen = set()
        headers = [h for h in headers if h.lower() not in seen and not seen.add(h.lower())]

        # Keep only the first 10 valid headers
        headers = headers[:10]

        # Build output JSON
        output = {f"header{i+1}": headers[i] if i < len(headers) else "" for i in range(10)}
        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Header extraction failed",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500


# ──────── Load Right-hand Data JSON (DataRightHS.json) ────────
rhs_path = Path("data/DataRightHS.json")
raw_rhs = json.loads(rhs_path.read_text(encoding="utf-8"))

# Unwrap nested "data" key if present
rhs_data = raw_rhs.get("items") if isinstance(raw_rhs, dict) and "items" in raw_rhs else raw_rhs

def build_faiss_index(rhs_data: list[dict]):
    all_fields = set()

    if not isinstance(rhs_data, list):
        raise ValueError("Right-hand data must be a list of dictionaries.")

    for row in rhs_data:
        if isinstance(row, dict):
            for k in row.keys():
                if isinstance(k, str) and k.strip():
                    all_fields.add(k.strip())

    if not all_fields:
        raise ValueError("❌ No field names found in right-hand data.")

    docs = [Document(page_content=field) for field in all_fields]
    embeddings = OllamaEmbeddings(model="llama3.2")
    return FAISS.from_documents(docs, embeddings)

faiss_index = build_faiss_index(rhs_data)

@app.post("/api/match_fields")
def api_match_fields():
    try:
        # Get headers dynamically
        top10 = api_top10()
        if not top10.is_json:
            return jsonify({"error": "Top 10 headers failed"}), 500
        headers = [h for h in top10.get_json().values() if h.strip()]

        out = {}
        for header in headers:
            results = faiss_index.similarity_search(header, k=3)
            matches = [{"field": r.page_content} for r in results]
            out[header] = matches

        return jsonify(out)
    except Exception as e:
        return jsonify({"error": "Semantic field match failed", "details": str(e)}), 500

@app.get("/")
def home():
    return jsonify({"message": "Use /api/top10 to extract headers or /api/match_fields to match fields."})

if __name__ == "__main__":
    print("✅ Running LangChain + FAISS + Ollama Matching App")
    app.run(debug=True)