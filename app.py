from flask import Flask, request, jsonify
from pathlib import Path
import json, re, ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

app = Flask(__name__)

# Load Figma UI JSON
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
    return list(dict.fromkeys(out))

ui_text = extract_figma_text(lhs_data)

def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing raw UI text extracted from a sales dashboard built in Figma.

Your job is to identify the 10 most likely **column headers** in that table.

🎯 Return ONLY a JSON object with keys "header1" through "header10"
- ❌ Exclude generic labels like “Value”, “Info”, “Details”, or “Stage” — unless part of a known header like "Sales Stage"

Example:
{{
  "header1": "___",
  "header2": "___",
  ...
  "header10": "___"
}}

Raw UI Text:
------------
{blob}
""".strip()

@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)
    try:
        resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        headers = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)
        cleaned = [h.strip() for h in headers if h.strip()]
        output = {f"header{i+1}": cleaned[i] if i < len(cleaned) else "" for i in range(10)}
        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e)
        }), 500

# Load RHS JSON (data fields)
rhs_path = Path("data/DataRightHS.json")
rhs_data = json.loads(rhs_path.read_text(encoding="utf-8"))

# Build LangChain FAISS index
def build_faiss_index(rhs_json: list[dict]) -> FAISS:
    unique_fields = set()
    for row in rhs_json:
        if isinstance(row, dict):
            unique_fields.update(row.keys())

    docs = [Document(page_content=f, metadata={"field": f}) for f in unique_fields]
    embeddings = OllamaEmbeddings(model="llama3.2")
    db = FAISS.from_documents(docs, embeddings)
    return db

faiss_index = build_faiss_index(rhs_data)

@app.post("/api/match_fields")
def api_match_fields():
    try:
        # Get headers dynamically
        top10_response = api_top10()
        if not top10_response.is_json:
            return jsonify({"error": "Top 10 headers response is not JSON"}), 500

        headers = list(top10_response.get_json().values())
        headers = [h for h in headers if h.strip()]

        results = {}
        for header in headers:
            matches = faiss_index.similarity_search(header, k=3)
            results[header] = [m.page_content for m in matches]

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": "LangChain matching failed",
            "details": str(e)
        }), 500

@app.get("/")
def home():
    return jsonify({"message": "Use /api/top10 to extract headers, /api/match_fields to find matches"})

if __name__ == "__main__":
    app.run(debug=True)