from flask import Flask, request, render_template, jsonify
import json
from pathlib import Path
from langchain_ollama import OllamaEmbeddings  # ✅ NEW import location
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

app = Flask(__name__)

# Load dataset
DATA_PATH = Path("data/DataRightHS.json")
with DATA_PATH.open(encoding="utf-8") as f:
    raw_data = json.load(f)

# Extract records
records = raw_data.get("items", raw_data)

# Extract unique, non-empty key samples
key_samples = {}
for row in records:
    for key, val in row.items():
        if isinstance(val, str) and val.strip():
            key_samples.setdefault(key, val.strip())

# Build LangChain documents
docs = [Document(page_content=key) for key in key_samples]

# Create FAISS vector store with Ollama llama3.2
embedding = OllamaEmbeddings(model="llama3.2")  # ✅ using your model
vectorstore = FAISS.from_documents(docs, embedding)

# Figma column headers
figma_headers = [
    "Name", "Account", "Sales Stage", "Win Probability", "AI Score",
    "Total value", "Source", "Expected closure", "Created", "Alerts"
]

@app.route("/", methods=["GET"])
def index():
    # Generate matches for all headers in one go
    all_matches = {header: get_matches(header) for header in figma_headers}
    return render_template("index.html", all_matches=all_matches, headers=figma_headers)

@app.route("/api/match", methods=["POST"])
def api_match():
    data_in = request.get_json(force=True, silent=True) or {}
    header = data_in.get("header", "")
    if not header:
        return jsonify({"error": "Missing 'header' in request"}), 400

    matches = get_matches(header)
    return jsonify({
        "header": header,
        "matches": matches
    })

@app.route("/api/match-all", methods=["GET"])
def api_match_all():
    result = {header: get_matches(header) for header in figma_headers}
    return jsonify(result)

def get_matches(query):
    results = vectorstore.similarity_search(query, k=5)
    final = []
    for doc in results:
        key = doc.page_content
        sample = key_samples.get(key, "")
        if sample.strip():
            final.append({"field": key, "value": sample})
        if len(final) == 3:
            break
    return final

if __name__ == "__main__":
    print("🚀 Running with Ollama model: llama3.2")
    app.run(debug=True)