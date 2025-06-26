from flask import Flask, request, render_template, jsonify
import json
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

app = Flask(__name__)

# Load dataset on startup for UI usage
DATA_PATH = Path("data/DataRightHS.json")
with DATA_PATH.open(encoding="utf-8") as f:
    raw_data = json.load(f)
records = raw_data.get("items", raw_data)

# Extract key samples
key_samples = {}
for row in records:
    for key, val in row.items():
        if isinstance(val, str) and val.strip():
            key_samples.setdefault(key, val.strip())

# Vector store from static RHS keys
docs = [Document(page_content=key) for key in key_samples]
embedding = OllamaEmbeddings(model="llama3.2")
vectorstore = FAISS.from_documents(docs, embedding)

figma_headers = [
    "Name", "Account", "Sales Stage", "Win Probability", "AI Score",
    "Total value", "Source", "Expected closure", "Created", "Alerts"
]

@app.route("/", methods=["GET"])
def index():
    all_matches = {header: get_matches(header) for header in figma_headers}
    return render_template("index.html", all_matches=all_matches, headers=figma_headers)

@app.route("/api/match", methods=["POST"])
def api_match():
    data_in = request.get_json(force=True, silent=True) or {}
    header = data_in.get("header", "")
    if not header:
        return jsonify({"error": "Missing 'header' in request"}), 400
    return jsonify({"header": header, "matches": get_matches(header)})

@app.route("/api/match-all", methods=["GET"])
def api_match_all():
    return jsonify({header: get_matches(header) for header in figma_headers})

@app.route("/api/map", methods=["POST"])
def api_map():
    payload = request.get_json(force=True, silent=True) or {}
    lhs_headers = payload.get("lhs", [])
    rhs_rows = payload.get("rhs", [])

    if not lhs_headers or not rhs_rows:
        return jsonify({"error": "Missing 'lhs' or 'rhs' in request"}), 400

    rhs_key_samples = {}
    for row in rhs_rows:
        for key, val in row.items():
            if isinstance(val, str) and val.strip():
                rhs_key_samples.setdefault(key, val.strip())

    rhs_docs = [Document(page_content=key) for key in rhs_key_samples]
    rhs_store = FAISS.from_documents(rhs_docs, OllamaEmbeddings(model="llama3.2"))

    def match(header):
        results = rhs_store.similarity_search(header, k=5)
        top = []
        for doc in results:
            key = doc.page_content
            val = rhs_key_samples.get(key, "")
            if val:
                top.append({"field": key, "value": val})
            if len(top) == 3:
                break
        return top

    result = {header: match(header) for header in lhs_headers}
    return jsonify(result)

def get_matches(query):
    results = vectorstore.similarity_search(query, k=5)
    top = []
    for doc in results:
        key = doc.page_content
        val = key_samples.get(key, "")
        if val:
            top.append({"field": key, "value": val})
        if len(top) == 3:
            break
    return top

if __name__ == "__main__":
    print("🚀 Running with Ollama model: llama3.2")
    app.run(debug=True)