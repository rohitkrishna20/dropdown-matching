from flask import Flask, request, jsonify, render_template
import json
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

app = Flask(__name__)


RHS_PATH = Path("data/DataRightHS.json")
with RHS_PATH.open(encoding="utf-8") as f:
    rhs_data = json.load(f)
rhs_records = rhs_data.get("items", rhs_data)


rhs_key_samples = {}
for row in rhs_records:
    for key, val in row.items():
        if isinstance(val, str) and val.strip():
            rhs_key_samples.setdefault(key, val.strip())


docs = [Document(page_content=key) for key in rhs_key_samples]
embedding = OllamaEmbeddings(model="llama3.2")
vectorstore = FAISS.from_documents(docs, embedding)


LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

figma_headers = lhs_data.get("headers", lhs_data)  # support both formats

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

def get_matches(query):
    results = vectorstore.similarity_search(query, k=5)
    top = []
    for doc in results:
        key = doc.page_content
        val = rhs_key_samples.get(key, "")
        if val:
            top.append({"field": key, "value": val})
        if len(top) == 3:
            break
    return top

if __name__ == "__main__":
    print("Running dropdown-matching with Ollama model: llama3.2")
    app.run(debug=True)