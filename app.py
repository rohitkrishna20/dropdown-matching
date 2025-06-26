from flask import Flask, request, render_template, jsonify
import json
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

app = Flask(__name__)

# Load your dataset from correct path
with open("data/DataRightHS.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# If the data is nested under "items", extract it
records = raw_data.get("items", raw_data)

# Collect unique non-empty keys and their values
key_samples = {}
for row in records:
    for key, val in row.items():
        if isinstance(val, str) and val.strip():
            key_samples.setdefault(key, val)

# Create LangChain documents from keys
docs = [Document(page_content=key) for key in key_samples]

# Create FAISS vector store using Ollama model
embedding = OllamaEmbeddings(model="llama3.2")
vectorstore = FAISS.from_documents(docs, embedding)

# Header labels (for UI dropdown)
figma_headers = [
    "Name", "Account", "Sales Stage", "Win Probability", "AI Score",
    "Total value", "Source", "Expected closure", "Created", "Alerts"
]

@app.route("/", methods=["GET", "POST"])
def index():
    header = None
    matches = []
    selected = None

    if request.method == "POST":
        if "selected_match" in request.form:
            selected = request.form["selected_match"]
            header = request.form["header"]
            matches = get_matches(header)
        else:
            header = request.form.get("header")
            matches = get_matches(header)

    return render_template("index.html", header=header, matches=matches, selected=selected, headers=figma_headers)

@app.route("/api/match", methods=["POST"])
def api_match():
    data_in = request.json
    header = data_in.get("header", "")
    if not header:
        return jsonify({"error": "Missing 'header' in request"}), 400

    matches = get_matches(header)
    return jsonify({
        "header": header,
        "matches": matches
    })

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
    app.run(debug=True)