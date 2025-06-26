from flask import Flask, request, render_template, jsonify
import json
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

app = Flask(__name__)

# Load your dataset
with open("DataRightHS.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
records = raw_data.get("items", [])

# Extract unique keys and their sample values (only non-empty strings)
key_samples = {}
for row in records:
    for key, val in row.items():
        if isinstance(val, str) and val.strip():
            key_samples.setdefault(key, val)

# Create LangChain documents for each key
docs = [Document(page_content=key) for key in key_samples.keys()]

# Create embeddings using Ollama Llama3.2
embedding = OllamaEmbeddings(model="llama3.2")
vectorstore = FAISS.from_documents(docs, embedding)

# All available headers from the Figma interface
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

    return render_template("index.html", header=header, matches=matches, selected=selected)

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