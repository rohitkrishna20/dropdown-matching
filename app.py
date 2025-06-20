from flask import Flask, request, render_template
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import json
import os

app = Flask(__name__)

# Fields shown on the webpage
DROPDOWN_FIELDS = [
    "Name", "Account", "Sales Stage", "Primary Revenue Win Probability",
    "AI Score", "Total Value", "Source", "Expected Closure", "Created",
    "Status", "Opportunity Segment"
]

# Load JSON data
with open("data/DataRightHS.json", "r") as f:
    records = json.load(f)["items"]

# Extract dropdown values for each field
field_options = {}
for field in DROPDOWN_FIELDS:
    options = set()
    for rec in records:
        val = rec.get(field)

        if not val and field == "Source":
            val = rec.get("Primary Source Name") or rec.get("Referral Source")
        if not val and field == "Total Value":
            val = rec.get("Primary Revenue Amount") or rec.get("Annual Revenue") or rec.get("Debt")
        if not val and field == "AI Score":
            val = rec.get("Growth%")

        if val:
            options.add(str(val).strip())
    field_options[field] = list(options)

# Embed with Ollama llama3:3.2
embedding = OllamaEmbeddings(model="llama3:3.2")

# Create FAISS vector store
vectorstores = {
    field: FAISS.from_documents([Document(page_content=opt) for opt in options], embedding)
    for field, options in field_options.items()
}

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""
    selected_field = ""
    if request.method == "POST":
        query = request.form.get("query")
        selected_field = request.form.get("field")
        if selected_field in vectorstores:
            matches = vectorstores[selected_field].similarity_search_with_score(query, k=3)
            results = [{"value": doc.page_content, "score": f"{score:.3f}"} for doc, score in matches]

    return render_template("index.html", fields=DROPDOWN_FIELDS, results=results, query=query, selected_field=selected_field)

if __name__ == "__main__":
    app.run(debug=True)
