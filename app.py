from flask import Flask, request, render_template
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import json
import os

app = Flask(__name__)

# Define dropdown fields of interest
DROPDOWN_FIELDS = [
    "Name", "Account", "Sales Stage", "Primary Revenue Win Probability",
    "AI Score", "Total Value", "Source", "Expected Closure", "Created",
    "Status", "Opportunity Segment"
]

# Load JSON data
with open("data/DataRightHS.json", "r") as f:
    records = json.load(f)["items"]

# Extract options for each field
field_options = {}
for field in DROPDOWN_FIELDS:
    options = set()
    for rec in records:
        val = rec.get(field)

        # Fallbacks for missing keys
        if not val and field == "Source":
            val = rec.get("Primary Source Name") or rec.get("Referral Source")
        if not val and field == "Total Value":
            val = rec.get("Primary Revenue Amount") or rec.get("Annual Revenue") or rec.get("Debt")
        if not val and field == "AI Score":
            val = rec.get("Growth%")

        if val:
            options.add(str(val).strip())
    field_options[field] = list(options)

# Initialize Ollama embedding model
embedding = OllamaEmbeddings(model="llama3.2")

# Build vectorstores per field
vectorstores = {}

for field, options in field_options.items():
    if not options:
        print(f"Skipping '{field}' — no options found.")
        continue

    # Label each entry to give LLM context
    labeled_options = [f"{field}: {opt}" for opt in options]

    # Log for debugging
    print(f"\nIndexing field: {field} ({len(labeled_options)} items)")
    for entry in labeled_options:
        print(" -", entry)

    docs = [Document(page_content=entry) for entry in labeled_options]
    vectorstores[field] = FAISS.from_documents(docs, embedding)

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
            for doc, score in matches:
                clean_text = doc.page_content.replace(f"{selected_field}: ", "")
                results.append({"value": clean_text, "score": f"{score:.3f}"})

    return render_template("index.html", fields=DROPDOWN_FIELDS, results=results, query=query, selected_field=selected_field)

if __name__ == "__main__":
    app.run(debug=True)