from flask import Flask, jsonify
import json
from pathlib import Path
import ollama

app = Flask(__name__)

# ---------- Load Figma JSON (Left-Hand Side) ----------
LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

# ---------- Extract visible UI text ----------
def extract_figma_text(figma_json: dict):
    text_nodes = []

    def _is_mostly_numeric(text):
        cleaned = text.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def _crawl(node):
        if node.get("type") == "TEXT" and node.get("characters", "").strip():
            txt = node["characters"].strip()
            if not _is_mostly_numeric(txt):
                text_nodes.append(txt)
        for child in node.get("children", []):
            _crawl(child)

    _crawl(figma_json)
    return list(dict.fromkeys(text_nodes))  # de-duplicate while preserving order

ui_text = extract_figma_text(lhs_data)

# ---------- Generate prompt for top 10 headers ----------
def generate_top10_prompt(text_items: list[str]) -> str:
    header_blob = "\n".join(f"- {h}" for h in text_items)

    return f"""
You are analyzing UI text extracted from a Figma-based sales dashboard design.

This UI includes a structured data table with labeled columns, like a spreadsheet.

🎯 Your task: Identify the **10 most relevant table column headers or field labels** from the list below.

Only select values that represent:
- Structured fields that repeat for each row (like “Name”, “AI Score”, or “Created”)
- Labels used for table columns in business dashboards

❌ Ignore:
- Section titles (e.g. “Sales Dashboard”, “Overview”)
- Action items (e.g. “Create Lead”, “View All”)
- Navigation tabs (e.g. “My Orders”, “My Quotes”)
- Numeric-only strings or alert counts

UI Text Extracted:
------------------
{header_blob}

Return a JSON response with exactly 10 headers:
{{
  "top_headers": ["...", "...", "..."]
}}
""".strip()

# ---------- /api/top10 route ----------
@app.route("/api/top10", methods=["GET"])
def api_top10():
    prompt = generate_top10_prompt(ui_text)
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return jsonify({
        "prompt": prompt,
        "top_10": response['message']['content']
    })

# ---------- Optional root route ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Use /api/top10 to extract top headers from Figma UI."})

# ---------- Run server ----------
if __name__ == "__main__":
    print("Running header extractor (Ollama + Figma JSON)")
    app.run(debug=True)