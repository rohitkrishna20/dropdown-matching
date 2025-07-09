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
    return list(dict.fromkeys(text_nodes))  # de-duplicate

ui_text = extract_figma_text(lhs_data)

# ---------- Fine-Tuned Prompt ----------
def generate_top9_prompt(text_items: list[str]) -> str:
    header_blob = "\n".join(f"- {h}" for h in text_items)

    return f"""
You are analyzing all visible UI text extracted from a Figma-based **sales dashboard** design.

Your job is to find and return the **9 most meaningful data table column headers or field labels**.
These labels will be used to extract structured data from the UI — so focus only on **table columns**, **key data values**, and **record-level fields**.

🔴 Ignore:
- Navigation items (e.g. “Sales Dashboard”, “Overview”)
- Action items (e.g. “Create Lead”, “View All”)
- Channel names (e.g. “Sales visit”, “Direct mail”)
- Section titles or UI tabs (e.g. “My Quotes”, “My Orders”)

✅ Focus on:
- What a data engineer or analyst would consider a **data column**
- Fields you’d expect to match to a database (e.g. “Name”, “Win Probability”, “Total Value”)

Here is the list of all UI text extracted from the file:
---------------------------------------------------------
{header_blob}

Return only a ranked list of the 9 best column headers as JSON:
{{
  "top_headers": ["...", "...", "..."]
}}
""".strip()

# ---------- /api/top9 route ----------
@app.route("/api/top9", methods=["GET"])
def api_top9():
    prompt = generate_top9_prompt(ui_text)
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return jsonify({
        "prompt": prompt,
        "top_9": response['message']['content']
    })

# ---------- Root route (optional) ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Use /api/top9 to extract top headers from Figma UI."})

# ---------- Run App ----------
if __name__ == "__main__":
    print("Running Ollama-powered header extractor...")
    app.run(debug=True)