from flask import Flask, request, jsonify
import json
from pathlib import Path
import ollama

app = Flask(__name__)

# ---------- Load Figma JSON (Left-Hand Side) ----------
LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

# ---------- Extract all visible UI text ----------
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
    return list(dict.fromkeys(text_nodes))  # deduplicated, ordered

ui_text = extract_figma_text(lhs_data)

# ---------- Generate prompt ----------
def generate_top9_prompt(text_items: list[str]) -> str:
    header_blob = "\n".join(f"- {h}" for h in text_items)

    return f"""
You are analyzing all visible text extracted from a Figma-based business dashboard UI.

From the following list of text items, return the **9 most important and general UI headers** that likely represent meaningful data fields, columns, or metrics.

Avoid numeric-only values or minor UI counters (e.g. “13”, “2”, “6”). Pick items that are broad, high-level indicators of business-relevant content.

Text items:
-----------
{header_blob}

Output only a ranked JSON array of the 9 best items as:
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

# ---------- Start the app ----------
if __name__ == "__main__":
    print("Running Ollama-driven Figma header extractor...")
    app.run(debug=True)