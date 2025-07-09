from flask import Flask, jsonify
from pathlib import Path
import json
import ollama
import re

app = Flask(__name__)

# ────────────────────────────────────────────────────────────────
# 1. Load Figma JSON
# ────────────────────────────────────────────────────────────────
LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

# ────────────────────────────────────────────────────────────────
# 2. Extract visible UI text (skip numeric-only)
# ────────────────────────────────────────────────────────────────
def extract_figma_text(figma_json: dict) -> list[str]:
    labels = []

    def _is_numeric(text: str) -> bool:
        cleaned = text.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def _walk(node: dict):
        if node.get("type") == "TEXT":
            txt = node.get("characters", "").strip()
            if txt and not _is_numeric(txt):
                labels.append(txt)
        for child in node.get("children", []):
            _walk(child)

    _walk(figma_json)
    return list(dict.fromkeys(labels))  # remove duplicates

ui_text = extract_figma_text(lhs_data)

# ────────────────────────────────────────────────────────────────
# 3. Prompt builder (ban 'timestamp' and markdown output)
# ────────────────────────────────────────────────────────────────
def make_prompt_top10(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)

    return f"""
You are analyzing UI text extracted from a Figma-based sales dashboard.

The design includes a structured data table with labeled columns.

🎯 Task: Choose the **10 most likely column headers** (labels describing columns in the data table).

❗ Only include values that describe the kind of data shown across rows (like “Name”, “AI Score”, or “Expected Closure”).

❌ DO NOT include row values such as:
   - “Qualify”
   - “Negotiation”
   - “Discovery”
   - “Sales Visit”
   - “Direct Mail”
   - “Timestamp”

❌ Also ignore:
   - Section titles (e.g. “Sales Dashboard”, “Overview”)
   - Action items (e.g. “Create Lead”, “View All”)
   - Navigation tabs
   - Numeric-only items

Return only this raw JSON object — without markdown or explanation:
{{
  "top_headers": ["Header1", "Header2", ..., "Header10"]
}}

UI Text Extracted:
------------------
{blob}
""".strip()

# ────────────────────────────────────────────────────────────────
# 4. Route: /api/top10
# ────────────────────────────────────────────────────────────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt_top10(ui_text)

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"]

        # Remove markdown block like ```json ... ```
        cleaned = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()

        # Parse the JSON and extract
        parsed = json.loads(cleaned)
        return jsonify({"top_10": parsed["top_headers"]})

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": response["message"]["content"]
        }), 500

# Optional root
@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers from Figma UI"})

# ────────────────────────────────────────────────────────────────
# 5. Run app
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running header extractor with strict filters + clean JSON...")
    app.run(debug=True)