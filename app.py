from flask import Flask, jsonify
from pathlib import Path
import json
import ollama
import re

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. Load Figma JSON
# ─────────────────────────────────────────────
LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

# ─────────────────────────────────────────────
# 2. Extract visible text (skip numeric-only)
# ─────────────────────────────────────────────
def extract_figma_text(figma_json: dict):
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
    return list(dict.fromkeys(labels))

ui_text = extract_figma_text(lhs_data)

# ─────────────────────────────────────────────
# 3. Build prompt with built-in exclusions
# ─────────────────────────────────────────────
def make_prompt_top10(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)

    return f"""
You are analyzing UI text extracted from a Figma-based sales dashboard.

The design includes a structured data table with labeled columns.

🎯 Task: Choose the **10 most likely table column headers** from the list below.

These headers should represent structured, high-level fields like:
- Name, Account

❌ Do NOT include any of the following row values or unrelated items:
- "Qualify"
- "Qualified"
- "Negotiation"
- "Negotation"
- "Discovery"
- "Sales Visit"
- "Direct Mail"
- "Timestamp"

❌ Also ignore:
- Navigation items, action buttons, and section titles (e.g. “Overview”)
- Numeric-only strings (like “11”, “13”, “2”)

✅ Return only the most semantically meaningful **column names**.

Output format (strict):
-----------------------
A single JSON list, exactly 10 items, like this:

["Name", "Account", "Created", "Total Value", "AI Score", ..., "Alerts"]

UI Text:
--------
{blob}
""".strip()

# ─────────────────────────────────────────────
# 4. /api/top10 endpoint
# ─────────────────────────────────────────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt_top10(ui_text)

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"]

        # Extract the first JSON array (["..."]) from the text using regex
        match = re.search(r"\[\s*\".*?\"\s*(,\s*\".*?\"\s*){9}\]", raw, re.DOTALL)
        if not match:
            raise ValueError("Could not extract a 10-item list from Ollama response.")
        
        header_list = json.loads(match.group(0))

        return jsonify({"top_10": header_list})

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": response["message"]["content"] if 'response' in locals() else "No response"
        }), 500

# ─────────────────────────────────────────────
# 5. Root endpoint
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers from Figma UI"})

# ─────────────────────────────────────────────
# 6. Run the app
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running Ollama-powered header extractor (prompt-based filtering only)…")
    app.run(debug=True)