from flask import Flask, jsonify
from pathlib import Path
import json
import ollama
import re

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load Figma JSON
# ─────────────────────────────────────────────
LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

# ─────────────────────────────────────────────
# Extract visible Figma text
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
    return list(dict.fromkeys(labels))  # maintain order and uniqueness

ui_text = extract_figma_text(lhs_data)

# ─────────────────────────────────────────────
# Aggressive prompt to enforce clean top-10 headers
# ─────────────────────────────────────────────
def make_prompt_top10(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)

    return f"""
You are analyzing extracted UI text from a Figma-based **sales dashboard** showing a structured data table.

🎯 Your goal: Choose the **10 most likely column headers** used in this table.

✅ Only include text that represents **actual column labels**


🚫 Absolutely do NOT include:
- Row values like: Qualify, Negotiation, Discovery, At Risk, Due to closure
- Status terms: Won, Lost, Closed, In Progress
- Activities or visit types: Sales Visit, Email, Direct Mail
- Section titles or UI tabs: Sales Dashboard, Overview, Quotes, Orders
- Time-based labels like: Timestamp, Created on, Created at
- ❗ Any duplicate or semantically similar terms — include **only one version**.
  (e.g. If "Expected Closure" is used, do NOT include "Due to closure")

❗ Do NOT include both lowercase and title case variants of the same field (e.g. "closure" vs "Closure")

🎯 Output:
A clean, valid **JSON list of exactly 10 unique column headers**, like:
["___", "___", "___", "___", "___", ...]

UI Text:
--------
{blob}
""".strip()

# ─────────────────────────────────────────────
# API Endpoint
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

        # Extract the first valid JSON array using regex
        match = re.search(r"\[\s*\".*?\"\s*(,\s*\".*?\"\s*){1,}\]", raw, re.DOTALL)
        if not match:
            raise ValueError("Could not extract a valid JSON list from Ollama response.")

        header_list = json.loads(match.group(0))
        return jsonify({"top_10": header_list})

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": response["message"]["content"] if 'response' in locals() else "No response"
        }), 500

# ─────────────────────────────────────────────
# Root Route
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers from Figma UI"})

# ─────────────────────────────────────────────
# Run App
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running with strong duplicate filtering via prompt…")
    app.run(debug=True)