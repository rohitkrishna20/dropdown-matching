from flask import Flask, jsonify
from pathlib import Path
import json, re, ollama

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. Load and parse the Figma JSON
# ─────────────────────────────────────────────
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

# ─────────────────────────────────────────────
# 2. Extract all text from visible UI elements
# ─────────────────────────────────────────────
def extract_figma_text(figma_json: dict) -> list[str]:
    out = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def walk(node: dict):
        if node.get("type") == "TEXT":
            txt = node.get("characters", "").strip()
            if txt and not is_numeric(txt):
                out.append(txt)
        for child in node.get("children", []):
            walk(child)

    walk(figma_json)
    return list(dict.fromkeys(out))  # de-dupe while preserving order

ui_text = extract_figma_text(lhs_data)

# ─────────────────────────────────────────────
# 3. Construct Ollama prompt
# ─────────────────────────────────────────────
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)

    return f"""
You are analyzing UI text from a Figma-based dashboard with a large structured table.

Your goal is to extract exactly **10 column header labels** used in this data table.

✅ Only include labels used as headers for structured columns in the table.

🚫 Do NOT include:
• Navigation text, filters, or tabs
• Buttons like "Create Quote"
• Generic short labels (e.g. “Open”, “My”, “Web”)
• Stage/status labels (e.g. “Qualify”, “Negotiation”, “Discovery”, “Leads”, “Opportunities”, “Activities”)
• Timestamps, numeric-only values, or duplicate/near-duplicate terms

⭐ Focus on:
• Labels that appear once
• Appear aligned above multiple structured data rows
• Usually 2–3 words long
• Positioned across a horizontal row in the layout

Return a valid JSON object in this format only:
{{
  "header1": "...",
  "header2": "...",
  ...
  "header10": "..."
}}

Text from the UI:
------------------
{blob}
""".strip()

# ─────────────────────────────────────────────
# 4. API endpoint to fetch top 10 headers
# ─────────────────────────────────────────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        headers = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)
        output = {f"header{i+1}": headers[i] if i < len(headers) else "" for i in range(10)}
        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500

# ─────────────────────────────────────────────
# 5. Home route
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers from Figma UI"})

# ─────────────────────────────────────────────
# 6. Run the app
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🔁 Running Flask app to extract headers using Ollama…")
    app.run(debug=True)