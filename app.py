from flask import Flask, jsonify
from pathlib import Path
import json, re, ollama

app = Flask(__name__)

# Load Figma JSON
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

# Extract visible UI text
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
    return list(dict.fromkeys(out))  # de-dupe and preserve order

ui_text = extract_figma_text(lhs_data)

# Prompt function
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing UI text from a Figma sales dashboard.

This dashboard contains one main table with a row of column headers followed by many data rows.

🎯 Your task is to extract **exactly 10 unique column header labels** from the UI.

🔒 Strict Rules:
- ✅ Include only field names used as **column headers**
- ❌ Exclude row-level values like "Qualify", "Negotiation", "Discovery", "At Risk", etc.
- ❌ Exclude stage names, contact methods (like “E-Mail”, “Phone”, “Web”), or pipeline phases
- ❌ Do NOT include time values, user actions (e.g. "Create Quote"), or section headers (e.g. "Dashboard")
- ❌ Skip single words that appear multiple times — headers are usually unique
- ❌ Avoid labels shorter than 3 characters or overly generic words (e.g. "Web", "My", "Open")

🧪 Format:
- Return a JSON list with 10 items using the format: 
  ["header1", "header2", "header3", ..., "header10"]
- Must be valid JSON, no extra text, no markdown

Text from the UI:
------------------
{blob}
""".strip()

# API endpoint
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        # Try to extract JSON from response
        match = re.search(r"\[\s*\".*?\"\s*(,\s*\".*?\"\s*){1,}\]", raw, re.DOTALL)
        if match:
            headers = json.loads(match.group(0))
        else:
            # Fallback to numbered extraction
            headers = re.findall(r'"([^"]+)"', raw)
            headers = headers[:10]

        # Ensure exactly 10 items
        headers = headers[:10] + [""] * (10 - len(headers))

        # Return headers as labeled fields
        return jsonify({f"header{i+1}": h for i, h in enumerate(headers)})

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500

@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers"})

if __name__ == "__main__":
    print("Running with advanced prompt rules…")
    app.run(debug=True)