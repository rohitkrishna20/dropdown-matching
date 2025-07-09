from flask import Flask, jsonify
from pathlib import Path
import json
import re
import ollama

app = Flask(__name__)

# Load Figma JSON data
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

# Extract visible text from Figma
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
    return list(dict.fromkeys(out))  # remove duplicates but preserve order

ui_text = extract_figma_text(lhs_data)

# Build the prompt for Ollama
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing UI text from a Figma sales dashboard.

This dashboard contains one main table with a row of column headers followed by many data rows.

🎯 Your task is to extract **exactly 10 unique column header labels** from the UI.

🔒 Strict Rules:
- ✅ Include only field names used as **column headers**
- ❌ Exclude anything that looks like a data value or contact method (e.g. emails, stages, statuses, links)
- ❌ Skip short repeated values or vague/overly generic UI labels
- ❌ Ignore navigation labels, dashboard section headers, timestamps, or actions
- ❌ Avoid short one-word entries that could be part of dropdowns or badges

🧪 Format:
Return a **JSON object** with keys "header1" through "header10", like this:
{{
  "header1": "Name",
  "header2": "Account",
  ...
  "header10": "Expected Closure"
}}

No markdown, no commentary. Just return valid JSON.

UI Text
-------
{blob}
""".strip()

# API route to get top 10 headers
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp["message"]["content"]

        # Try structured JSON object format: "header1": "Name"
        headers = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)

        # Fallback: numbered list format: 1. "Name"
        if not headers:
            headers = re.findall(r'\d+\.\s*"([^"]+)"', raw)

        # Fallback: generic JSON list: ["Name", "Account", ...]
        if not headers:
            headers = re.findall(r'"([^"]+)"', raw)

        headers = headers[:10]
        output = {f"header{i+1}": headers[i] if i < len(headers) else "" for i in range(10)}

        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500

# Home route
@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers"})

# Run the app
if __name__ == "__main__":
    print("Running with labeled header output and prompt fallback handling...")
    app.run(debug=True)