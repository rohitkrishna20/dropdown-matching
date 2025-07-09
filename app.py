from flask import Flask, jsonify
from pathlib import Path
import json, re, ollama

app = Flask(__name__)

# Load Figma JSON
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

# Extract all visible UI text from the Figma JSON
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
    return list(dict.fromkeys(out))  # de-duplicate, preserve order

ui_text = extract_figma_text(lhs_data)

# Construct prompt for Ollama without hardcoding any specific headers
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing UI text extracted from a Figma sales dashboard.

This dashboard contains one main table with a row of column headers followed by many rows of data.

🎯 Your task is to extract **exactly 10 unique column header labels** from the UI.

🧠 True column headers:
- Usually appear **only once** at the top of a table
- Are typically more descriptive or technical than row-level data
- Label structured fields — not user statuses, navigation, or process steps
- Are not UI actions, repeated tags, timestamps, or badges

🔒 Rules:
- ✅ Include only **labels used as column headers**
- ❌ Skip entries that appear many times across the UI
- ❌ Exclude short words, vague terms, and repeated row-level values
- ❌ Avoid navigation tabs, section titles, button labels, or call-to-action text

🧪 Format:
Return a **valid JSON object** with keys "header1" through "header10", like:
{{
  "header1": "_____",
  "header2": "_____",
  ...
  "header10": "_____"
}}

No markdown, no explanation — just the JSON object.

UI Text
-------
{blob}
""".strip()

# API Endpoint
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

@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers"})

if __name__ == "__main__":
    print("Running with generalized prompt to avoid hardcoded labels…")
    app.run(debug=True)