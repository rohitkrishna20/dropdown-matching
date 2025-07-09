from flask import Flask, jsonify
from pathlib import Path
import json, re, ollama

app = Flask(__name__)

lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

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
    return list(dict.fromkeys(out))  # de-dupe, preserve order

ui_text = extract_figma_text(lhs_data)

def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing UI text from a Figma sales dashboard.

This dashboard contains one main table with a row of column headers followed by many data rows.

🎯 Your task is to extract **exactly 10 unique column header labels** from the UI.

🔒 Strict Rules:
- ✅ Include only field names used as **column headers**
- ❌ Exclude labels that repeat across multiple rows or appear to represent stages, statuses, or contact types
- ❌ Avoid short or generic words (under 3 characters or vague terms like “Web”, “My”, “Open”, “YoY”)
- ❌ Skip section titles, navigation elements, action triggers, or metadata
- ❌ Do not include similar variations of the same word (e.g. "Closure" and "Expected Closure")
- ❌ Headers should appear once and be semantically distinct from other UI text

🧪 Output format:
Return a **JSON object** exactly like this:
{{
  "header1": "Name",
  "header2": "Account",
  ...
  "header10": "Expected Closure"
}}

No markdown. No explanations. Return only valid JSON.

UI Text
-------
{blob}
""".strip()

@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        # Extract key-value pair headers
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
    print("Running with enhanced filtering and labeled headers…")
    app.run(debug=True)