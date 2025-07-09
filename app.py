from flask import Flask, jsonify
from pathlib import Path
import json, re, ollama

app = Flask(__name__)

# Load Figma JSON
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

# Extract text from Figma
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
    return list(dict.fromkeys(out))  # remove duplicates, preserve order

ui_text = extract_figma_text(lhs_data)

# Construct prompt for Ollama
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing UI text from a Figma-based sales dashboard.

This dashboard contains one main data table, with a row of column headers followed by many rows of values.

🎯 Your task is to extract **exactly 10 unique column header labels** from the UI text.

🔒 Strict Rules:
- ✅ Include only labels used as **table column headers**
- ❌ Exclude values that are used as row-level content, stages, contact methods, or actions
- ❌ Skip overly generic or duplicated values
- ❌ Do NOT include labels shorter than 3 characters

🧪 Format:
Return a valid JSON object using keys `"header1"` through `"header10"`:
{{
  "header1": "Name",
  "header2": "Account",
  ...
  "header10": "Expected Closure"
}}

Return only valid JSON — no markdown, no commentary.

UI Text:
--------
{blob}
""".strip()

# Endpoint for top 10 headers
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        # Try to extract exact JSON format
        headers = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)

        # Fallback: get up to 10 plain quoted strings
        if not headers:
            headers = re.findall(r'"\s*([^"]{3,}?)\s*"', raw)

        # Pad or trim to exactly 10
        headers = headers[:10] + [""] * (10 - len(headers))
        output = {f"header{i+1}": headers[i] for i in range(10)}

        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500

@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers from Figma UI"})

if __name__ == "__main__":
    print("Running with improved prompt, label formatting, and fallback parsing…")
    app.run(debug=True)