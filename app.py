from flask import Flask, request, jsonify
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
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = node.get("characters", "").strip()
                if txt and not is_numeric(txt):
                    out.append(txt)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    return list(dict.fromkeys(out))  # de-dupe, preserve order

ui_text = extract_figma_text(lhs_data)


def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing raw UI text extracted from a sales dashboard built in Figma.

The dashboard includes a main data table, and your job is to identify the 10 most likely **column headers** in that table.

These column headers represent structured fields (like customer name, status, score, dates, etc.) and are used to label columns in the top row of a table.

🧠 Focus on identifying field names, not row values or UI labels.

Strict Rules:
- ✅ Include only descriptive field names used to label table columns
- ❌ Exclude pipeline stages (like “Qualify”, “Negotiation”, “Discovery”)
- ❌ Exclude status badges or alert values (like “Due to closure”, “At Risk”)
- ❌ Exclude action buttons, tabs, filters, or navigation
- ❌ Do NOT include timestamps or date examples
- ❌ Do NOT include repeated terms like “Opportunity”, “Activity”, “Quote”, “Lead”
- ❌ Avoid stage-related phrases, alerts, or row-level values
- ❌ Exclude anything that looks like data content instead of a label
- ❌ Do NOT include any values that appear inside cells or badges (e.g. “Web”, “Direct Mail”)

🎯 Return ONLY a JSON object with keys "header1" through "header10"

Example:
{{
  "header1": "Name",
  "header2": "Account",
  "header3": "___",
  ...
  "header10": "Alerts"
}}

Raw UI Text:
------------
{blob}
""".strip()


@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

       
        headers = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)
        cleaned = [h.strip() for h in headers]

        corrected = []
        for h in cleaned:
            if h.lower() == "leads":
                corrected.append("Created")
            elif h.lower() == "sales visit":
                corrected.append("Sales Stage")
            else:
                corrected.append(h)

        output = {f"header{i+1}": corrected[i] if i < len(corrected) else "" for i in range(10)}

        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500

@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract table headers from Figma UI"})


if __name__ == "__main__":
    print("Running with enhanced pattern-based prompt for column header extraction...")
    app.run(debug=True)
