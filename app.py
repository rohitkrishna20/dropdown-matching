from flask import Flask, jsonify
from pathlib import Path
from collections import Counter
import json, re, ollama

app = Flask(__name__)

# ─────────── 1. Load Figma JSON ───────────
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

# ─────────── 2. Extract text once-only & >2 chars ───────────
def extract_figma_text(figma_json: dict) -> list[str]:
    strings = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def walk(node: dict):
        if node.get("type") == "TEXT":
            t = node.get("characters", "").strip()
            if t and not is_numeric(t):
                strings.append(t)
        for c in node.get("children", []):
            walk(c)

    walk(figma_json)
    counts = Counter(strings)
    ordered_unique = list(dict.fromkeys(strings))
    # keep length>2 & appears <=1  (nav/status often repeats; row values often short)
    return [s for s in ordered_unique if len(s) > 2 and counts[s] == 1]

ui_text = extract_figma_text(lhs_data)

# ─────────── 3. Prompt builder ───────────
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing UI text extracted from a Figma sales dashboard.

There is one main data table: a single header-row followed by many rows of values.

🎯 Return **exactly 10 unique column-header labels**.

Headers:
• Appear once (row values, nav items, and badges repeat)
• Sit above structured numeric / textual data
• Are not page sections, navigation, buttons, or pipeline **stage names** (e.g. “Qualify”, “Negotiation”, “Discovery”, “At Risk”)

🔒 Exclude anything that is:
• A pipeline/status/stage label
• A navigation or section label (“Overview”, “My To-do's”, “Open opportunities”)
• A short generic word (“Value”, “Primary”) or anything <3 characters
• Repeated elsewhere in the UI text

🧪 Output — return only JSON like:
{{
  "header1": "...",
  ...
  "header10": "..."
}}

No markdown, no commentary.

UI text candidates
------------------
{blob}
""".strip()

# ─────────── 4. /api/top10 endpoint ───────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        # A) proper "headerN": "Value"
        headers = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)
        # B) fallback quoted strings
        if not headers:
            headers = re.findall(r'"\s*([^"]{3,}?)\s*"', raw)

        headers = headers[:10] + [""]*(10-len(headers))
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
    print("Running with stage-name ban & once-only filter …")
    app.run(debug=True)