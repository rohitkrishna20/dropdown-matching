from flask import Flask, jsonify
from pathlib import Path
from collections import Counter
import json, re, ollama

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. Load Figma JSON and extract visible text
# ─────────────────────────────────────────────
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

def extract_figma_text(figma_json: dict) -> list[str]:
    """Return UI text that appears exactly once (likely column headers)."""
    all_strings = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def walk(node: dict):
        if node.get("type") == "TEXT":
            txt = node.get("characters", "").strip()
            if txt and not is_numeric(txt):
                all_strings.append(txt)
        for child in node.get("children", []):
            walk(child)

    walk(figma_json)

    # Keep first occurrence order, but only strings that appear once
    counts = Counter(all_strings)
    dedup_ordered = list(dict.fromkeys(all_strings))
    return [s for s in dedup_ordered if counts[s] == 1]

ui_text = extract_figma_text(lhs_data)

# ─────────────────────────────────────────────
# 2. Prompt builder (no hard-coded headers)
# ─────────────────────────────────────────────
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing UI text extracted from a Figma sales dashboard.

This dashboard contains one main table with a single row of column headers followed by many data rows.

🎯 Extract **exactly 10 unique column-header labels**.

Traits of true column headers
• They appear **only once** in the UI (row values & navigation labels repeat)  
• They label structured data fields, not statuses or steps  
• They are longer than 2 characters and context-specific  
• They are not navigation/menu items, actions, timestamps, or badges

🔒 Rules
• ✅ Return only field labels that fit the traits above  
• ❌ Exclude anything that appears in multiple places (e.g. nav words or statuses)  
• ❌ Skip contact methods, generic terms, or short/ambiguous words  
• ❌ No duplicates or near-duplicates

🧪 Output
Return **only** a valid JSON object with keys "header1" … "header10":

{{
  "header1": "...",
  "header2": "...",
  …
  "header10": "..."
}}

Do NOT add markdown or commentary.

UI text candidates
------------------
{blob}
""".strip()

# ─────────────────────────────────────────────
# 3. /api/top10 endpoint
# ─────────────────────────────────────────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp["message"]["content"]

        # A) Preferred: "headerN": "Value"
        headers = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)

        # B) Fallback: numbered list 1. "Value"
        if not headers:
            headers = re.findall(r'\d+\.\s*"([^"]+)"', raw)

        # C) Fallback: any quoted strings
        if not headers:
            headers = re.findall(r'"([^"]{3,}?)"', raw)

        headers = headers[:10] + [""] * (10 - len(headers))
        return jsonify({f"header{i+1}": h for i, h in enumerate(headers)})

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500

# ─────────────────────────────────────────────
@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers"})

# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Running with unique-label filtering and robust parsing…")
    app.run(debug=True)