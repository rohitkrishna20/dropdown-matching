from flask import Flask, jsonify
from pathlib import Path
import json, re, ollama

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. Load the left-hand Figma JSON
# ─────────────────────────────────────────────
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

# ─────────────────────────────────────────────
# 2. Extract every visible TEXT label (skip numbers)
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
    return list(dict.fromkeys(out))      # de-dupe, preserve order

ui_text = extract_figma_text(lhs_data)

# ─────────────────────────────────────────────
# 3. Prompt builder  (no hard-coded headers)
# ─────────────────────────────────────────────
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)

    return f"""
You are analyzing UI text extracted from a Figma-based sales dashboard.

The dashboard has **one data table**: a single row of column headers followed by many data rows.

🎯 Task: Return **exactly 10 unique column-header labels**.

Rules
-----
- ✅ Only include text that would be a structured data field name (column header).
- ❌ Exclude row values, statuses, timestamps, buttons, tabs, and section titles.
- ❌ Skip navigation or action text such as “Sales Dashboard”, “Overview”, “Quotes”, “Orders”.
- ❌ Skip row labels like Qualify, Negotiation, Discovery, Sales Visit, Due to closure, At Risk, Timestamp.
- ❌ No duplicates or semantic near-duplicates (e.g. “Expected Closure” vs “Due to closure”; “Created” vs “Created on”).
- ✅ Choose labels that are **short, singular, and appear only once** in the UI — like real headers.

Output
------
Return a valid JSON list of **exactly 10 strings**, no markdown, no prose.  
Example:  
["Header 1", "Header 2", ..., "Header 10"]

UI Text List
------------
{blob}
""".strip()

# ─────────────────────────────────────────────
# 4. /api/top10 – robust extractor
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

        # A) Try to capture a real JSON list first
        m_json = re.search(r"\[\s*\".*?\"\s*(,\s*\".*?\"\s*){1,}\]", raw, re.DOTALL)
        if m_json:
            headers = json.loads(m_json.group(0))
        else:
            # B) Fallback: grab quoted strings from numbered / bullet lists
            headers = re.findall(r"\"([^\"]+)\"", raw)[:10]

        # Guarantee exactly 10 slots
        headers = headers[:10] + [""] * (10 - len(headers))
        return jsonify({"top_10": headers})

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
    print("Running – prompt-only filtering with stronger header instructions …")
    app.run(debug=True)