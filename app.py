from flask import Flask, jsonify
from pathlib import Path
import json
import ollama

app = Flask(__name__)

# ────────────────────────────────────────────────────────────────
# 1.  Load Figma JSON (left-hand side design)
# ────────────────────────────────────────────────────────────────
LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

# ────────────────────────────────────────────────────────────────
# 2.  Extract all visible UI text                   (skip numbers)
# ────────────────────────────────────────────────────────────────
def extract_figma_text(figma_json: dict) -> list[str]:
    labels: list[str] = []

    def _is_numeric(text: str) -> bool:
        cleaned = text.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def _walk(node: dict):
        if node.get("type") == "TEXT":
            txt = node.get("characters", "").strip()
            if txt and not _is_numeric(txt):
                labels.append(txt)
        for child in node.get("children", []):
            _walk(child)

    _walk(figma_json)
    return list(dict.fromkeys(labels))          # dedupe, keep order

ui_text = extract_figma_text(lhs_data)

# ────────────────────────────────────────────────────────────────
# 3.  Build the fine-tuned prompt  (no “Focus on” checklist)
# ────────────────────────────────────────────────────────────────
def make_prompt_top10(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)

    return f"""
You are analyzing UI text extracted from a Figma-based sales dashboard.

The design contains a structured **data table** with column headers (like a spreadsheet).

🎯 Task: choose the **10 most likely table column headers** from the list below.

❗ Only include labels that describe table **columns**.  
❌ Do NOT include row values such as sales-stage names (“Negotiation”, “Discovery”, etc.).  
❌ Skip section titles (“Sales Dashboard”, “Overview”), navigation items, action buttons, or numeric counters.

UI text:
--------
{blob}

Return a JSON object exactly in this form:
{{
  "top_headers": ["<header1>", "<header2>", ...]   // 10 items total
}}
""".strip()

# ────────────────────────────────────────────────────────────────
# 4.  API endpoint: /api/top10
# ────────────────────────────────────────────────────────────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt_top10(ui_text)
    resp   = ollama.chat(model="llama3.2",
                         messages=[{"role": "user", "content": prompt}])
    return jsonify({
        "prompt" : prompt,
        "top_10" : resp["message"]["content"]
    })

# optional root
@app.get("/")
def hello():
    return jsonify({"message": "Hit /api/top10 to get the 10 best column headers."})

# ────────────────────────────────────────────────────────────────
# 5.  Run the app
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Ollama-powered header extractor …")
    app.run(debug=True)