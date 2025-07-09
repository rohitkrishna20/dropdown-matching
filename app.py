from flask import Flask, jsonify
from pathlib import Path
import json, re, ollama

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. Load Figma JSON
# ─────────────────────────────────────────────
LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

# ─────────────────────────────────────────────
# 2. Extract visible text (skip numeric-only)
# ─────────────────────────────────────────────
def extract_figma_text(figma_json: dict) -> list[str]:
    labels = []

    def _is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def _walk(node: dict):
        if node.get("type") == "TEXT":
            txt = node.get("characters", "").strip()
            if txt and not _is_numeric(txt):
                labels.append(txt)
        for child in node.get("children", []):
            _walk(child)

    _walk(figma_json)
    return list(dict.fromkeys(labels))   # unique, preserve order

ui_text = extract_figma_text(lhs_data)

# ─────────────────────────────────────────────
# 3. Build prompt (excludes bad row values)
# ─────────────────────────────────────────────
def make_prompt_top10(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing UI text extracted from a Figma-based sales dashboard.

This design contains a structured data table with labeled columns.

🎯 Task: Pick the **10 best column headers** from the list below.

❌ Never include:
   - "Qualify"           - "Negotiation" / "Negotation"
   - "Discovery"         - "Sales Visit"
   - "Direct Mail"       - "Timestamp"

❌ Ignore navigation items, action buttons, section titles, and numeric-only strings.

Return **only** a JSON list (exactly 10 items), e.g.:
["___", "___", "___", "___", ..., "name"]

UI Text:
--------
{blob}
""".strip()

# ─────────────────────────────────────────────
# 4. /api/top10 endpoint
# ─────────────────────────────────────────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt_top10(ui_text)

    try:
        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw  = resp["message"]["content"]

        # Grab first JSON array of 10 strings
        match = re.search(r"\[\s*\".*?\"\s*(,\s*\".*?\"\s*){9}\]", raw, re.DOTALL)
        if not match:
            raise ValueError("Could not find a 10-item JSON list in Ollama output.")
        headers = json.loads(match.group(0))       # list[str]

        # ── Deduplicate by keyword overlap ──
        deduped, seen = [], set()
        for label in headers:
            words = set(re.findall(r"\w+", label.lower()))
            if seen & words:           # any overlap? → skip
                continue
            deduped.append(label)
            seen |= words

        # Pad to exactly 10 slots
        while len(deduped) < 10:
            deduped.append("")

        return jsonify({"top_10": deduped})

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "No response"
        }), 500

# ─────────────────────────────────────────────
# 5. Root endpoint
# ─────────────────────────────────────────────
@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers"})

# ─────────────────────────────────────────────
# 6. Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Header-extractor running (prompt filtering + keyword de-dup)…")
    app.run(debug=True)