from flask import Flask, request, jsonify, render_template
import json
from pathlib import Path

app = Flask(__name__)

# ---------- Load Figma JSON (Left-Hand Side) ----------
LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

# ---------- Extract Headers ----------
def extract_figma_headers(figma_json: dict, header_keywords=None):
    headers = []

    def _is_mostly_numeric(text):
        cleaned = text.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def _crawl(node):
        if node.get("type") == "TEXT" and node.get("characters", "").strip():
            txt = node["characters"].strip()
            if _is_mostly_numeric(txt):
                return  # Ignore numeric headers
            if not header_keywords or any(kw.lower() in txt.lower() for kw in header_keywords):
                headers.append(txt)
        for child in node.get("children", []):
            _crawl(child)

    _crawl(figma_json)
    return list(dict.fromkeys(headers))  # Deduplicate, preserve order

figma_headers = extract_figma_headers(lhs_data)

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", headers=figma_headers)

@app.route("/api/headers", methods=["GET"])
def api_headers():
    return jsonify({"headers": figma_headers, "count": len(figma_headers)})

# ---------- Run Server ----------
if __name__ == "__main__":
    print("Running UI-header extractor (Figma only, numeric headers skipped)")
    app.run(debug=True)