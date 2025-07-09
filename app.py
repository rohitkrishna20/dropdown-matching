from flask import Flask, request, jsonify, render_template
import json
from pathlib import Path

app = Flask(__name__)

# ---------- Load Figma JSON and Extract Headers ----------
LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

def extract_figma_headers(figma_json: dict, header_keywords=None):
    headers = []

    def _crawl(node):
        if node.get("type") == "TEXT" and node.get("characters", "").strip():
            txt = node["characters"].strip()
            if not header_keywords or any(kw.lower() in txt.lower() for kw in header_keywords):
                headers.append(txt)
        for child in node.get("children", []):
            _crawl(child)

    _crawl(figma_json)
    return list(dict.fromkeys(headers))  # de-duplicate while preserving order

figma_headers = extract_figma_headers(lhs_data)

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", headers=figma_headers)

@app.route("/api/headers", methods=["GET"])
def api_headers():
    return jsonify({"headers": figma_headers, "count": len(figma_headers)})

if __name__ == "__main__":
    print("Running UI-header viewer (left-hand JSON only)")
    app.run(debug=True)