from flask import Flask, jsonify
from pathlib import Path
import json, re, ollama

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. Load Figma JSON  ➜  data/FigmaLeftHS.json
# ─────────────────────────────────────────────
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

# ─────────────────────────────────────────────
# 2. Pull every visible TEXT node (skip numbers)
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
    return list(dict.fromkeys(out))          # de-dupe, preserve order

ui_text = extract_figma_text(lhs_data)

# ─────────────────────────────────────────────
# 3. Prompt builder (aggressive de-dup wording)
# ─────────────────────────────────────────────
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analysing UI text extracted from a Figma sales dashboard.

🎯 **Return exactly 10 unique column-header labels** (no duplicates / near-duplicates).

❌ Exclude any row values, statuses, timestamps, actions, or navigation text.
❌ Examples to skip: Qualify, Negotiation, Discovery, Sales Visit, At Risk, Due to closure, Timestamp.
✅ Keep only true table headers like Name, Account, AI Score, Created, Source, Expected Closure, Alerts, etc.

**Output format (strict) – choose one of these two options only**  
1. A bare JSON list:  
   `["Name", "Account", "AI Score", "Total Value", …]`  
2. A numbered list where each line is `"Header"` in quotes, e.g.  
   `1. "Name"\n2. "Account"\n…10. "Alerts"`

No commentary, no markdown.

UI Text
-------
{blob}
""".strip()

# ─────────────────────────────────────────────
# 4. /api/top10  – resilient JSON extraction
# ─────────────────────────────────────────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        # A) Try to grab a real JSON list first
        m_json = re.search(r"\[\s*\".*?\"\s*(,\s*\".*?\"\s*){1,}\]", raw, re.DOTALL)
        if m_json:
            headers = json.loads(m_json.group(0))
        else:
            # B) Fallback: pull up to 10 quoted strings (handles 1. "Name" style)
            headers = re.findall(r"\"([^\"]+)\"", raw)
            headers = headers[:10]

        # Ensure exactly 10 items (pad with "")
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
    print("Running – prompt-only filtering, resilient parser …")
    app.run(debug=True)