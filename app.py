from flask import Flask, jsonify
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
        if node.get("type") == "TEXT":
            txt = node.get("characters", "").strip()
            if txt and not is_numeric(txt):
                out.append(txt)
        for child in node.get("children", []):
            walk(child)

    walk(figma_json)
    return list(dict.fromkeys(out))          # de-dupe, preserve order

ui_text = extract_figma_text(lhs_data)


def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analysing UI text extracted from a Figma sales dashboard.

🎯 **Return exactly 10 unique column-header labels** (no duplicates / near-duplicates).

❌ Exclude any row values, statuses, timestamps, actions, or navigation text.
❌ Examples to skip: Qualify, Negotiation, Discovery, Sales Visit, At Risk, Due to closure, Timestamp.
✅ Keep only true table headers 

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


@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        m_json = re.search(r"\[\s*\".*?\"\s*(,\s*\".*?\"\s*){1,}\]", raw, re.DOTALL)
        if m_json:
            headers = json.loads(m_json.group(0))
        else:
            headers = re.findall(r"\"([^\"]+)\"", raw)
            headers = headers[:10]

        headers = headers[:10] + [""] * (10 - len(headers))
        return jsonify({"top_10": headers})

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
    print("Running –  …")
    app.run(debug=True)
