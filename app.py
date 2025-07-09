from flask import Flask, jsonify
from pathlib import Path
import json
import ollama
import re

app = Flask(__name__)

LHS_PATH = Path("data/FigmaLeftHS.json")
with LHS_PATH.open(encoding="utf-8") as f:
    lhs_data = json.load(f)

def extract_figma_text(figma_json: dict):
    labels = []

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
    return list(dict.fromkeys(labels))  # keep unique order

ui_text = extract_figma_text(lhs_data)


def make_prompt_top10(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)

    return f"""
You are analyzing UI text extracted from a Figma-based table UI design.

Your task is to identify the 10 most likely column headers shown in this table.

Only include values that are actual **field labels** for structured data columns

Do not include:
- Row values like "Qualify", "Negotiation", "Negotation", or "Discovery"
- Labels such as “Sales Visit”, “Direct Mail”, “Timestamp”
- Section titles such as "Sales Dashboard" or "Overview"
- Tabs, filters, or navigational UI elements
- Numeric-only values or date-like values
- Duplicate or semantically overlapping column labels (e.g. avoid including both "Expected Closure" and "Due to closure")
- Anything with similar names - ex: closure and Closure must not be in the output 
- created on and created as well - ONLY PICK ONE!

Return ONLY a valid JSON list of **10 unique column headers**, like:
["___", "___", "___", "Total", "Expected Closure", ...]

UI Text:
--------
{blob}
""".strip()


@app.get("/api/top10")
def api_top10():
    prompt = make_prompt_top10(ui_text)

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"]

        # Extract JSON list from response using regex
        match = re.search(r"\[\s*\".*?\"\s*(,\s*\".*?\"\s*){1,}\]", raw, re.DOTALL)
        if not match:
            raise ValueError("Could not extract a valid JSON list from Ollama response.")

        header_list = json.loads(match.group(0))
        return jsonify({"top_10": header_list})

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": response["message"]["content"] if 'response' in locals() else "No response"
        }), 500


@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract column headers from Figma UI"})


if __name__ == "__main__":
    print("Running with enhanced prompt filtering duplicates and junk…")
    app.run(debug=True)
