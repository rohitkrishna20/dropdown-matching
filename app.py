from flask import Flask, request, jsonify, Response
from pathlib import Path
import json, re, ollama

from typing import List

app = Flask(__name__)

# ──────────────── Load static JSON data ────────────────
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

rhs_path = Path("data/DataRightHS.json")
rhs_data = json.loads(rhs_path.read_text(encoding="utf-8"))


# ──────────────── Helpers ────────────────
def extract_figma_text(figma_json: dict) -> List[str]:
    out = []

    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def walk(node: dict):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = node.get("characters", "").strip()
                if txt and not is_numeric(txt):
                    out.append(txt)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    return list(dict.fromkeys(out))  # de-dupe, preserve order


ui_text = extract_figma_text(lhs_data)


def make_prompt(labels: List[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are identifying column headers from raw Figma UI text extracted from a sales dashboard table.

🧠 Your task:
Return exactly 10 field names most likely used as table **column headers** (structured metadata). These should NOT be row values, buttons, filters, or status indicators.

✅ Include:
- Only structured metadata used as column labels
- Unique, descriptive terms (no repeats or vague labels)
- Labels likely found at the top of table rows

❌ Exclude:
- Anything with the word "status", "metadata", "value", "info", "details", "date", or "time"
- Terms from badges, cells, pipelines, or labels like "Qualify", "Negotiation"
- Entries with company names, business terms (e.g., “Consulting”, “Solutions”)
- Action items, styles, navigation, timestamps, stages, alerts, or vague terms

📦 Output format:
Return only a JSON object:
{{
  "header1": "___",
  "header2": "___",
  ...
  "header10": "___"
}}

Extracted UI Text:
{blob}
""".strip()


def filter_non_empty_fields(data: dict) -> dict:
    filtered = {}
    for key, val in data.items():
        if isinstance(val, str) and val.strip():
            filtered[key] = val
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    filtered[key] = val
                    break
                elif isinstance(item, list) and item:
                    filtered[key] = val
                    break
                elif isinstance(item, dict) and any(item.values()):
                    filtered[key] = val
                    break
    return filtered


def make_match_prompt(headers: List[str], rhs_json: dict) -> str:
    return f"""
You are an AI assistant matching column headers from a UI to field-value pairs from a JSON data dictionary.

Your goal: For each header, select the **three most semantically related fields** from the JSON, using BOTH the field name and a meaningful sample value.

🧠 Match rules:
- Do NOT rely on string overlap alone — match based on meaning, context, and intent.
- Fields with empty values should be skipped, but always find 3 of the best available.
- You must still return 3 results even if the match is imperfect — prioritize **closeness of meaning**.
- Avoid exact duplicate field names unless they provide different values.
- If a header is vague or broad, choose the **most likely candidates** that relate to it in a business/sales context.

🚫 Never return:
- Empty lists
- Empty strings
- Placeholder values (e.g. "N/A", "null", or "Metadata")
- Repeat fields with the same sample value

✅ Always:
- Return exactly 3 distinct field–value objects per header
- Include a representative value for each
- Pick the most meaningful and populated data fields, not just string matches

Output format (strict JSON):
{{
  "Header1": [
    {{ "field": "FieldName1", "value": "Example value 1" }},
    {{ "field": "FieldName2", "value": "Example value 2" }},
    {{ "field": "FieldName3", "value": "Example value 3" }}
  ],
  ...
}}

Headers:
{json.dumps(headers, indent=2)}

Data JSON (cleaned and partial):
{json.dumps(rhs_json, indent=2)[:3500]}
""".strip()


# ──────────────── API Routes ────────────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)
    try:
        resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        if not resp or "message" not in resp or "content" not in resp["message"]:
            raise ValueError("Missing content in Ollama response")

        raw = resp["message"]["content"]

        headers = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)
        cleaned = [h.strip() for h in headers]

        corrected = []
        for h in cleaned:
            if h.lower() == "leads":
                corrected.append("Created")
            elif h.lower() == "sales visit":
                corrected.append("Sales Stage")
            else:
                corrected.append(h)

        output = {}
        for i in range(10):
            key = f"header{i+1}"
            output[key] = corrected[i] if i < len(corrected) else ""

        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500


@app.post("/api/match_fields")
def api_match_fields():
    try:
        # Get headers via internal function call and parse the response correctly
        with app.test_client() as client:
            top10_response = client.get("/api/top10")
            headers_json = json.loads(top10_response.get_data(as_text=True))
            headers = list(headers_json.values())

        cleaned_rhs = filter_non_empty_fields(rhs_data)
        prompt = make_match_prompt(headers, cleaned_rhs)

        resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        try:
            parsed = json.loads(raw)
        except Exception:
            matches = re.findall(r'"([^"]+)"\s*:\s*\[\s*(.*?)\s*\]', raw, re.DOTALL)
            parsed = {
                k: [{"field": f, "value": v} for f, v in re.findall(r'"field"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*"([^"]+)"', v)]
                for k, v in matches
            }

        return jsonify(parsed)

    except Exception as e:
        return jsonify({
            "error": "Failed to match fields",
            "details": str(e)
        }), 500


@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract table headers from Figma UI"})


# ──────────────── App Entrypoint ────────────────
if __name__ == "__main__":
    print("Running with enhanced pattern-based prompt for column header extraction...")
    app.run(debug=True)