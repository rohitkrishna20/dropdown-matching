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
Strict Rules:
- ✅ Only include structured metadata field names that are likely used as column headers
- ✅ Include only one unique field per row — no duplicates (if "Sales Stage" appears twice, include it only once)
- ❌ Exclude "Status", "Status Indicators", or any term with "Status" — these are not table headers
- ❌ Exclude generic or vague terms like “Value”, “Details”, “Date”, “Time”, or “Indicators”
- ❌ Exclude terms that appear in cells or badges, not headers
- ❌ If a field appears twice (like “Created”), only keep it once
- ❌ Do not include duplicates under different names (e.g., “Sales Stage” twice with slightly different cases)
- Include labels associated with status indicators - especially if they appear at the edge or top of a card or row. 
- ✅ favor labels grouped with known column headers 
- ✅ include unique, structured field labels that appear at the end of table rows or near other headers
- ✅ select labels that appear once or appear early in the visual order as they are likely table header rows
- Do include high-level field labels if they appear alone, early, or near table rows, treat them as structured metadata
- ❌ Exclude value-like terms such as "E-Mail", "Web" - these are row-level values, not column headers
- ✅ Include only descriptive field names used to label table columns
- ❌ Exclude pipeline stages (like “Qualify”, “Negotiation”, “Discovery”)
- ❌ Exclude status badges or alert values (like “Due to closure”, “At Risk”)
- ❌ Exclude anything that sounds like a text style, label category, or design artifact
- ❌ Exclude any header that contains "metadata"
- EXCLUDE any header containing the word "status"
- ❌ Exclude action buttons, tabs, filters, or navigation
- ❌ Do NOT include timestamps or date examples
- ❌ Do NOT include repeated terms like “Opportunity”, “Activity”, “Quote”, “Lead”
- ❌ Do not include duplicate values - every "header" key must have a unique field name. If any name repeats, reject it and pick a new one.
- ❌ Avoid stage-related phrases, alerts, or row-level values such as "status" - DO NOT INCLUDE "STATUS"
- ❌ Exclude anything that looks like data content instead of a label
- ❌ Do NOT include any values that appear inside cells or badges (e.g. “Web”, “Direct Mail”)
- ❌ Exclude entries with names, company references, connectivity types, or network technologies (e.g. “MPLS”, “SAT”, “Connectivity”)
- ❌ Exclude any item containing multiple segments separated by dashes (e.g. "A - B - C") — these are likely data entries, not headers
- ❌ If a phrase contains words like “Group”, “Edge”, “Consulting”, “Solutions”, “Health”, “Global”, or “Services”, exclude it — these are likely business names or customers
- ❌ Exclude terms like “Status”, “Creation Date”, “Date”, or “Time” — these are often metadata rows or timestamps, not true column headers
- ❌ Exclude generic labels like “Value”, “Info”, “Details”, or “Stage” unless part of a specific known column label - do not include any header that contains status

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
