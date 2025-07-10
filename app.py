from flask import Flask, request, jsonify
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


def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are analyzing raw UI text extracted from a sales dashboard built in Figma.

The dashboard includes a main data table, and your job is to identify the 10 most likely **column headers** in that table.

These column headers represent structured fields (like customer name, status, score, dates, etc.) and are used to label columns in the top row of a table.

🧠 Focus on identifying field names, not row values or UI labels.

Strict Rules:
- ✅ prefer structured field labels that represent system-generated metadata, like warnings
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

🎯 Return ONLY a JSON object with keys "header1" through "header10"

Example:
{{
  "header1": "___",
  "header2": "___",
  "header3": "___",
  ...
  "header10": "___"
}}

Raw UI Text:
------------
{blob}
""".strip()


@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)

    try:
        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
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
            output[key]=corrected[i] if i < len(corrected) else ""

        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500

# ─────────────────────────────────────────────────────
# Additional endpoint: Match fields to headers
# ─────────────────────────────────────────────────────
from pprint import pprint

rhs_path = Path("data/DataRightHS.json")
rhs_data = json.loads(rhs_path.read_text(encoding="utf-8"))

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

def make_match_prompt(headers: list[str], rhs_json: dict) -> str:
    return f"""
You are a helpful assistant performing **semantic field matching** between UI column headers and a structured JSON dataset.

Your task:
For each UI column header, return the **three most semantically relevant field–value pairs** from the JSON data.

🧠 Definitions:
- A "match" is when a JSON field's name and sample value closely relate to the meaning of the header.
- Do NOT rely on string similarity alone — use **semantic/contextual** understanding.
- Avoid repeating the same JSON key for multiple matches unless unavoidable.
- If a field has no valid values (empty list or meaningless data), skip it.

Strict formatting:
Return a valid JSON object like this:
{{
  "Header1": [
    {{ "field": "MatchingFieldName1", "value": "SampleValue1" }},
    {{ "field": "MatchingFieldName2", "value": "SampleValue2" }},
    {{ "field": "MatchingFieldName3", "value": "SampleValue3" }}
  ],
  "Header2": [
    ...
  ],
  ...
}}

✅ Rules:
- Always return **exactly 3** matches per header.
- Do NOT leave any header empty.
- Only include matches that are clearly related in meaning.
- Never output empty string or blank values.
- Use the most representative sample value for each field.

Headers:
{json.dumps(headers, indent=2)}

Data JSON:
{json.dumps(rhs_json, indent=2)[:4000]}  # truncated
""".strip()

@app.post("/api/match_fields")
def api_match_fields():
    try:
        top10_resp = api_top10().json
        headers = list(top10_resp.values())

        prompt = make_match_prompt(headers, rhs_data)

        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        try:
            parsed = json.loads(raw)
        except Exception:
            matches = re.findall(r'"([^"]+)"\s*:\s*\[\s*(.*?)\s*\]', raw, re.DOTALL)
            parsed = {k: re.findall(r'"field"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*"([^"]+)"', v)
                      for k, v in matches}

        return jsonify(parsed)

    except Exception as e:
        return jsonify({
            "error": "Failed to match fields",
            "details": str(e)
        }), 500

@app.get("/")
def home():
    return jsonify({"message": "GET /api/top10 to extract table headers from Figma UI"})

if __name__ == "__main__":
    print("Running with enhanced pattern-based prompt for column header extraction...")
    app.run(debug=True)
