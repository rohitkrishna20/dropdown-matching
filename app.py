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
- ✅ Only include structured metadata field names that are likely used as column headers
- ✅ Include only one unique field per row — no duplicates (if "Sales Stage" appears twice, include it only once)
- A label that names the sales opportunity
- A label for the customer/account/business entity
- ✅ include unique, structured field labels that appear at the end of table rows or near other headers
- ✅ select labels that appear once or appear early in the visual order as they are likely table header rows
- Do include high-level field labels if they appear alone, early, or near table rows, treat them as structured metadata
- ✅ Include only descriptive field names used to label table columns
- ❌ Exclude value-like terms such as "E-Mail", "Web" - these are row-level values, not column headers
- ❌ NEVER INCLUDE "Status", "Status Indicators", or any term with "Status" — these are not table headers
- ❌ Exclude generic or vague terms like “Value”, “Details”, “Date”, “Time”, or “Indicators”
- ❌ Exclude terms that appear in cells or badges, not headers
- ❌ If a field appears twice (like “Created”), only keep it once
- ❌ Do not include duplicates under different names (e.g., “Sales Stage” twice with slightly different cases)
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
- Once again make sure to ignore any "status" headers and never have two of the same headers outputted at the same time for ex. (sales stage and sales stage) should never both be outputted

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
- ✅ Always:
- Return **exactly 3 distinct field–value objects per header**
- Even if the match is imperfect, choose the closest semantic fit
- Use only non-empty values from the JSON
- Format the output as strict JSON only — no commentary or explanation
- Each field must have a meaningful, non-empty value from the data
- If the field is not an exact match, choose the best semantic alternatives
🎯 Priority Guidelines:
1. Match based on **meaning**, not name (e.g., “Created” might match “createdOn”)
2. Prefer fields with clear values (e.g., “2025-03-17”, “Web”, “Sales visit”)
3. Avoid dummy/placeholder/empty strings

Headers:
{json.dumps(headers, indent=2)}

Data JSON:
{json.dumps(rhs_json, indent=2)[:4000]}  # truncated
""".strip()

@app.post("/api/match_fields")
def api_match_fields():
    try:
        # Get headers from /api/top10
        top10_response = api_top10()
        if not top10_response.is_json:
            return jsonify({"error": "Top 10 headers response is not JSON"}), 500

        headers = list(top10_response.get_json().values())
        headers = [h for h in headers if h.strip()]

        # Collect all unique field names from rhs_data
        field_names = set()
        for record in rhs_data:
            if isinstance(record, dict):
                field_names.update(record.keys())
        field_names_only = list(field_names)

        # Prompt Ollama
# Build a pool of all non-empty field-value pairs across records
        all_field_value_pool = {}
        for record in rhs_data:
            if isinstance(record, dict):
                for key, val in record.items():
                    if isinstance(val, str) and val.strip():
                        all_field_value_pool[key] = val.strip()
        
        # Call Ollama for top 3 semantic field matches
        match_prompt = make_match_prompt(headers, rhs_data)
        resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": match_prompt}])
        raw = resp["message"]["content"]
        
        # Try parsing the response as JSON
        try:
            field_only_result = json.loads(raw)
        except Exception:
            field_only_result = {}
            matches = re.findall(r'"([^"]+)"\s*:\s*\[\s*(.*?)\s*\]', raw, re.DOTALL)
            for header, block in matches:
                fields = re.findall(r'"field"\s*:\s*"([^"]+)"', block)
                field_only_result[header] = [{"field": f} for f in fields[:3]]

# Final enrichment: attach values from rhs_data
        final_output = {}
        
        for header, items in field_only_result.items():
            enriched = []
            used_fields = set()
        
            for item in items:
                field = item.get("field")
                value = "[empty]"
        
                for record in rhs_data:
                    if isinstance(record, dict) and field in record:
                        temp = record[field]
                        if isinstance(temp, str) and temp.strip():
                            value = temp.strip()
                            break
                        elif isinstance(temp, (list, dict)) and temp:
                            value = json.dumps(temp)
                            break
        
                enriched.append({"field": field, "value": value})
                used_fields.add(field)
        
            # Pad with any non-empty fields if less than 3
            if len(enriched) < 3:
                for field, value in all_field_value_pool.items():
                    if field not in used_fields:
                        enriched.append({"field": field, "value": value})
                        used_fields.add(field)
                    if len(enriched) == 3:
                        break
        
            final_output[header] = enriched
        
        return jsonify(final_output)
            except Exception as e:
                return jsonify({
                    "error": "Failed to match fields",
                    "details": str(e)
                }), 500
def home():
    return jsonify({"message": "GET /api/top10 to extract table headers from Figma UI"})

if __name__ == "__main__":
    print("Running with enhanced pattern-based prompt for column header extraction...")
    app.run(debug=True)
