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

🎯 Return ONLY a JSON object with keys "header1" through "header10"

Example:
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
  "header3": "___",
  ...
  "header10": "___"
}}

Raw UI Text:
------------
Extracted UI Text:
{blob}
""".strip()


@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)
