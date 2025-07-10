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

def make_prompt_for_top10_headers(text_chunks: list[str]) -> str:
    flat_text = "\n".join(text_chunks)
    return f"""
You are an expert UI parser. Your task is to extract only the most likely **table column headers** from a raw UI text dump.

### Very Strict Instructions:
- Only include **headers** that represent columns in a table, such as "Name", "Account", or "AI Score".
- Do NOT include actual data entries, such as "Titan Edge" or "Momentum Group".
- Do NOT include any label if the data field underneath it is **blank, missing, or empty**.
- Do NOT include things like contact methods (e.g., “Web”), dates (unless used as headers), alert flags (e.g., “At Risk”), or generic text (e.g., “Open”, “Overview”).
- Prefer labels that are short (1–3 words) and commonly used to categorize rows of structured data.
- These headers usually appear once per column and are followed by multiple values.
- Do not include duplicates or empty items.
- Output only a list of the top 10 most likely column headers.

### Output Format (strict JSON list of strings):
[
  "Header 1",
  "Header 2",
  ...
]

### Raw UI Text:
{flat_text}
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
            output[key] = corrected[i] if i < len(corrected) else ""

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
You are an AI assistant matching column headers from a UI to field–value pairs from a JSON data dictionary.

🎯 Task:
For each UI column header, return **exactly 3 semantically related** field–value pairs from the data JSON.

✅ Guidelines:
- Use contextual/semantic meaning — not just string overlap.
- Skip empty, null, or meaningless values (like "" or "N/A").
- You must return 3 relevant matches per header, even if imperfect.
- Do NOT reuse the same key unless unavoidable.

🚫 Never include:
- Empty lists or values
- Repeated field names
- Placeholder terms or generic keys

📦 Output format:
You must only use the headers provided — do not invent new keys. Return:
{{
  "Header1": [
    {{ "field": "FieldName1", "value": "Example 1" }},
    {{ "field": "FieldName2", "value": "Example 2" }},
    {{ "field": "FieldName3", "value": "Example 3" }}
  ],
  ...
}}

Headers:
{json.dumps(headers, indent=2)}

Data:
{json.dumps(rhs_json, indent=2)[:3500]}
""".strip()

@app.post("/api/match_fields")
def api_match_fields():
    try:
        top10_resp = api_top10().json
        headers = [v for k, v in top10_resp.items() if v]

        cleaned_data = filter_non_empty_fields(rhs_data)
        prompt = make_match_prompt(headers, cleaned_data)

        resp = ollama.chat(model="llama3.2",
                           messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        try:
            parsed = json.loads(raw)
        except Exception:
            matches = re.findall(r'"([^"]+)"\s*:\s*\[\s*(.*?)\s*\]', raw, re.DOTALL)
            parsed = {k: re.findall(r'"field"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*"([^"]+)"', v)
                      for k, v in matches}

        result = {header: parsed.get(header, []) for header in headers}
        return jsonify(result)

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