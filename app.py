from flask import Flask, request, jsonify
from pathlib import Path
import json, re, ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)

# ──────── Load Figma UI JSON ────────
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
    return list(dict.fromkeys(out))  # de-dupe

ui_text = extract_figma_text(lhs_data)

def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are extracting **structured column headers** from raw UI text taken from a sales dashboard in Figma.

The dashboard contains a large data table with records and metadata. Your task is to extract the **10 most likely table column headers** based on the raw text.

🧠 Column headers are short, structured field names shown at the **top of a table column** (e.g., "Name", "AI Score", "Created", "Sales Stage").

They are **NOT**:
- ❌ status messages
- ❌ vague terms like “Value”, “Details”, “Date”, “Status”, “Info”
- ❌ data entries like “Web”, “Email”, “Direct Mail”
- ❌ alert phrases like “Due to closure”, “At Risk”
- ❌ customer or business names like “Health Group”, “Global Edge”
- ❌ anything containing “status”, “connectivity”, “services”, “solution”, “consulting”, etc.
- ❌ duplicate headers (e.g., two versions of “Sales Stage” — keep one only)

✅ Column headers usually appear:
- near the top or left edge of a table
- once per column
- with short, descriptive names like:
  - "Name"
  - "Sales Stage"
  - "Created"
  - "Probability"
  - "AI Score"
  - "Owner"

🎯 Your job is to return 10 unique column headers **only**, using this format:
{{
  "header1": "...",
  "header2": "...",
  ...
  "header10": "..."
}}

DO NOT include:
- status terms
- repeated or blank entries
- vague/generic terms
- row data values
- long company names
- fields with "status", "info", or "value"

Raw UI text:
{blob}
""".strip()

@app.get("/api/top10")
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)
    try:
        resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]

        # Try to parse valid JSON block
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # fallback to regex if JSON fails
            parsed = {}
            matches = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)
            for i, h in enumerate(matches[:10]):
                parsed[f"header{i+1}"] = h.strip()

        # Remove empty or duplicate headers
        seen = set()
        output = {}
        i = 1
        for key in sorted(parsed.keys()):
            val = parsed[key].strip()
            if val and val.lower() not in seen:
                seen.add(val.lower())
                output[f"header{i}"] = val
                i += 1
                if i > 10:
                    break

        # Fill in blanks if less than 10
        for j in range(i, 11):
            output[f"header{j}"] = ""

        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Header extraction failed",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500

# ──────── Load Right-hand Data JSON (DataRightHS.json) ────────
rhs_path = Path("data/DataRightHS.json")
raw_rhs = json.loads(rhs_path.read_text(encoding="utf-8"))

# Unwrap nested "data" key if present
rhs_data = raw_rhs.get("items") if isinstance(raw_rhs, dict) and "items" in raw_rhs else raw_rhs

def build_faiss_index(rhs_data: list[dict]):
    all_fields = set()

    if not isinstance(rhs_data, list):
        raise ValueError("Right-hand data must be a list of dictionaries.")

    for row in rhs_data:
        if isinstance(row, dict):
            for k in row.keys():
                if isinstance(k, str) and k.strip():
                    all_fields.add(k.strip())

    if not all_fields:
        raise ValueError("❌ No field names found in right-hand data.")

    docs = [Document(page_content=field) for field in all_fields]
    embeddings = OllamaEmbeddings(model="llama3.2")
    return FAISS.from_documents(docs, embeddings)

faiss_index = build_faiss_index(rhs_data)

@app.post("/api/match_fields")
def api_match_fields():
    try:
        # Get headers dynamically
        top10 = api_top10()
        if not top10.is_json:
            return jsonify({"error": "Top 10 headers failed"}), 500
        headers = [h for h in top10.get_json().values() if h.strip()]

        out = {}
        for header in headers:
            results = faiss_index.similarity_search(header, k=3)
            matches = [{"field": r.page_content} for r in results]
            out[header] = matches

        return jsonify(out)
    except Exception as e:
        return jsonify({"error": "Semantic field match failed", "details": str(e)}), 500

@app.get("/")
def home():
    return jsonify({"message": "Use /api/top10 to extract headers or /api/match_fields to match fields."})

if __name__ == "__main__":
    print("✅ Running LangChain + FAISS + Ollama Matching App")
    app.run(debug=True)