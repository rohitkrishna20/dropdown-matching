from flask import Flask, request, jsonify
from pathlib import Path
import json, re, ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

app = Flask(__name__)

# ─────────── Load UI (Left-hand Side) ───────────
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

def extract_figma_text(figma_json: dict) -> list[str]:
    out = []
    
    def is_likely_header(txt: str) -> bool:
        # Exclude values with special characters, numbers, or too long
        return (
            txt
            and txt[0].isupper()
            and len(txt.split()) <= 3
            and re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)*$', txt)
            and not any(c in txt for c in "-@%/:()[]0123456789")
        )


    def is_numeric(t: str) -> bool:
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def walk(node: dict):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = node.get("characters", "").strip()
                if txt and not is_numeric(txt) and txt.lower() != "text":
                    out.append(txt)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    return list(dict.fromkeys(out))  # preserve order, remove duplicates

ui_text = extract_figma_text(lhs_data)

# ─────────── Prompt Template ───────────
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
def make_prompt_for_headers(figma_text):
    return f"""
You are analyzing UI text extracted from a Figma design file for a Sales dashboard table. Your task is to identify the **10 most likely column headers** shown in a **data table** within the UI. Focus only on items that:
- Are concise, capitalized, and descriptive
- Represent unique opportunity-related attributes (e.g., 'AI Score', 'Expected Closure')
- Are **not** vague (like 'Value', 'Time', 'Info')
- Are **not** navigation elements (like 'Leads', 'Quotes', 'Opportunities')
- Are **not** generic data values (like 'Email', 'Web', 'Direct mail')
- Are **not** pipeline statuses (like 'Open Leads', 'Primary', 'At Risk')
- Are **not** full objects (like 'Activities', 'Quotations')

Return only 10 column headers **exactly** as they appear in the UI. Do not fabricate or generalize. Do not add explanation.

Here is the raw text extracted from the Figma UI:
{figma_text}

Respond in this strict JSON format only:
{{
  "header1": "...",
  "header2": "...",
  "header3": "...",
  "header4": "...",
  "header5": "...",
  "header6": "...",
  "header7": "...",
  "header8": "...",
  "header9": "...",
  "header10": "..."
}}
"""
Raw UI text:
{blob}
""".strip()

# ─────────── /api/top10 ───────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)
    try:
        resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]
        print("---- Ollama Raw Response ----")
        print(raw)

        # Try JSON parsing
        parsed = {}
try:
    raw_clean = raw.replace("“", "\"").replace("”", "\"").strip()
    parsed = json.loads(raw_clean)
except json.JSONDecodeError:
    json_block = re.search(r"\{[\s\S]*?\}", raw)
    if json_block:
        parsed = json.loads(json_block.group())

        # Convert keys to header1, header2, ...
        headers = list(parsed.keys())
        clean = [h.strip() for h in headers if h.strip()]

        output = {f"header{i+1}": clean[i] for i in range(min(10, len(clean)))}
        for i in range(len(clean), 10):
            output[f"header{i+1}"] = ""

        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Header extraction failed",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500
# ─────────── Load RHS JSON ───────────
rhs_path = Path("data/DataRightHS.json")
raw_rhs = json.loads(rhs_path.read_text(encoding="utf-8"))
rhs_data = raw_rhs.get("items") if isinstance(raw_rhs, dict) and "items" in raw_rhs else raw_rhs

# ─────────── FAISS Index Builder ───────────
def build_faiss_index(rhs_data: list[dict]):
    fields = set()
    for row in rhs_data:
        if isinstance(row, dict):
            for k in row:
                if k and isinstance(k, str):
                    fields.add(k.strip())
    docs = [Document(page_content=field) for field in fields]
    return FAISS.from_documents(docs, OllamaEmbeddings(model="llama3.2"))

faiss_index = build_faiss_index(rhs_data)

# ─────────── /api/match_fields ───────────
@app.post("/api/match_fields")
def api_match_fields():
    try:
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
    return jsonify({"message": "Use /api/top10 or /api/match_fields"})

if __name__ == "__main__":
    print("✅ Running LangChain + FAISS + Ollama Matching App")
    app.run(debug=True)