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
You are extracting column headers from a raw Figma-based UI. Focus only on **structured table column headers**.

❌ DO NOT include:
- status fields (like “Status”, “At Risk”, “Status Indicators”)
- vague terms (“Date”, “Value”, “Info”, “Details”, “Time”)
- repeated labels
- empty strings
- data values (like “Web”, “Direct Mail”, “Email”)
- company names (like “Edge Consulting” or “Health Group”)
- pipeline stages (“Negotiation”, “Discovery”)
- anything that includes the word “status”

✅ DO INCLUDE:
- field names that appear once near the top of a table
- typical column headers like “Name”, “Owner”, “Created”, “Sales Stage”

Return a JSON like this:
{{
  "header1": "...",
  "header2": "...",
  ...
  "header10": "..."
}}

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
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract dict body manually if wrapped in explanation
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