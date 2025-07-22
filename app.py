from flask import Flask, request, jsonify
from pathlib import Path
import json, re, ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

app = Flask(__name__)

# ──────── Load Figma UI JSON ────────
lhs_path = Path("data/FigmaLeftHS.json")
lhs_data = json.loads(lhs_path.read_text(encoding="utf-8"))

def extract_figma_text(figma_json: dict) -> list[str]:
    out = []

    def is_likely_header(txt: str) -> bool:
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
                if txt and not is_numeric(txt) and is_likely_header(txt):
                    out.append(txt)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    return list(dict.fromkeys(out))

ui_text = extract_figma_text(lhs_data)

# ──────── Strong Prompt ────────
def make_prompt(labels: list[str]) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    return f"""
You are an expert in user interface parsing.

You have received raw UI text from a **Figma-based sales dashboard**. Your task is to extract the **10 most likely column headers** that label structured fields in a data table.

🧠 Column headers are short, capitalized field names at the **top of a table**. They describe what kind of data appears in each row (like account name, sales stage, score, dates, etc.).

❌ Do NOT include:
- Values or data entries (like “Negotiation”, “Web”, “Closed”, “MPLS”, or “Titan Edge”)
- Anything containing special characters like dashes, colons, slashes, or numbers
- Business names, company references, or long compound names
- Terms with lowercase-only letters, ALL CAPS, or generic labels like “Info”, “Details”, “Value”
- Duplicate entries or headers containing “status”, “indicator”, or “alert”

✅ DO include:
- Only short, capitalized, clean terms (1–3 words max)
- Labels that likely appear as the **top row in a data table**
- Unique, structured field names that describe each column

Return exactly 10 column headers in this strict JSON format:

{{
  "header1": "...",
  "header2": "...",
  ...
  "header10": "..."
}}

Raw UI text:
{blob}
""".strip()

# ──────── /api/top10 ────────
@app.get("/api/top10")
def api_top10():
    prompt = make_prompt(ui_text)
    print("\n---- Ollama Raw Prompt ----\n", prompt)

    try:
        resp = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw = resp["message"]["content"]
        print("\n---- Ollama Raw Response ----\n", raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
            matches = re.findall(r'"header\d+"\s*:\s*"([^"]+)"', raw)
            for i, h in enumerate(matches[:10]):
                parsed[f"header{i+1}"] = h.strip()

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

        for j in range(i, 11):
            output[f"header{j}"] = ""

        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Header extraction failed",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500

# ──────── RHS Matching ────────
rhs_path = Path("data/DataRightHS.json")
raw_rhs = json.loads(rhs_path.read_text(encoding="utf-8"))
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