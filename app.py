from flask import Flask, request, jsonify
from pathlib import Path
import json, re, os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables if using .env
load_dotenv()

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
    return list(dict.fromkeys(out))

ui_text = extract_figma_text(lhs_data)

# ─────────── Prompt Template ───────────
def make_prompt(figma_text):
    return f"""
You are an expert UI parser. You are given raw UI text extracted from a Figma design file for a sales dashboard. Your task is to extract exactly **10 column headers** that are part of a **data table** showing opportunity-level information.

✅ Include only if:
- Short (1–3 words), title-cased, clearly descriptive
- Represents a concrete data attribute like “Win Probability”, “AI Score”, “Expected Closure”
- Not vague (e.g., “Value”, “Date”, “Info”, “Time”)
- Not navigation or section labels (“Leads”, “Quotes”, “Activities”, “Dashboard”)
- Not company/user names or pipeline labels (“Primary”, “Open Leads”)
- Not generic data values (“Email”, “Web”, “Direct Mail”)
- Not process verbs/steps (“Qualify”, “Negotiate”, “Sales Visit”)
- Not longer than 3 words or duplicated
- Not status terms like “Due to Closure”, “At Risk”

📌 Return JSON only like this (no explanation):
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

Raw UI text:
{figma_text}
""".strip()

# ─────────── /api/top10 ───────────
@app.get("/api/top10")
def api_top10():
    try:
        prompt = make_prompt("\n".join(f"- {t}" for t in ui_text))
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        response = llm.invoke(prompt)
        raw = response.content
        print("---- GPT-4 Response ----")
        print(raw)

        parsed = {}
        try:
            raw_clean = raw.replace("“", "\"").replace("”", "\"").strip()
            parsed = json.loads(raw_clean)
        except json.JSONDecodeError:
            json_block = re.search(r"\{[\s\S]*?\}", raw)
            if json_block:
                parsed = json.loads(json_block.group())

        headers = list(parsed.values())
        headers = [h.strip() for h in headers if h.strip()]
        output = {f"header{i+1}": headers[i] if i < len(headers) else "" for i in range(10)}
        return jsonify(output)

    except Exception as e:
        return jsonify({
            "error": "Header extraction failed",
            "details": str(e),
            "raw_response": raw if 'raw' in locals() else "no response"
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
    return FAISS.from_documents(docs, OpenAIEmbeddings())

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
    print("✅ Running LangChain + FAISS + OpenAI Matching App")
    app.run(debug=True)