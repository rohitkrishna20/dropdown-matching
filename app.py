from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, os, ollama

app = Flask(__name__)
FEEDBACK_FILE = "feedback.json"

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        return json.loads(open(FEEDBACK_FILE).read())
    return {"good_patterns": {}, "bad_patterns": {}}

def save_feedback(store):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(store, f, indent=2)

feedback_store = load_feedback()

def extract_figma_text(figma_json: dict) -> list[str]:
    out = []
    def is_numeric(t): return t.replace(",", "").replace("%", "").replace("$", "").strip().replace(".", "").isdigit()
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = node.get("characters", "").strip()
                if txt and not is_numeric(txt) and txt.lower() != "text":
                    out.append(txt)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(figma_json)
    return list(dict.fromkeys(out))

def make_prompt(labels, good_patterns=None, bad_patterns=None) -> str:
    blob = "\n".join(f"- {t}" for t in labels)
    base = f"""
You are extracting column headers from a raw Figma-based UI. Focus only on **structured table column headers**.

❌ DO NOT include:
- Status fields or indicators
- Vague terms (like general words for time, value, details, etc.)
- Repeated labels or empty strings
- Anything related to email or contact method
- Company or customer names
- Process stages or pipeline labels
- Words that include "status"
- Data values from inside table rows
- UI section names, menus, or action buttons
- Alerts or warnings
- Dashboard widgets, activity counters
- Any duplicates or empty entries

✅ DO INCLUDE:
- Labels that appear once per column in a table
- Compact and clearly descriptive field names
- Short phrases (1–3 words)
- Likely to appear in the top row of a table
- Structured data field categories (not individual values)
- Not vague, status-based, or action-based
"""
    if good_patterns:
        base += f"\nPrioritize patterns like: {', '.join(good_patterns)}"
    if bad_patterns:
        base += f"\nAvoid patterns like: {', '.join(bad_patterns)}"

    return base + f"\n\nRaw UI text:\n{blob}"

def build_faiss_index(rhs_data: list[dict]):
    fields = set()
    for row in rhs_data:
        if isinstance(row, dict):
            for k in row:
                if k and isinstance(k, str):
                    fields.add(k.strip())
    docs = [Document(page_content=field) for field in fields]
    return FAISS.from_documents(docs, OllamaEmbeddings(model="llama3.2"))

def extract_patterns_for_header(header: str) -> list:
    prompt = f"What patterns, keywords, or visual cues might have caused you to extract the header '{header}'?"
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    text = response["message"]["content"]
    patterns = []
    for line in text.splitlines():
        if "-" in line:
            parts = line.split("-", 1)
            pattern = parts[-1].strip()
            if pattern:
                patterns.append(pattern)
    return patterns

@app.post("/api/find_fields")
def api_find_fields():
    try:
        body = request.get_json(force=True)
        if isinstance(body, str):
            body = json.loads(body)

        # 🔁 CHANGED THIS BLOCK ONLY:
        figma_str = body["figma_json"]
        data_str = body["data_json"]

        figma_json = json.loads(figma_str)
        data_json = json.loads(data_str)

        rhs_items = data_json["items"] if isinstance(data_json, dict) and "items" in data_json else (
            data_json if isinstance(data_json, list) else [data_json]
        )

        good = [p for plist in feedback_store["good_patterns"].values() for p in plist]
        bad = [p for plist in feedback_store["bad_patterns"].values() for p in plist]

        figma_text = extract_figma_text(figma_json)
        prompt = make_prompt(figma_text, good_patterns=good, bad_patterns=bad)
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        raw_response = response["message"]["content"]

        try:
            parsed_headers = json.loads(raw_response)
        except:
            match = re.search(r"\{[\s\S]*?\}", raw_response)
            parsed_headers = json.loads(match.group()) if match else {}

        headers = list(parsed_headers.keys())

        index = build_faiss_index(rhs_items)
        matches = {}
        for header in headers:
            results = index.similarity_search(header, k=5)
            matches[header] = [{"field": r.page_content} for r in results]

        return jsonify({"headers_extracted": headers, "matches": matches})

    except Exception as e:
        return jsonify({"error": "Find fields failed", "details": str(e)}), 500

@app.post("/api/header_feedback")
def header_feedback():
    try:
        body = request.get_json(force=True)
        if isinstance(body, str):
            body = json.loads(body)

        header = body.get("header")
        is_correct = body.get("is_correct")
        if not header or is_correct not in [True, False]:
            return jsonify({"error": "Missing 'header' or 'is_correct'"}), 400

        patterns = extract_patterns_for_header(header)
        if is_correct:
            feedback_store["good_patterns"][header] = patterns
        else:
            feedback_store["bad_patterns"][header] = patterns

        save_feedback(feedback_store)

        return jsonify({
            "header": header,
            "patterns_extracted": patterns,
            "stored_as": "good" if is_correct else "bad"
        })

    except Exception as e:
        return jsonify({"error": "Feedback processing failed", "details": str(e)}), 500

@app.get("/")
def home():
    return jsonify({"message": "Use /api/find_fields or /api/header_feedback"})

if __name__ == "__main__":
    print("✅ API running at http://localhost:5000")
    app.run(debug=True)