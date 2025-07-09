from flask import Flask, jsonify
from pathlib import Path
import json, re, ollama
from collections import Counter

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. Load Figma JSON once
# ─────────────────────────────────────────────
FIGMA_JSON = Path("data/FigmaLeftHS.json").read_text(encoding="utf-8")
figma_data  = json.loads(FIGMA_JSON)

# ─────────────────────────────────────────────
# 2. Crawl for visible text + frequencies
# ─────────────────────────────────────────────
def crawl_text(node: dict, collector: list[str]):
    if node.get("type") == "TEXT":
        txt = node.get("characters", "").strip()
        if txt:
            # drop pure numbers / money / % (very common in rows)
            stripped = txt.replace(",", "").replace("$", "").replace("%", "")
            if not stripped.replace(".", "").isdigit():
                collector.append(txt)
    for child in node.get("children", []):
        crawl_text(child, collector)

all_strings: list[str] = []
crawl_text(figma_data, all_strings)

freq = Counter(all_strings)
appear_once = [s for s in all_strings if freq[s] == 1]          # singletons, in original order

# ─────────────────────────────────────────────
# 3. Build the prompt
# ─────────────────────────────────────────────
def build_prompt(candidates: list[str]) -> str:
    bullet_blob = "\n".join(f"- {t}" for t in candidates)
    return f"""
You are looking at UI text extracted from a Figma sales‐dashboard design.

**Facts you can rely on**
• The main data table has one horizontal row of column headers (10 labels).  
• Each header string appears **exactly once** in the entire design file.  
• Navigation tabs, section titles, status chips, and row values repeat elsewhere.  
• Headers are concise (2–3 words max), start with a capital letter, and sit above numeric / date data.

**Your task**  
Return a JSON object giving exactly 10 unique column-header labels chosen **only** from the list below.
 • No values that repeat in the list below.  
 • No near-duplicates or synonyms.  
 • Skip strings shorter than 3 chars or obviously generic (“Open”, “Value”, “Action”, …).

**Output – REQUIRED format (nothing else)**  
{{
  "header1": "…",
  "header2": "…",
   ...
  "header10": "…"
}}

Candidate strings
-----------------
{bullet_blob}
""".strip()

# ─────────────────────────────────────────────
# 4. /api/top10 endpoint
# ─────────────────────────────────────────────
@app.get("/api/top10")
def api_top10():
    prompt = build_prompt(appear_once)

    try:
        resp = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp["message"]["content"]

        # Find the first {...} block that contains "header1"
        match = re.search(r"\{[^{}]*\"header1\"[^{}]*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No header JSON object found in model response.")

        obj_text = match.group(0)
        headers_obj = json.loads(obj_text)

        # Ensure we always deliver 10 keys (fill blanks if the model under-shoots)
        result = {f"header{i}": headers_obj.get(f"header{i}", "") for i in range(1, 11)}
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "Failed to parse Ollama response",
            "details": str(e),
            "raw_response": resp["message"]["content"] if 'resp' in locals() else "no response"
        }), 500

# ─────────────────────────────────────────────
# 5. Dev sanity route
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return jsonify({"note": "GET /api/top10 to fetch table headers extracted by the LLM"})

if __name__ == "__main__":
    print("⇢ Running with freq-filtered candidate list…")
    app.run(debug=True)