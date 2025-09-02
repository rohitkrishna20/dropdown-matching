# app.py
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json, re, os, requests, ollama, string

app = Flask(__name__)

FEEDBACK_PATH = "feedback_memory.json"
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# ============== persistence ==============
def load_feedback():
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("correct", {})
                    data.setdefault("incorrect", {})
                    data.setdefault("last_run", {})
                    return data
        except Exception as e:
            print(f"⚠️ load_feedback: {e}")
    return {"correct": {}, "incorrect": {}, "last_run": {}}

def save_feedback():
    try:
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ save_feedback: {e}")

feedback_memory = load_feedback()

# ============== basic utils ==============
def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else s

def _split_tokens(s):
    if not isinstance(s, str): return []
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)  # camelCase -> camel Case
    s = re.sub(r"[_\-\/\.]", " ", s)        # snake/kebab/etc -> spaces
    return [t.lower() for t in re.split(r"\s+", s.strip()) if t.strip()]

# noise-like header filter (generic)
def is_noise_header(s):
    """Filter out Figma labels that look like auto IDs / hashes / noise."""
    if not isinstance(s, str):
        return True
    s = s.strip()
    if not s:
        return True
    # long single-token-ish alnum label
    if len(s) >= 24 and all(ch in string.ascii_letters + string.digits + "_-=" for ch in s):
        return True
    # high digit ratio (IDs often have many digits)
    digits = sum(ch.isdigit() for ch in s)
    if len(s) >= 12 and digits / max(1, len(s)) >= 0.3:
        return True
    # no letters at all
    if not any(ch.isalpha() for ch in s):
        return True
    return False

# short field name from dotted path (generic)
def field_shortname(path):
    """
    From a dotted path like ...properties.Alert.x-siebel-scale -> 'Alert'.
    Generic rules:
      - if '.properties.' exists, take token after LAST '.properties.' (skip 'items' if present)
      - else take the last token that is NOT prefixed with 'x-'
    """
    if not isinstance(path, str):
        return ""
    parts = path.split(".")
    if ".properties." in path:
        idxs = [i for i, t in enumerate(parts) if t == "properties"]
        if idxs:
            j = idxs[-1] + 1
            if j < len(parts):
                cand = parts[j]
                if cand == "items" and j + 1 < len(parts):
                    cand = parts[j + 1]
                return str(cand)
    for token in reversed(parts):
        if not token.startswith("x-"):
            return str(token)
    return parts[-1] if parts else ""

# ============== tolerant JSON intake (generic) ==============
def force_decode(raw):
    try:
        rounds = 0
        while isinstance(raw, str) and rounds < 6:
            rounds += 1
            try:
                raw = json.loads(raw)
                continue
            except Exception:
                break
        return raw
    except Exception:
        return raw

def get_payload(req):
    # Primary: JSON body
    payload = req.get_json(silent=True)
    if isinstance(payload, dict) and payload:
        return payload
    # Fallbacks: form-data, files, or raw text could be added here if you need them again
    return None

# ============== Figma text extraction (generic) ==============
def extract_figma_text(figma_json):
    """
    Generic: collect text from nodes that look like text containers.
    Keep short strings (<= 4 words) as candidates. No domain-specific lists.
    """
    out = []

    def is_numeric(t):
        t = str(t)
        cleaned = t.replace(",", "").replace("%", "").replace("$", "").strip()
        return cleaned.replace(".", "").isdigit()

    def maybe_add(t):
        if not isinstance(t, str): return
        s = t.strip()
        if not s or is_numeric(s): return
        if len(s.split()) <= 4:
            out.append(s)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                maybe_add(str(node.get("characters", "")))
            # also scan any direct string-like values
            for v in node.values():
                if isinstance(v, (str, int, float)):
                    maybe_add(str(v))
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(figma_json)
    # de-dup preserve order
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq

# ============== RHS field collection (generic) ==============
def collect_candidate_fields(data):
    """
    Generic: return dotted paths that look like leaf-ish fields.
    - Include paths whose value is a primitive.
    - Also include any path containing ".properties." (common in schema-like JSONs).
    """
    out = set()

    def is_primitive(x):
        return isinstance(x, (str, int, float, bool)) or x is None

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                if ".properties." in new_prefix:
                    out.add(new_prefix)
                if is_primitive(v):
                    out.add(new_prefix)
                else:
                    walk(v, new_prefix)
        elif isinstance(node, list):
            for item in node:
                walk(item, prefix)

    walk(data)
    return sorted(out)

def build_faiss_over_fieldnames(candidate_paths):
    """
    Build FAISS over the *short name* of each path (better semantic match),
    but preserve full dotted path in metadata for returning results.
    """
    docs = []
    for p in candidate_paths:
        short = field_shortname(p)
        docs.append(Document(page_content=short, metadata={"path": p}))
    return FAISS.from_documents(docs, OllamaEmbeddings(model=OLLAMA_MODEL))

# ============== Deterministic matching (generic) ==============
def rank_rhs_candidates(header, rhs_paths, k=5):
    """
    Generic scoring on *short names*:
    - exact short == header (case-insensitive)
    - substring relations
    - token overlap (Jaccard)
    """
    header_norm = _norm(header)
    header_toks = set(_split_tokens(header))
    scored = []

    for p in rhs_paths:
        short = field_shortname(p)
        short_norm = _norm(short)
        short_toks = set(_split_tokens(short))

        score = 0.0
        if short_norm == header_norm:
            score += 1000.0
        if header_norm in short_norm or short_norm in header_norm:
            score += 25.0
        # token overlap
        inter = len(header_toks & short_toks)
        union = len(header_toks | short_toks) or 1
        score += 5.0 * (inter / union)

        if score > 0.0:
            scored.append((score, p, short))

    scored.sort(key=lambda x: x[0], reverse=True)
    prelim, seen = [], set()
    for s, p, short in scored:
        if p not in seen:
            seen.add(p)
            prelim.append({"field": p, "field_short": short, "score": s})
        if len(prelim) >= k:
            break
    return prelim

# ============== Ollama info (optional helper) ==============
def get_ollama_info():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return {"ollama_host": OLLAMA_URL,
                    "models": [m.get("name") for m in r.json().get("models", [])]}
        return {"ollama_host": OLLAMA_URL, "error": f"status {r.status_code}"}
    except Exception as e:
        return {"ollama_host": OLLAMA_URL, "error": str(e)}

@app.get("/api/ollama_info")
def api_ollama_info():
    return jsonify(get_ollama_info())

# ============== LLM prompt (neutral) ==============
def make_prompt_from_figma(labels):
    blob = "\n".join(f"- {t}" for t in labels)
    incorrect = set(p for pats in feedback_memory.get("incorrect", {}).values() for p in pats)
    correct   = set(p for pats in feedback_memory.get("correct",   {}).values() for p in pats)
    avoid = ("\nAvoid patterns like:\n" + "\n".join(f"- {p}" for p in incorrect)) if incorrect else ""
    prefer = ("\nPrefer patterns like:\n" + "\n".join(f"- {p}" for p in correct)) if correct else ""

    return f"""
Extract likely TABLE COLUMN HEADERS from the list of UI labels below.
- Use ONLY labels that appear in the list (do not invent).
- Keep them short (1–3 words).

{avoid}
{prefer}

Return STRICT JSON: keys = normalized header names, values = the EXACT label you matched.

UI labels:
{blob}
""".strip()

# ============== API routes ==============
@app.post("/api/find_fields")
def api_find_fields():
    try:
        raw = get_payload(request)
        if not isinstance(raw, dict):
            return jsonify({"error": "Request must include figma_json and data_json"}), 400
        if "figma_json" not in raw or "data_json" not in raw:
            return jsonify({"error": "Missing 'figma_json' or 'data_json' keys"}), 400

        figma_json = force_decode(raw["figma_json"])
        data_json  = force_decode(raw["data_json"])

        # RHS candidates and FAISS over short names
        rhs_paths = collect_candidate_fields(data_json)
        field_index = build_faiss_over_fieldnames(rhs_paths)

        # FIGMA labels
        figma_labels = extract_figma_text(figma_json)

        # Try model mapping (values must be in figma_labels)
        headers_from_model = []
        if figma_labels:
            try:
                prompt = make_prompt_from_figma(figma_labels)
                out = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
                content = (out.get("message") or {}).get("content", "") or ""
                parsed = {}
                if content:
                    try:
                        parsed = json.loads(content) or {}
                    except Exception:
                        m = re.search(r"\{[\s\S]*?\}", content)
                        if m:
                            try:
                                parsed = json.loads(m.group())
                            except Exception:
                                parsed = {}
                if isinstance(parsed, dict):
                    fig_norm = {_norm(x) for x in figma_labels}
                    for _, v in parsed.items():
                        if isinstance(v, str) and _norm(v) in fig_norm:
                            headers_from_model.append(v)
            except Exception:
                pass

        # Build headers with generic fallbacks (no domain lists)
        headers, seen = [], set()

        def push(xs, limit=12):
            for x in xs:
                if x and x not in seen:
                    seen.add(x); headers.append(x)
                if len(headers) >= limit: break

        # a) model-suggested
        push(headers_from_model)

        # b) short figma labels (1–3 words), generic
        if len(headers) < 12 and figma_labels:
            shorties = [s for s in figma_labels if 1 <= len(s.split()) <= 3]
            push(shorties)

        # c) raw figma labels
        if len(headers) < 12 and figma_labels:
            push(figma_labels)

        # d) last resort: synthesize from RHS short names
        if not headers and rhs_paths:
            rhs_short = [field_shortname(p) for p in rhs_paths]
            push([h for h in rhs_short if h], limit=8)

        # filter out noise-like headers (IDs/hashes), keep cap
        headers = [h for h in headers if not is_noise_header(h)][:12]
        if not headers:
            # fallback again if everything filtered
            fig_non_noise = [s for s in figma_labels if not is_noise_header(s)]
            headers = fig_non_noise[:8] or [field_shortname(p) for p in rhs_paths[:8]]

        # Build matches per header: deterministic (shortname) + FAISS fill (shortname)
        matches = {}
        for header in headers:
            prelim = rank_rhs_candidates(header, rhs_paths, k=5)

            # fill with FAISS (dedup by full path)
            try:
                res = field_index.similarity_search_with_score(header, k=8)
                for doc, sc in res:
                    full = doc.metadata.get("path", doc.page_content)
                    short = field_shortname(full)
                    if not any(m["field"] == full for m in prelim):
                        prelim.append({"field": full, "field_short": short, "score": float(sc)})
            except Exception:
                res = field_index.similarity_search(header, k=8)
                for doc in res:
                    full = doc.metadata.get("path", doc.page_content)
                    short = field_shortname(full)
                    if not any(m["field"] == full for m in prelim):
                        prelim.append({"field": full, "field_short": short})

            # sort by score (if missing, push to end)
            prelim.sort(key=lambda x: x.get("score", -1e9), reverse=True)
            matches[header] = prelim[:5]

        # record for feedback/explanations
        feedback_memory["last_run"] = {}
        for h in headers:
            feedback_memory["correct"].setdefault(h, []).append(h)
            feedback_memory["last_run"][h] = {
                "matched_ui_label": h,
                "figma_text": figma_labels,
                "top_rhs_candidates": matches.get(h, [])
            }
        save_feedback()

        # debug view
        if request.args.get("debug") in {"1","true"}:
            rhs_short_sample = [field_shortname(p) for p in rhs_paths[:20]]
            return jsonify({
                "headers_extracted": headers,
                "matches": matches,
                "debug": {
                    "figma_label_count": len(figma_labels),
                    "figma_sample": figma_labels[:20],
                    "rhs_paths_count": len(rhs_paths),
                    "rhs_short_sample": rhs_short_sample,
                }
            })

        return jsonify({"headers_extracted": headers, "matches": matches})

    except Exception as e:
        return jsonify({"error": "Find fields failed", "details": str(e)}), 500

# ============== Feedback route (generic explanation) ==============
@app.post("/api/feedback")
def api_feedback():
    try:
        body = request.get_json(force=True)
        header = body.get("header")
        status = body.get("status")  # "correct" | "incorrect"
        if not header or status not in {"correct", "incorrect"}:
            return jsonify({"error": "Invalid feedback format"}), 400

        patterns = feedback_memory["correct"].get(header, [])

        if status == "correct":
            feedback_memory["correct"].setdefault(header, [])
            for p in patterns:
                if p not in feedback_memory["correct"][header]:
                    feedback_memory["correct"][header].append(p)
        else:
            feedback_memory["incorrect"].setdefault(header, [])
            for p in patterns:
                if p not in feedback_memory["incorrect"][header]:
                    feedback_memory["incorrect"][header].append(p)
            feedback_memory["correct"].pop(header, None)

        ctx = (feedback_memory.get("last_run") or {}).get(header, {})
        matched_ui = ctx.get("matched_ui_label")
        rhs_cands  = ctx.get("top_rhs_candidates", [])
        figma_sample = "\n".join((ctx.get("figma_text") or [])[:40])

        explain_prompt = f"""
Explain briefly how the system arrived at the header selection in neutral terms
(3–5 sentences). Focus on using UI labels that were present, shortness of labels,
and generic similarity between the header text and candidate RHS short names.

Header: {header}
Matched UI label: {matched_ui}
Top RHS candidates: {json.dumps(rhs_cands, ensure_ascii=False)}
Sample of UI labels:
{figma_sample}
""".strip()

        try:
            exp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": explain_prompt}])
            explanation = exp["message"]["content"]
        except Exception as oe:
            explanation = f"(Explanation unavailable: {oe})"

        save_feedback()

        return jsonify({
            "header": header,
            "status": status,
            "patterns_used": patterns,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": "Feedback failed", "details": str(e)}), 500

# ============== Root ==============
@app.get("/")
def home():
    return jsonify({
        "message": "POST /api/find_fields with {figma_json, data_json}. "
                   "Headers filtered for noise; matches include full path and field_short. "
                   "POST /api/feedback with {header, status} for explanations. "
                   "GET /api/ollama_info to see Ollama host/models. Add ?debug=1 to inspect."
    })

# ============== Runner ==============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    args = parser.parse_args()
    print(f"✅ API running at http://localhost:{args.port}")
    print("ℹ️  Ollama:", get_ollama_info())
    app.run(host="0.0.0.0", port=args.port, debug=True)
