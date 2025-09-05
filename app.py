from __future__ import annotations
from flask import Flask, request, jsonify
import json, re, os, sys
from typing import Any, Dict, List, Iterable, Tuple

# Optional: local open-source LLM via Ollama
try:
    import ollama
    OLLAMA_OK = True
except Exception:
    OLLAMA_OK = False

app = Flask(__name__)

# ---------------- Utils ----------------

def force_decode(x: Any) -> Any:
    """
    Accepts dict/list (already JSON), JSON string, or base64-encoded JSON string.
    Returns a Python object or raises ValueError.
    """
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    s = str(x)
    try:
        return json.loads(s)
    except Exception:
        pass
    import base64
    try:
        s2 = base64.b64decode(s).decode("utf-8", errors="ignore")
        return json.loads(s2)
    except Exception:
        raise ValueError("Could not parse JSON (nor base64-encoded JSON).")

_WORD = re.compile(r"[A-Za-z0-9]+")

def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(s)]

def _is_numbery(s: str) -> bool:
    z = s.strip().replace(",", "").replace("%", "").replace("$", "")
    return z.replace(".", "", 1).isdigit()

def _titlecaseish(s: str) -> bool:
    letters = re.sub(r"[^A-Za-z]+","", s)
    if not letters:
        return False
    return s.istitle() or s.isupper()

# ---------------- Figma header harvest ----------------

GENERIC_BAD_WORDS = {
    "text","label","button","primary","action","subtitle","search","timestamp","icon","menu","close","cancel","ok"
}

def _walk(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)

def _header_likeliness(s: str) -> float:
    s = s.strip()
    if not s or _is_numbery(s) or len(s) < 2 or len(s) > 40:
        return 0.0
    toks = _tokens(s)
    if not toks or any(t in GENERIC_BAD_WORDS for t in toks):
        return 0.0
    score = 0.0
    if len(toks) <= 5: score += 0.35
    if _titlecaseish(s): score += 0.25
    score += min(0.4, 0.1 * sum(1 for t in toks if len(t) >= 3))
    return score

def extract_figma_headers(figma: Dict[str, Any]) -> List[str]:
    cand: List[str] = []
    for node in _walk(figma):
        if isinstance(node, dict):
            if node.get("type") == "TEXT":
                txt = (node.get("characters") or "").strip()
                if txt and txt.lower() != "text" and not _is_numbery(txt):
                    cand.append(txt)
            nm = node.get("name")
            if isinstance(nm, str):
                nm = nm.strip()
                if nm and len(nm) <= 40 and not _is_numbery(nm) and "/" not in nm and nm.lower() not in GENERIC_BAD_WORDS:
                    cand.append(nm)

    keep = [s.strip() for s in cand if _header_likeliness(s) >= 0.45]
    seen, out = set(), []
    for h in keep:
        k = re.sub(r"\s+"," ", h).lower()
        if k not in seen:
            seen.add(k); out.append(h.strip())
    return out

# ---------------- Schema field mining ----------------

def collect_schema_fields(openapi: Dict[str, Any]) -> List[str]:
    fields = set()
    comps = openapi.get("components", {}).get("schemas", {})
    for sch in comps.values():
        props = (sch or {}).get("properties", {})
        for k in props.keys():
            fields.add(str(k))
    return sorted(fields)

# ---------------- Similarity & mapping ----------------

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def char_ngrams(s: str, n: int = 3) -> List[str]:
    s2 = re.sub(r"[^A-Za-z0-9]+"," ", s.lower())
    s2 = re.sub(r"\s+"," ", s2).strip()
    if len(s2) < n: return [s2] if s2 else []
    return [s2[i:i+n] for i in range(len(s2)-n+1)]

SYNONYMS = {
    "expected closure": {"close","expected close","expected closure","close date"},
    "account": {"account","customer","client"},
    "total value": {"amount","total","value","revenue"},
    "sales stage": {"stage","status","pipeline stage"},
    "win probability": {"probability","win","likelihood","confidence"},
    "ai score": {"ai","score","model score"},
    "created": {"created","created date","created_on","created_at"},
    "source": {"source","channel","lead source"},
    "alerts": {"alert","alerts","flag","at risk"},
    "name": {"name","title"},
}

def synonym_boost(hdr: str, fld: str) -> float:
    h = " ".join(_tokens(hdr))
    f = " ".join(_tokens(fld))
    b = 0.0
    for alts in SYNONYMS.values():
        if any(w in h for w in alts) and any(w in f for w in alts):
            b = max(b, 0.25)
    return b

def field_score(header: str, field: str) -> float:
    t1 = jaccard(_tokens(header), _tokens(field))
    t2 = jaccard(char_ngrams(header), char_ngrams(field))
    t3 = synonym_boost(header, field)
    return 0.55*t1 + 0.35*t2 + t3

def top_k_fields(header: str, fields: List[str], k: int = 3) -> List[str]:
    ranked = sorted(fields, key=lambda f: field_score(header, f), reverse=True)
    return ranked[:k]

# ---------------- Optional LLM re-rank ----------------

LLM_SYSTEM = (
    "You map UI headers to schema fields. Rules:\n"
    "- Only choose from the provided field list.\n"
    "- Return compact JSON {header:[f1,f2,f3],...}.\n"
    "- Do not invent headers or fields.\n"
)

def refine_with_llm(model: str, headers: List[str], fields: List[str], draft: Dict[str, List[str]]):
    if not OLLAMA_OK:
        return None
    try:
        prompt = (
            f"Headers:\n{json.dumps(headers)}\n\n"
            f"SchemaFields:\n{json.dumps(fields)}\n\n"
            f"DraftMapping:\n{json.dumps(draft)}\n\n"
            "Re-rank each header's best three (stick strictly to SchemaFields)."
        )
        resp = ollama.chat(model=model, messages=[
            {"role":"system", "content": LLM_SYSTEM},
            {"role":"user", "content": prompt}
        ])
        txt = (resp.get("message") or {}).get("content","")
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m: return None
        obj = json.loads(m.group(0))
        fields_set = set(fields)
        clean = {}
        for h in headers:
            lst = obj.get(h, []) if isinstance(obj, dict) else []
            lst = [x for x in lst if x in fields_set][:3]
            # top up from draft if needed
            for x in draft.get(h, []):
                if len(lst) >= 3: break
                if x in fields_set and x not in lst:
                    lst.append(x)
            clean[h] = lst
        return clean
    except Exception:
        return None

# ---------------- API ----------------

@app.post("/api/headers-map")
def api_headers_map():
    """
    Body (Dashboard *or* Opportunities test):
    {
      "figma_jsons": [ { ... Figma JSON object ... } ],
      "schema_jsons": [ { ... OpenAPI JSON object ... } ],
      "use_llm": true,
      "llm_model": "llama3"
    }
    """
    try:
        body = request.get_json(force=True, silent=False)
        if not isinstance(body, dict):
            return jsonify({"error":"Body must be a JSON object"}), 400

        fj = body.get("figma_jsons")
        sj = body.get("schema_jsons")
        if not isinstance(fj, list) or not isinstance(sj, list):
            return jsonify({"error":"Provide arrays 'figma_jsons' and 'schema_jsons'."}), 400
        if len(fj) != 1 or len(sj) != 1:
            return jsonify({"error":"Provide exactly ONE Figma JSON and ONE Schema JSON per request."}), 400

        figma = force_decode(fj[0])
        schema = force_decode(sj[0])

        # 1) headers from that ONE figma
        headers = extract_figma_headers(figma)

        # 2) fields from that ONE schema
        fields = collect_schema_fields(schema)

        # 3) heuristic top-3 per header
        draft = {h: top_k_fields(h, fields, k=3) for h in headers}

        # 4) optional LLM refine
        use_llm = bool(body.get("use_llm", True))
        model = body.get("llm_model", "llama3")
        final = refine_with_llm(model, headers, fields, draft) if (use_llm and OLLAMA_OK) else None

        return jsonify({
            "headers": headers,
            "mapping": final or draft,
            "debug": {
                "used_llm": bool(final is not None),
                "notes": "One-to-one figma/schema mapping; headers only come from provided Figma."
            }
        }), 200

    except ValueError as ve:
        return jsonify({"error": f"Bad JSON: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

@app.get("/")
def root():
    return "OK", 200

if __name__ == "__main__":
    # Allow --port / -p or PORT env var
    port = int(os.environ.get("PORT", "5000"))
    for i, a in enumerate(sys.argv):
        if a in ("--port", "-p") and i + 1 < len(sys.argv):
            try: port = int(sys.argv[i+1])
            except ValueError: pass
            break
    app.run(host="0.0.0.0", port=port, debug=True)
