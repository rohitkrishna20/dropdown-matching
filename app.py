# app.py
from __future__ import annotations
from flask import Flask, request, jsonify
import json, re, os, sys
from typing import Any, Dict, List, Iterable, Tuple, Optional

# Optional: local open-source LLM via Ollama (used only for re-ranking)
try:
    import ollama
    OLLAMA_OK = True
except Exception:
    OLLAMA_OK = False

app = Flask(__name__)
app.url_map.strict_slashes = False  # accept both /path and /path/

# =========================== Permissive "JSON-ish" parsing ===========================

SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'",
}

def _strip_bom_zw(s: str) -> str:
    return s.replace("\ufeff", "").replace("\u200b", "").replace("\u200e", "").replace("\u200f", "")

def _normalize_quotes(s: str) -> str:
    for k, v in SMART_QUOTES.items():
        s = s.replace(k, v)
    return s

def _strip_comments(s: str) -> str:
    # remove // line comments and /* block */ comments
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return s

def _strip_trailing_commas(s: str) -> str:
    # remove trailing commas before } or ]
    return re.sub(r",\s*([}\]])", r"\1", s)

def _extract_bracketed(s: str) -> Optional[str]:
    # Extract the largest {...} or [...] block if there is noise around it
    first_obj = s.find("{"); last_obj = s.rfind("}")
    first_arr = s.find("["); last_arr = s.rfind("]")
    cand = None
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        cand = s[first_obj:last_obj+1]
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        arr = s[first_arr:last_arr+1]
        if cand is None or len(arr) > len(cand):
            cand = arr
    return cand

def loose_json_loads(s: str) -> Any:
    """
    Best-effort parser for 'JSON-ish' text:
    - strips BOM/ZW chars, comments, trailing commas, smart quotes
    - extracts the largest {...} or [...] block if there is leading/trailing noise
    - falls back to base64->JSON
    """
    if not isinstance(s, str):
        s = str(s)
    s = _strip_bom_zw(s)
    s = _normalize_quotes(s)
    s = _strip_comments(s)
    s = _strip_trailing_commas(s).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    cand = _extract_bracketed(s)
    if cand:
        try:
            return json.loads(cand)
        except Exception:
            try:
                return json.loads(_strip_trailing_commas(cand))
            except Exception:
                pass

    try:
        import base64
        s2 = base64.b64decode(s).decode("utf-8", errors="ignore")
        s2 = _strip_bom_zw(_normalize_quotes(_strip_comments(_strip_trailing_commas(s2))))
        return json.loads(s2)
    except Exception:
        pass

    raise ValueError("Could not parse body as JSON (even after cleanup).")

def force_decode_any(x: Any) -> Any:
    """
    Accept dict/list as-is; if str/bytes, attempt loose JSON parse or base64->JSON.
    """
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, str):
        return loose_json_loads(x)
    raise ValueError("Unsupported payload type.")

# =========================== Heuristics: classify Figma vs OpenAPI ===========================

def _walk(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)

def looks_like_figma(obj: Any) -> bool:
    # Clues: nodes with type: 'TEXT' or 'DOCUMENT', 'characters' fields, 'document'/'children'
    try:
        for n in _walk(obj):
            t = n.get("type")
            if isinstance(t, str) and t.upper() in {"TEXT", "DOCUMENT", "FRAME", "PAGE"}:
                return True
            if "characters" in n and isinstance(n["characters"], (str, int, float)):
                return True
            if "document" in n or "children" in n:
                return True
    except Exception:
        pass
    return False

def looks_like_openapi(obj: Any) -> bool:
    # Clues: 'openapi' or 'swagger' keys; components.schemas.*.properties; non-empty paths
    if not isinstance(obj, dict):
        return False
    if "openapi" in obj or "swagger" in obj:
        return True
    comps = obj.get("components", {})
    if isinstance(comps, dict):
        sch = comps.get("schemas", {})
        if isinstance(sch, dict) and sch:
            return True
    if isinstance(obj.get("paths", {}), dict) and obj.get("paths", {}):
        return True
    return False

# =========================== Header extraction & mapping ===========================

GENERIC_BAD_WORDS = {
    "text","label","button","primary","action","subtitle","search","timestamp","icon","menu","close","cancel","ok"
}
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

def collect_schema_fields(openapi: Dict[str, Any]) -> List[str]:
    fields = set()
    comps = openapi.get("components", {}).get("schemas", {})
    for sch in comps.values():
        props = (sch or {}).get("properties", {})
        for k in props.keys():
            fields.add(str(k))
    return sorted(fields)

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

# =========================== Optional LLM re-rank ===========================

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
            for x in draft.get(h, []):
                if len(lst) >= 3: break
                if x in fields_set and x not in lst:
                    lst.append(x)
            clean[h] = lst
        return clean
    except Exception:
        return None

# =========================== Robust body ingestion & classification ===========================

def _collect_candidates_from_body(body: Any) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """
    Returns (figma_candidates, schema_candidates, debug_info).
    Scans known keys and falls back to classifying all dicts found anywhere.
    """
    dbg = {"keys_seen": [], "paths_scanned": 0}
    figma_cands: List[Any] = []
    schema_cands: List[Any] = []

    def push_candidate(x: Any):
        nonlocal figma_cands, schema_cands
        try:
            obj = force_decode_any(x)
        except Exception:
            return
        try:
            if looks_like_figma(obj):
                figma_cands.append(obj); return
            if looks_like_openapi(obj):
                schema_cands.append(obj); return
        except Exception:
            pass

    def scan(node: Any):
        dbg["paths_scanned"] += 1
        if isinstance(node, dict):
            for k, v in node.items():
                dbg["keys_seen"].append(k)
                if k.lower() in {"figma", "figma_json", "figmajson"}:
                    push_candidate(v)
                if k.lower() in {"schema", "schema_json", "schemajson", "openapi"}:
                    push_candidate(v)
                if k.lower() in {"figma_jsons", "figmas", "figma_list"} and isinstance(v, list):
                    for it in v: push_candidate(it)
                if k.lower() in {"schema_jsons", "schemas", "schema_list"} and isinstance(v, list):
                    for it in v: push_candidate(it)
                scan(v)  # descend
        elif isinstance(node, list):
            for it in node:
                scan(it)
        elif isinstance(node, (str, bytes, bytearray)):
            try:
                obj = force_decode_any(node)
                scan(obj)
            except Exception:
                pass

    scan(body)

    # If none explicitly found, look for any dicts and classify
    if not figma_cands and not schema_cands:
        for n in _walk(body):
            if looks_like_figma(n): figma_cands.append(n)
            if looks_like_openapi(n): schema_cands.append(n)

    return figma_cands, schema_cands, dbg

# =========================== Core handler (shared by all POST aliases) ===========================

def _headers_map_core():
    """
    Accepts messy JSON and auto-detects Figma vs OpenAPI blobs.
    Returns headers harvested from Figma and top-3 schema fields for each header.
    """
    try:
        # Try standard JSON first; if missing/invalid, fall back to permissive parse
        body = request.get_json(silent=True)
        if body is None:
            raw = request.get_data(as_text=True)
            body = loose_json_loads(raw)

        # Accept list-or-dict; if list, wrap so we can scan uniformly
        if not isinstance(body, (dict, list)):
            return jsonify({"error": "Request body must be a JSON object or array."}), 400
        wrapper = {"payload": body} if isinstance(body, list) else body

        # Gather candidates from conventional keys and anywhere in the body
        figma_cands, schema_cands, dbg = _collect_candidates_from_body(wrapper)

        if not figma_cands:
            return jsonify({"error": "No Figma-like JSON detected in request.",
                            "debug": dbg}), 400
        if not schema_cands:
            return jsonify({"error": "No OpenAPI/Schema-like JSON detected in request.",
                            "debug": dbg}), 400

        # Choose the first of each (most requests send one of each)
        figma = figma_cands[0]
        schema = schema_cands[0]

        # Extract headers & schema fields
        headers = extract_figma_headers(figma)
        fields = collect_schema_fields(schema)

        # Map each header to top-3 fields (optionally refine with LLM)
        draft = {h: top_k_fields(h, fields, k=3) for h in headers}
        if isinstance(wrapper, dict):
            use_llm = bool(wrapper.get("use_llm", True))
            model = wrapper.get("llm_model", "llama3")
        else:
            use_llm, model = True, "llama3"

        final = refine_with_llm(model, headers, fields, draft) if (use_llm and OLLAMA_OK) else None

        return jsonify({
            "headers": headers,
            "mapping": final or draft,
            "debug": {
                "used_llm": bool(final is not None),
                "figma_candidates": len(figma_cands),
                "schema_candidates": len(schema_cands),
                "keys_seen": sorted(set(dbg.get("keys_seen", [])))[:50],
                "notes": "Permissive parsing enabled: comments, trailing commas, base64, stringified JSON supported."
            }
        }), 200

    except ValueError as ve:
        return jsonify({"error": f"Bad JSON: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

# =========================== Routes ===========================

# Helpful GET so opening in a browser doesn't 404
@app.get("/api/headers-map")
@app.get("/api/headers_map")
@app.get("/api/header-map")
@app.get("/api/find_fields")
def headers_map_help():
    return jsonify({
        "hint": "Use POST with JSON body. Endpoint tolerates messy inputs and auto-detects Figma vs OpenAPI.",
        "expected_body": {
            "figma_jsons": ["<Figma JSON object or string>"],
            "schema_jsons": ["<OpenAPI JSON object or string>"],
            "use_llm": True,
            "llm_model": "llama3"
        },
        "examples": [
            "POST /api/headers-map",
            "POST /api/headers_map",
            "POST /api/header-map",
            "POST /api/find_fields"
        ]
    }), 200

# POST aliases (any of these will call the same core)
@app.post("/api/headers-map")
def api_headers_map_post():
    return _headers_map_core()

@app.post("/api/headers_map")
def api_headers_map_post_us():
    return _headers_map_core()

@app.post("/api/header-map")
def api_header_map_post():
    return _headers_map_core()

@app.post("/api/find_fields")
def api_find_fields_alias():
    return _headers_map_core()

# Root
@app.get("/")
def root():
    return "OK", 200

# =========================== Startup banner ===========================

def _print_routes_banner():
    print("\n=== Registered Flask routes ===")
    with app.app_context():
        for rule in app.url_map.iter_rules():
            methods = ",".join(sorted(m for m in rule.methods if m not in {"HEAD","OPTIONS"}))
            print(f"{rule.rule:30s}  [{methods}]")
    print("================================\n")

# =========================== Main ===========================

if __name__ == "__main__":
    # Default to 5001 to avoid common conflicts on 5000
    port = int(os.environ.get("PORT", "5001"))
    for i, a in enumerate(sys.argv):
        if a in ("--port", "-p") and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i+1])
            except ValueError:
                pass
            break
    _print_routes_banner()
    app.run(host="0.0.0.0", port=port, debug=True)
