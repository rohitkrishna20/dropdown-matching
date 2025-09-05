# app.py
from __future__ import annotations
from flask import Flask, request, jsonify
import json, re, os, sys
from typing import Any, Dict, List, Iterable, Tuple, Optional

# Optional: local open-source LLM via Ollama (used for extraction/reranking)
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

def _is_junky_label(s: str) -> bool:
    sl = s.lower()
    # obvious instance/variant IDs like I61:1669;59:17061;...
    if ";" in s or re.search(r"[A-Za-z]?\d+[:;]\d+", s):
        return True
    # more digits than letters -> likely an ID
    if len(re.findall(r"\d", s)) > len(re.findall(r"[A-Za-z]", s)):
        return True
    # variant/style/templating words we don't want as headers
    if any(w in sl for w in ["template", "variant", "private/", "light", "dark"]):
        return True
    # generic component names
    if any(w in sl for w in ["page header", "list item", "container", "frame meta"]):
        return True
    return False


def loose_json_loads(s: str) -> Any:
    """
    Best-effort parser for 'JSON-ish' text:
    - strips BOM/ZW chars, comments, trailing commas, smart quotes
    - extracts the largest {...} or [...] if there is leading/trailing noise
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
        return loose_json_loads(s=x)
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
    """
    Looser Figma detection:
    - accepts lowercase 'type'
    - detects TEXT nodes, 'characters' fields
    - tolerates plugin/table exports (columns/header/headerName/title/label/name)
    - recognizes common Figma node keys (absoluteBoundingBox, fills, strokes, style, layoutMode, etc.)
    """
    try:
        for n in _walk(obj):
            if not isinstance(n, dict):
                continue

            # type hints (case-insensitive, partial)
            t = n.get("type")
            if isinstance(t, str) and any(tok in t.lower() for tok in ("text","document","frame","page","group","component")):
                return True

            # canonical figma text/content
            if isinstance(n.get("characters"), (str, int, float)):
                return True

            # table-like structures many plugins export
            cols = n.get("columns")
            if isinstance(cols, list):
                for col in cols:
                    if isinstance(col, dict) and any(k in col for k in ("header","headerName","name","title","field")):
                        return True
                if any(isinstance(x, str) for x in cols):
                    return True

            # typical figma keys on nodes
            if any(k in n for k in ("absoluteBoundingBox","strokes","fills","style","layoutMode","constraints","effects","exportSettings","imageRef")):
                return True

            # label-ish keys with string values
            for k in ("text","content","title","label","heading","name","header","headerName"):
                v = n.get(k)
                if isinstance(v, str) and v.strip():
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
    if not s or _is_numbery(s) or len(s) < 2 or len(s) > 60:  # allow slightly longer labels
        return 0.0
    toks = _tokens(s)
    if not toks or any(t in GENERIC_BAD_WORDS for t in toks):
        return 0.0
    score = 0.0
    if len(toks) <= 6: score += 0.35
    if _titlecaseish(s): score += 0.25
    score += min(0.4, 0.1 * sum(1 for t in toks if len(t) >= 3))
    return score

# ---- helpers to turn keys/paths into phrases
_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NONWORD_SPLIT = re.compile(r"[^\w]+")

def _to_phrase(s: str) -> str:
    """Turn keys like 'DY_HomepageMgr_SalesDashboard_VBC' into 'DY Homepage Mgr Sales Dashboard VBC'."""
    # drop id/hash suffixes like '#1234:0'
    s = re.sub(r"#\d+:\d+", " ", s)
    # split on non-word
    parts = _NONWORD_SPLIT.split(s)
    parts2 = []
    for p in parts:
        if not p:
            continue
        # split camelCase / PascalCase
        parts2 += _CAMEL_SPLIT.split(p)
    phrase = " ".join(parts2)
    # collapse spaces
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase

def _prefer_longer_phrases(cands: List[str]) -> List[str]:
    """If 'Sales Dashboard' and 'Dashboard' both exist, prefer the longer and drop clear subsets."""
    keep = []
    lowered = [c.lower() for c in cands]
    for i, c in enumerate(cands):
        cl = lowered[i]
        drop = False
        for j, d in enumerate(cands):
            if i == j:
                continue
            dl = lowered[j]
            # if c is a strict substring of d and d is at least 2 tokens longer, drop c
            if cl != dl and cl in dl and len(_tokens(d)) >= len(_tokens(c)) + 1:
                drop = True
                break
        if not drop:
            keep.append(c)
    return keep

def extract_figma_headers(figma: Dict[str, Any]) -> List[str]:
    """
    Harvest headers from:
      - TEXT.characters (canonical)
      - common label keys: name/text/content/title/label/heading/header/headerName/field
      - columns[] (header/headerName/name/title or plain strings)
      - ALSO: from *keys themselves* by splitting delimiters & camelCase, then scoring like headers
      - NEW: if a value is a JSON string (e.g., frameMeta), parse it and recurse
    """
    cand: List[str] = []

    HEADER_VALUE_KEYS = {"characters","name","text","content","title","label","heading","header","headerName","field"}
    _CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
    _NONWORD_SPLIT = re.compile(r"[^\w]+")

    def _to_phrase(s: str) -> str:
        s = re.sub(r"#\d+:\d+", " ", s)           # drop id/hash suffixes like '#1234:0'
        parts = _NONWORD_SPLIT.split(s)
        parts2 = []
        for p in parts:
            if not p: continue
            parts2 += _CAMEL_SPLIT.split(p)       # split camelCase / PascalCase
        phrase = " ".join(parts2)
        return re.sub(r"\s+", " ", phrase).strip()

    def consider(val: Any):
        if isinstance(val, str):
            s = val.strip()
            # try to parse JSON-in-string (e.g., frameMeta)
            if s and (s.startswith("{") or s.startswith("[")):
                try:
                    sub = loose_json_loads(s)
                    harvest(sub)
                    return
                except Exception:
                    pass
            # otherwise treat as a candidate label
            if s and s.lower() != "text" and not _is_numbery(s):
                if _header_likeliness(s) >= 0.45:
                    cand.append(s)
        elif isinstance(val, list):
            for x in val:
                consider(x)
        elif isinstance(val, dict):
            harvest(val)

    def harvest(node: Any):
        if isinstance(node, list):
            for it in node: harvest(it)
            return
        if not isinstance(node, dict):
            return

        # 1) values in known label-ish fields
        for k in HEADER_VALUE_KEYS:
            if k in node:
                consider(node.get(k))

        # 2) table-like 'columns'
        cols = node.get("columns")
        if isinstance(cols, list):
            for col in cols:
                if isinstance(col, dict):
                    for kk in ("header","headerName","name","title"):
                        if kk in col:
                            consider(col.get(kk))
                elif isinstance(col, str):
                    consider(col)

        # 3) mine *keys* themselves (convert to phrases)
        for k in list(node.keys()):
            if not isinstance(k, str): continue
            if k in {"id","type","class","href","url","key"}:  # skip very technical keys
                continue
            phrase = _to_phrase(k).strip(" /")
            if phrase and _header_likeliness(phrase) >= 0.50:
                cand.append(phrase)

        # 4) always descend into values (and auto-parse JSON-ish strings)
        for v in node.values():
            consider(v)

    # kick off
    harvest(figma)

    # normalize, prefer longer composites, then rank
    def _prefer_longer_phrases(cands: List[str]) -> List[str]:
        keep, lowered = [], [c.lower() for c in cands]
        for i, c in enumerate(cands):
            cl, drop = lowered[i], False
            for j, d in enumerate(cands):
                if i == j: continue
                dl = lowered[j]
                if cl != dl and cl in dl and len(_tokens(d)) >= len(_tokens(c)) + 1:
                    drop = True; break
            if not drop: keep.append(c)
        return keep

    norm = lambda x: re.sub(r"\s+", " ", x).strip()
    uniq, seen = [], set()
    for s in cand:
        ss = norm(s)
        if ss.lower() not in seen:
            seen.add(ss.lower())
            uniq.append(ss)

    uniq = _prefer_longer_phrases(uniq)
    uniq_scored = sorted([(s, _header_likeliness(s)) for s in uniq], key=lambda t: t[1], reverse=True)
    return [s for s, _ in uniq_scored[:50]]


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

# =========================== Optional classic LLM re-rank ===========================

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

# =========================== LLM-first extraction (primary) ===========================

LLM_SYSTEM_FULL = (
    "You extract UI section headers from a Figma export and map them to OpenAPI schema fields.\n"
    "STRICT RULES:\n"
    "1) Headers MUST be chosen ONLY from FIGMA_CANDIDATES (exact strings). Do not invent.\n"
    "2) Prefer section-level labels, widget titles, and business column headers. "
    "   Ignore technical/atomic UI labels like Icon, Label, Primary, Secondary, Width, Height, etc.\n"
    "3) For each chosen header, pick up to 3 schema fields ONLY from SCHEMA_FIELDS by semantic relevance.\n"
    "4) Output STRICT JSON: {\"headers\":[...], \"mapping\":{\"<Header>\":[\"<field>\",...]}}\n"
    "5) If no good fields for a header, use [] for that header.\n"
    "6) No prose, no markdown, no explanations — only the JSON object.\n"
)

def _limit_list(a: List[str], max_items: int) -> List[str]:
    # Keep the most informative strings first by rough signal: longer & ‘header-likeliness’
    scored = [(s, _header_likeliness(s) + 0.1*len(_tokens(s))) for s in a]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [s for s,_ in scored[:max_items]]

def ollama_headers_and_mapping(
    model: str,
    figma_candidates: List[str],
    schema_fields: List[str],
    user_prompt: Optional[str] = None,
    seed_mapping: Optional[Dict[str, List[str]]] = None,
    max_figma: int = 250,
    max_fields: int = 500,
) -> Optional[Dict[str, Any]]:
    """
    Ask Ollama to: (1) select headers from figma_candidates, (2) map each to up to 3 schema_fields.
    Returns a dict like {"headers": [...], "mapping": {...}} or None on failure.
    """
    if not OLLAMA_OK:
        return None

    # Trim huge inputs to fit model context comfortably
    figma_cands = _limit_list(list(dict.fromkeys(figma_candidates)), max_figma)
    schema_list = list(dict.fromkeys(schema_fields))[:max_fields]

    payload = {
        "FIGMA_CANDIDATES": figma_cands,
        "SCHEMA_FIELDS": schema_list,
        "SEED_MAPPING": seed_mapping or {},
        "USER_GUIDANCE": (user_prompt or "").strip()
    }
    user_msg = (
        "Use the arrays below.\n"
        "Choose headers ONLY from FIGMA_CANDIDATES; choose fields ONLY from SCHEMA_FIELDS.\n"
        "If USER_GUIDANCE is present, follow it as long as it doesn’t violate the rules.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    try:
        resp = ollama.chat(model=model, messages=[
            {"role": "system", "content": LLM_SYSTEM_FULL},
            {"role": "user", "content": user_msg}
        ])
        txt = (resp.get("message") or {}).get("content", "").strip()
        m = re.search(r"\{.*\}", txt, flags=re.S)  # extract the JSON object
        if not m:
            return None
        obj = json.loads(m.group(0))

        # Sanitize: enforce subsets
        allowed_headers = set(figma_cands)
        allowed_fields  = set(schema_list)

        hdrs = [h for h in obj.get("headers", []) if isinstance(h, str) and h in allowed_headers]
        mapping = {}
        raw_map = obj.get("mapping", {}) if isinstance(obj.get("mapping", {}), dict) else {}
        for h in hdrs:
            arr = raw_map.get(h, [])
            if not isinstance(arr, list):
                arr = []
            clean = [f for f in arr if isinstance(f, str) and f in allowed_fields]
            mapping[h] = clean[:3]

        return {"headers": hdrs, "mapping": mapping}
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

# =========================== Utilities for explicit preference ===========================

def _prefer_explicit_over_detect(wrapper: dict) -> Tuple[Optional[Any], Optional[Any]]:
    """
    If the user provided figma_jsons / schema_jsons keys, decode the first entry directly,
    even if the classifier is unsure. This avoids 'no figma-like json' when users paste
    exotic Figma exports.
    """
    figma = schema = None
    # singular or plural keys accepted
    fj = wrapper.get("figma_jsons") or wrapper.get("figma_json")
    sj = wrapper.get("schema_jsons") or wrapper.get("schema_json") or wrapper.get("openapi")
    try:
        if isinstance(fj, list) and fj:
            figma = force_decode_any(fj[0])
        elif isinstance(fj, (dict, str, bytes, bytearray)):
            figma = force_decode_any(fj)
    except Exception:
        figma = None
    try:
        if isinstance(sj, list) and sj:
            schema = force_decode_any(sj[0])
        elif isinstance(sj, (dict, str, bytes, bytearray)):
            schema = force_decode_any(sj)
    except Exception:
        schema = None
    return figma, schema

# =========================== Core handler (LLM-first) ===========================

def _headers_map_core():
    """
    Accepts messy JSON and auto-detects Figma vs OpenAPI blobs.
    Prefers explicitly provided figma_jsons/schema_jsons if present.
    LLM-first: Ollama selects headers from candidates and maps to schema fields.
    """
    try:
        # Parse request body (permissive)
        body = request.get_json(silent=True)
        if body is None:
            raw = request.get_data(as_text=True)
            body = loose_json_loads(raw)

        if not isinstance(body, (dict, list)):
            return jsonify({"error": "Request body must be a JSON object or array."}), 400
        wrapper = {"payload": body} if isinstance(body, list) else body

        # Prefer explicitly provided blobs
        figma_explicit, schema_explicit = (None, None)
        if isinstance(wrapper, dict):
            figma_explicit, schema_explicit = _prefer_explicit_over_detect(wrapper)

        # Collect candidates via scanning (handles messy placements)
        figma_cands_all, schema_cands_all, dbg = _collect_candidates_from_body(wrapper)

        # Resolve final figma/schema objects
        figma = figma_explicit or (figma_cands_all[0] if figma_cands_all else None)
        schema = schema_explicit or (schema_cands_all[0] if schema_cands_all else None)

        if figma is None:
            return jsonify({"error": "No Figma-like JSON detected (and none provided in figma_jsons).",
                            "debug": dbg}), 400
        if schema is None:
            return jsonify({"error": "No OpenAPI/Schema-like JSON detected (and none provided in schema_jsons).",
                            "debug": dbg}), 400

        # --- Build candidate label list from Figma (values + keys) ---
        figma_candidates = extract_figma_headers(figma)  # smart harvesting (values, keys, columns, etc.)

        # --- Build schema field list from OpenAPI ---
        schema_fields = collect_schema_fields(schema)

        # --- Heuristic seed mapping (used as hints for the LLM) ---
        seeds = {h: [f for f in top_k_fields(h, schema_fields, k=3)] for h in figma_candidates}

        # --- Flags from request ---
        use_llm     = bool(wrapper.get("use_llm", True)) if isinstance(wrapper, dict) else True
        require_llm = bool(wrapper.get("require_llm", False)) if isinstance(wrapper, dict) else False
        model       = wrapper.get("llm_model", "llama3") if isinstance(wrapper, dict) else "llama3"
        user_prompt = wrapper.get("prompt", "") if isinstance(wrapper, dict) else ""

        # --- LLM-first extraction ---
        used_llm = False
        result = None
        if use_llm:
            result = ollama_headers_and_mapping(model, figma_candidates, schema_fields, user_prompt, seeds)
            used_llm = result is not None

        # --- Fallback to heuristics if LLM unavailable or returns nothing ---
        if not result:
            if require_llm and not OLLAMA_OK:
                return jsonify({"error": "Ollama not available (require_llm=true). Start Ollama or set require_llm=false."}), 503
            headers = figma_candidates
            mapping = {h: [f for f in top_k_fields(h, schema_fields, k=3)] for h in headers}
        else:
            headers = result["headers"]
            mapping = result["mapping"]

        return jsonify({
            "headers": headers,
            "mapping": mapping,
            "debug": {
                "used_llm": used_llm,
                "figma_candidates": len(figma_cands_all),
                "schema_candidates": len(schema_cands_all),
                "figma_label_candidates_used": len(figma_candidates),
                "schema_fields_used": len(schema_fields),
                "notes": "LLM-first extraction. Headers chosen only from figma candidates; fields only from schema.",
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
            "require_llm": False,
            "llm_model": "llama3",
            "prompt": "optional guidance to the model"
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
