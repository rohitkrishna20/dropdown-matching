# app.py
from __future__ import annotations
from flask import Flask, request, jsonify
import json, re, math
from typing import Any, Dict, List, Tuple, Iterable, Optional

try:
    import ollama  # optional (local, open-source)
    OLLAMA_OK = True
except Exception:
    OLLAMA_OK = False

app = Flask(__name__)

# ----------------------------- Utilities -----------------------------

def force_decode(x: Any) -> Any:
    """
    Accepts: dict/list (already-parsed JSON), JSON string, or base64-encoded JSON string.
    Returns: Python object (dict/list).
    """
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    s = str(x)
    # Try plain JSON first
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try base64 -> JSON
    import base64
    try:
        s2 = base64.b64decode(s).decode("utf-8", errors="ignore")
        return json.loads(s2)
    except Exception:
        raise ValueError("Could not parse JSON (nor base64-encoded JSON).")


# ----------------------------- Figma harvest -----------------------------

GENERIC_BAD_WORDS = {
    "text","label","button","primary","action","subtitle","search","timestamp","icon",
    "open in gmail","open","menu","close","cancel","ok"
}

_WORD = re.compile(r"[A-Za-z0-9]+")
def _tokens(s: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(s)]

def _is_numbery(s: str) -> bool:
    z = s.strip().replace(",", "").replace("%", "").replace("$", "")
    return z.replace(".", "", 1).isdigit()

def _titlecaseish(s: str) -> bool:
    # short heuristic: a header-ish label is often Title Case or ALL CAPS
    letters = re.sub(r"[^A-Za-z]+","", s)
    if not letters:
        return False
    return s.istitle() or s.isupper()

def _header_likeliness(s: str) -> float:
    """Score how 'header-like' a text string is, w/o any hard-coded titles."""
    s_clean = s.strip()
    if not s_clean:
        return 0.0
    if _is_numbery(s_clean):
        return 0.0
    if len(s_clean) < 2 or len(s_clean) > 40:
        return 0.0

    toks = _tokens(s_clean)
    if not toks:
        return 0.0
    if any(t in GENERIC_BAD_WORDS for t in toks):
        return 0.0

    # heuristics: shorter phrases (1–5 tokens), contains letters, title-case-ish
    score = 0.0
    score += 0.35 if len(toks) <= 5 else 0.0
    score += 0.25 if _titlecaseish(s_clean) else 0.0
    # reward meaningful tokens (no all-stopword-y)
    score += min(0.4, 0.1 * sum(1 for t in toks if len(t) >= 3))
    return score

def _walk_figma(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk_figma(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_figma(v)

def extract_figma_headers(figma: Dict[str, Any]) -> List[str]:
    """
    Collect all TEXT node 'characters' plus select 'name' fields when useful,
    then filter by header-likeliness and dedupe by normalized form.
    """
    candidates: List[str] = []
    for node in _walk_figma(figma):
        try:
            if node.get("type") == "TEXT":
                txt = (node.get("characters") or "").strip()
                if txt and txt.lower() != "text" and not _is_numbery(txt):
                    candidates.append(txt)
        except Exception:
            pass

        # Fallback to 'name' fields for things like table headers if characters are absent
        nm = node.get("name")
        if isinstance(nm, str):
            nm_s = nm.strip()
            if nm_s and len(nm_s) <= 40 and not _is_numbery(nm_s):
                # Avoid very technical layer names with slashes
                if "/" not in nm_s and nm_s.lower() not in GENERIC_BAD_WORDS:
                    candidates.append(nm_s)

    # Score & keep header-like
    scored = [(s, _header_likeliness(s)) for s in candidates]
    keep = [s for s, sc in scored if sc >= 0.45]

    # normalize & dedupe (favor longest canonical variant per root word)
    norm = lambda x: re.sub(r"\s+", " ", x).strip()
    seen_roots, out = set(), []
    def root_key(s: str) -> str:
        tt = _tokens(s)
        return tt[0] if tt else norm(s).lower()
    for h in sorted({norm(s) for s in keep}, key=len):  # stable-ish
        r = root_key(h)
        if r in seen_roots:
            continue
        seen_roots.add(r)
        out.append(h)
    return out


# ----------------------------- OpenAPI field mining -----------------------------

def collect_schema_fields(openapi: Dict[str, Any]) -> List[str]:
    """
    Collect candidate field names from components.schemas.*.properties.*
    Also include top-level example 'fields' if present.
    """
    fields = set()

    # 1) components.schemas.*.properties
    comps = openapi.get("components", {}).get("schemas", {})
    for schema_obj in comps.values():
        props = (schema_obj or {}).get("properties", {})
        for k in props.keys():
            fields.add(str(k))

    # 2) example fields from 'paths'->...->parameters (if present)
    paths = openapi.get("paths", {}) or {}
    for p in paths.values():
        for method in (p or {}).values():
            for prm in (method or {}).get("parameters", []) or []:
                if prm.get("name") == "fields":
                    # nothing else to mine here, but keeping door open for future examples
                    pass

    return sorted(fields)


# ----------------------------- Similarity & mapping -----------------------------

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def char_ngram_tokens(s: str, n: int = 3) -> List[str]:
    s2 = re.sub(r"[^A-Za-z0-9]+", " ", s.lower())
    s2 = re.sub(r"\s+", " ", s2).strip()
    if len(s2) < n:
        return [s2] if s2 else []
    return [s2[i:i+n] for i in range(len(s2)-n+1)]

SYNONYMS = {
    # lightweight, non-hardcoded-to-your-app synonyms to help generic mapping
    "expected closure": {"close", "close date", "expected close", "expected closure"},
    "account": {"account", "customer", "client"},
    "total value": {"amount", "total", "value", "revenue"},
    "sales stage": {"stage", "status", "pipeline stage"},
    "win probability": {"probability", "win", "confidence", "likelihood"},
    "ai score": {"ai", "score", "model score", "ai score"},
    "created": {"created", "created date", "created_on", "created_at"},
    "source": {"source", "channel", "lead source"},
    "alerts": {"alert", "alerts", "flag", "at risk"},
    "name": {"name", "title"},
}

def synonym_boost(hdr: str, fld: str) -> float:
    h = " ".join(_tokens(hdr))
    f = " ".join(_tokens(fld))
    best = 0.0
    for canon, alts in SYNONYMS.items():
        if any(w in h for w in alts):
            if any(w in f for w in alts):
                best = max(best, 0.25)
    return best

def field_score(header: str, field: str) -> float:
    # token-level and character-level overlaps
    ht, ft = _tokens(header), _tokens(field)
    s1 = jaccard(ht, ft)
    s2 = jaccard(char_ngram_tokens(header), char_ngram_tokens(field))
    s3 = synonym_boost(header, field)
    # blend
    return 0.55*s1 + 0.35*s2 + s3

def top_k_fields_for_header(header: str, fields: List[str], k: int = 3) -> List[Tuple[str, float]]:
    scored = [(f, field_score(header, f)) for f in fields]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ----------------------------- (Optional) LLM refinement -----------------------------

LLM_SYSTEM = (
    "You map UI headers to schema field names. RULES:\n"
    "- Only choose from the provided list of field names.\n"
    "- Return compact JSON: {header: [field1, field2, field3], ...}\n"
    "- If you cannot find matches, return an empty list for that header.\n"
    "- Do not invent fields or headers.\n"
)

def refine_with_llm(model: str, headers: List[str], fields: List[str], draft: Dict[str, List[str]]) -> Optional[Dict[str, List[str]]]:
    if not OLLAMA_OK:
        return None
    prompt = (
        f"Headers:\n{json.dumps(headers, ensure_ascii=False)}\n\n"
        f"SchemaFields:\n{json.dumps(fields, ensure_ascii=False)}\n\n"
        f"DraftMapping (seeds):\n{json.dumps(draft, ensure_ascii=False)}\n\n"
        "Re-rank each header's top 3 best fields (stick strictly to SchemaFields)."
    )
    try:
        resp = ollama.chat(model=model, messages=[
            {"role":"system", "content": LLM_SYSTEM},
            {"role":"user", "content": prompt}
        ])
        txt = (resp.get("message") or {}).get("content","").strip()
        # Extract JSON block
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if not m:
            return None
        obj = json.loads(m.group(0))
        # sanitize: ensure outputs are from fields
        fields_set = set(fields)
        clean = {}
        for h, lst in obj.items():
            if h not in headers or not isinstance(lst, list):
                clean[h] = draft.get(h, [])
                continue
            clean[h] = [x for x in lst if x in fields_set][:3]
            if len(clean[h]) < 3:
                # top up from draft
                for x in draft.get(h, []):
                    if x in fields_set and x not in clean[h]:
                        clean[h].append(x)
                    if len(clean[h]) == 3:
                        break
        return clean
    except Exception:
        return None


# ----------------------------- API -----------------------------

@app.post("/api/headers-map")
def api_headers_map():
    """
    Request JSON:
    {
      "figma_jsons": [ ... one or more Figma JSON docs ... ],
      "schema_jsons": [ ... one or more OpenAPI JSON docs ... ],
      "use_llm": true,                # optional (default: true if ollama installed)
      "llm_model": "llama3"           # optional, defaults to 'llama3' if use_llm
    }
    Response JSON:
    {
      "headers": ["Sales Dashboard", "Overview", "My To-do's", ...],
      "mapping": { "Sales Dashboard": ["FieldA","FieldB","FieldC"], ... },
      "debug": { "notes": "...", "used_llm": true/false }
    }
    """
    try:
        raw = request.get_json(force=True, silent=False)
        if not isinstance(raw, dict):
            return jsonify({"error":"Body must be JSON object"}), 400

        figma_jsons = [force_decode(x) for x in raw.get("figma_jsons", [])]
        schema_jsons = [force_decode(x) for x in raw.get("schema_jsons", [])]
        if not figma_jsons or not schema_jsons:
            return jsonify({"error":"Provide 'figma_jsons' and 'schema_jsons' arrays"}), 400

        # 1) harvest headers from all figmas
        headers: List[str] = []
        for fj in figma_jsons:
            headers.extend(extract_figma_headers(fj))
        # de-dup while preserving order
        seen, headers = set(), [h for h in headers if not (h.lower() in seen or seen.add(h.lower()))]

        # 2) collect all fields from all OpenAPI docs
        schema_fields: List[str] = []
        seenf = set()
        for sj in schema_jsons:
            for f in collect_schema_fields(sj):
                if f not in seenf:
                    seenf.add(f)
                    schema_fields.append(f)

        # 3) heuristic top-3 for each header
        draft_map: Dict[str, List[str]] = {}
        for h in headers:
            top3 = [f for f,_ in top_k_fields_for_header(h, schema_fields, k=3)]
            draft_map[h] = top3

        # 4) optional LLM re-ranking
        use_llm = bool(raw.get("use_llm", True)) and OLLAMA_OK
        model = raw.get("llm_model", "llama3")
        final_map = refine_with_llm(model, headers, schema_fields, draft_map) if use_llm else None
        mapping = final_map or draft_map

        return jsonify({
            "headers": headers,
            "mapping": mapping,
            "debug": {
                "notes": "Headers are harvested only from Figma text; mapping chosen from schema fields via similarity, optionally refined by local LLM.",
                "used_llm": bool(final_map is not None)
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# ----------------------------- Local dev helper -----------------------------
@app.get("/")
def root():
    return "OK", 200

if __name__ == "__main__":
    import os, sys
    port = int(os.environ.get("PORT", "5000"))
    # allow --port / -p on the command line
    for i, a in enumerate(sys.argv):
        if a in ("--port", "-p") and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except ValueError:
                pass
            break
    app.run(host="0.0.0.0", port=port, debug=True)
