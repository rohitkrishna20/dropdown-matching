# === put near your imports (after app = Flask(__name__)) ===
app.url_map.strict_slashes = False  # accept both /path and /path/

# ---- wrap your core handler so we can alias it cleanly ----
def _headers_map_core():
    # MOVE the body of your current api_headers_map() into this function
    # and make it return (jsonify(...), 200) like before.
    # Example:
    #   body = ...  # parse
    #   return jsonify({...}), 200
    raise RuntimeError("Move your core logic here")  # <--- remove after moving

# ---- POST aliases (any of these will call the same core) ----
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
    # keeps compatibility if you or Postman still call the old route
    return _headers_map_core()

# ---- Helpful GET so opening in a browser doesn't 404 ----
@app.get("/api/headers-map")
@app.get("/api/headers_map")
@app.get("/api/header-map")
@app.get("/api/find_fields")
def headers_map_help():
    from flask import jsonify
    return jsonify({
        "hint": "Use POST with JSON body.",
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

# ---- Print your route table at startup (super useful) ----
def _print_routes_banner():
    print("\n=== Registered Flask routes ===")
    with app.app_context():
        for rule in app.url_map.iter_rules():
            methods = ",".join(sorted(m for m in rule.methods if m not in {"HEAD","OPTIONS"}))
            print(f"{rule.rule:30s}  [{methods}]")
    print("================================\n")

if __name__ == "__main__":
    import os, sys
    port = int(os.environ.get("PORT", "5001"))  # default to 5001 to avoid 5000 conflicts
    for i, a in enumerate(sys.argv):
        if a in ("--port", "-p") and i + 1 < len(sys.argv):
            try: port = int(sys.argv[i+1])
            except ValueError: pass
            break
    _print_routes_banner()  # <<< shows you exactly what’s live
    app.run(host="0.0.0.0", port=port, debug=True)
