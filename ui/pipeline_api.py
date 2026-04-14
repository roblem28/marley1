#!/usr/bin/env python3
"""
Marley1 Pipeline API
Runs on Little Boy port 8092.
"""

import os
import sys
import json
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS

SPECS_DIR = os.path.expanduser("~/marley1/specs")
PIPELINE  = os.path.expanduser("~/marley1/compression/pipeline_routed.py")

app = Flask(__name__)
CORS(app)

def parse_output(output):
    answer = ""
    total_time = ""
    cache_hit = False

    for line in output.splitlines():
        if "CACHE HIT" in line:
            cache_hit = True
        if "Total:" in line:
            total_time = line.strip()

    if "ANSWER:" in output:
        parts = output.split("ANSWER:")[1]
        lines = [l.strip() for l in parts.splitlines()
                 if l.strip() and not l.strip().startswith("-")]
        answer = " ".join(lines).strip()

    return answer, total_time, cache_hit

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    q   = data.get("query", "").strip()
    pdf = data.get("pdf", "")

    if not q:
        return jsonify({"error": "query required"}), 400

    if not pdf:
        pdfs = sorted(p for p in os.listdir(SPECS_DIR) if p.endswith(".pdf"))
        if not pdfs:
            return jsonify({"error": "no docs available"}), 404
        pdf = pdfs[0]

    pdf_path = os.path.join(SPECS_DIR, pdf)
    if not os.path.exists(pdf_path):
        return jsonify({"error": f"{pdf} not found"}), 404

    try:
        result = subprocess.run(
            ["python3", PIPELINE, pdf_path, q],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        answer, total_time, cache_hit = parse_output(output)

        if not answer and cache_hit:
            # re-run with cache cleared for this query
            import hashlib
            cache_dir = os.path.expanduser("~/marley1/compression/.cache")
            h = hashlib.md5((pdf_path + q).encode()).hexdigest()
            cache_file = os.path.join(cache_dir, h + ".json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            result = subprocess.run(
                ["python3", PIPELINE, pdf_path, q],
                capture_output=True, text=True, timeout=120
            )
            output = result.stdout + result.stderr
            answer, total_time, _ = parse_output(output)

        return jsonify({
            "answer":     answer,
            "pdf":        pdf,
            "total_time": total_time,
            "pipeline":   True,
            "cache_hit":  cache_hit
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "pipeline timeout"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/docs", methods=["GET"])
def docs():
    pdfs = sorted(p for p in os.listdir(SPECS_DIR) if p.endswith(".pdf"))
    return jsonify({"docs": pdfs})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import ssl
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(
        os.path.expanduser("~/marley1/ui/cert.pem"),
        os.path.expanduser("~/marley1/ui/key.pem")
    )
    app.run(host="0.0.0.0", port=8092, ssl_context=ctx, debug=False)
