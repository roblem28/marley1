#!/usr/bin/env python3
"""
Marley1 Embedding Daemon
Runs on Little Boy (Pi 5). Pre-loads all PDFs at startup, keeps embeddings in RAM.
Serves relevance filter requests via HTTP on port 8091.

Endpoints:
  POST /filter   {"pdf": "filename.pdf", "query": "...", "top_k": 10}
  GET  /status   returns loaded docs + memory info
  POST /reload   re-scans specs dir and loads any new PDFs
"""

import os
import sys
import json
import time
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPECS_DIR   = os.path.expanduser("~/marley1/specs")
PORT        = 8091
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE  = 5          # sentences per chunk
CHUNK_STEP  = 2          # overlap step
TOP_K_MAX   = 20

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model       = None
doc_store   = {}         # {filename: {"chunks": [...], "embeddings": np.array}}
store_lock  = threading.Lock()
status      = {"state": "starting", "loaded": [], "errors": []}

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------
def extract_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    pages.append(t)
            return "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"PDF extract failed: {e}")

def split_chunks(text, size=CHUNK_SIZE, step=CHUNK_STEP):
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) > 20]
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = ". ".join(sentences[i:i+size])
        if chunk:
            chunks.append(chunk)
    return chunks

# ---------------------------------------------------------------------------
# Load a single PDF into the store
# ---------------------------------------------------------------------------
def load_pdf(path):
    fname = os.path.basename(path)
    t0 = time.time()
    print(f"[daemon] loading {fname}...", flush=True)
    try:
        text   = extract_pdf(path)
        chunks = split_chunks(text)
        if not chunks:
            raise ValueError("no chunks extracted")
        embeddings = model.encode(chunks, batch_size=64, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        with store_lock:
            doc_store[fname] = {
                "chunks":     chunks,
                "embeddings": embeddings,
                "tokens_est": len(text.split()),
                "loaded_at":  time.time()
            }
        elapsed = time.time() - t0
        print(f"[daemon] {fname}: {len(chunks)} chunks, {len(text.split())} tokens, {elapsed:.1f}s", flush=True)
        return True
    except Exception as e:
        print(f"[daemon] ERROR loading {fname}: {e}", flush=True)
        status["errors"].append({"file": fname, "error": str(e)})
        return False

# ---------------------------------------------------------------------------
# Startup: load model + all PDFs
# ---------------------------------------------------------------------------
def startup():
    global model
    status["state"] = "loading_model"
    print("[daemon] loading embedding model...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"[daemon] model ready in {time.time()-t0:.1f}s", flush=True)

    status["state"] = "loading_docs"
    pdfs = sorted(Path(SPECS_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[daemon] no PDFs found in {SPECS_DIR}", flush=True)
    for p in pdfs:
        ok = load_pdf(str(p))
        if ok:
            status["loaded"].append(os.path.basename(str(p)))

    status["state"] = "ready"
    print(f"[daemon] ready — {len(doc_store)} docs loaded", flush=True)

# ---------------------------------------------------------------------------
# Filter: cosine similarity top-k
# ---------------------------------------------------------------------------
def filter_chunks(fname, query, top_k=10):
    with store_lock:
        if fname not in doc_store:
            return None, f"{fname} not loaded"
        entry = doc_store[fname]

    q_emb = model.encode([query])
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    scores = (entry["embeddings"] @ q_emb.T).flatten()
    top_k  = min(top_k, TOP_K_MAX, len(entry["chunks"]))
    idx    = np.argsort(scores)[::-1][:top_k]
    chunks = [entry["chunks"][i] for i in sorted(idx)]
    return chunks, None

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            with store_lock:
                info = {
                    "state":  status["state"],
                    "docs":   {k: {"chunks": len(v["chunks"]), "tokens_est": v["tokens_est"]}
                               for k, v in doc_store.items()},
                    "errors": status["errors"]
                }
            self.send_json(200, info)
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        if self.path == "/reload":
            threading.Thread(target=reload_docs, daemon=True).start()
            self.send_json(200, {"status": "reload started"})
            return

        if self.path == "/filter":
            try:
                req    = json.loads(body)
                pdf    = req.get("pdf", "")
                query  = req.get("query", "")
                top_k  = int(req.get("top_k", 10))

                if not pdf or not query:
                    self.send_json(400, {"error": "pdf and query required"})
                    return

                if status["state"] != "ready":
                    self.send_json(503, {"error": f"daemon not ready: {status['state']}"})
                    return

                t0 = time.time()
                chunks, err = filter_chunks(pdf, query, top_k)
                elapsed = time.time() - t0

                if err:
                    self.send_json(404, {"error": err})
                    return

                context = " ".join(chunks)
                self.send_json(200, {
                    "chunks":   chunks,
                    "context":  context,
                    "n_chunks": len(chunks),
                    "elapsed":  round(elapsed, 3)
                })
            except Exception as e:
                traceback.print_exc()
                self.send_json(500, {"error": str(e)})
            return

        self.send_json(404, {"error": "not found"})

# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------
def reload_docs():
    pdfs = sorted(Path(SPECS_DIR).glob("*.pdf"))
    with store_lock:
        loaded = set(doc_store.keys())
    new = [p for p in pdfs if os.path.basename(str(p)) not in loaded]
    for p in new:
        ok = load_pdf(str(p))
        if ok:
            status["loaded"].append(os.path.basename(str(p)))
    print(f"[daemon] reload complete, {len(new)} new docs", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    startup()
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[daemon] listening on port {PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[daemon] stopped", flush=True)
