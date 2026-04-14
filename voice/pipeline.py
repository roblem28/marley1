#!/usr/bin/env python3
"""Marley1 Voice Pipeline — continuous listen, offline, edge AI."""

import argparse
import io
import json
import struct
import subprocess
import sys
import tempfile
import time
import wave

import requests
import sounddevice as sd
import webrtcvad

sys.path.insert(0, "/home/marley1/marley1/compression")
from abbrev import ConstructionCompressor
from relevance_filter import RelevanceFilter
import pdfplumber
import os

COMPRESSOR = ConstructionCompressor()

SPECS_DIR = os.path.expanduser("~/marley1/specs")
DOC_TEXTS = {}
print("[MARLEY1] Loading specs...")
rf = RelevanceFilter()
for pdf in sorted(os.listdir(SPECS_DIR)):
    if pdf.endswith(".pdf"):
        path = os.path.join(SPECS_DIR, pdf)
        try:
            with pdfplumber.open(path) as doc:
                text = " ".join(p.extract_text() or "" for p in doc.pages)
            DOC_TEXTS[pdf] = rf.chunk(text, chunk_size=5)
            print(f"  Loaded {pdf}: {len(DOC_TEXTS[pdf])} chunks")
        except Exception as e:
            print(f"  WARN: {pdf}: {e}")
print(f"[MARLEY1] {len(DOC_TEXTS)} specs ready.")

# Pre-compute all chunk embeddings
all_chunks_flat = []
for chunks in DOC_TEXTS.values():
    all_chunks_flat.extend(chunks)
if all_chunks_flat:
    print(f"[MARLEY1] Pre-computing embeddings for {len(all_chunks_flat)} chunks...")
    rf.precompute(all_chunks_flat)
    print("[MARLEY1] Embeddings ready.")

def search_specs(query, top_k=10):
    if not hasattr(rf, "_cached_embs"):
        return ""
    chunks, _ = rf.filter_cached(query, top_k=top_k)
    return " ".join(chunks)








# --- Config ---
FAT_MAN_URL = "http://100.97.87.86:8080/v1/chat/completions"
WHISPER_CLI = "/home/marley1/marley1/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/marley1/marley1/whisper.cpp/models/ggml-tiny.bin"
PIPER_MODEL = "/home/marley1/marley1/voice/models/en_US-lessac-medium.onnx"
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
SILENCE_FRAMES = 30  # ~900ms of silence to trigger end
MIN_SPEECH_FRAMES = 10  # ~300ms minimum to count as speech
MAX_TOKENS = 512
LLM_TIMEOUT = 30


def record_utterance():
    """Record until speech detected then silence."""
    vad = webrtcvad.Vad(2)
    speech_frames = []
    silent_count = 0
    speaking = False
    print("[MIC] Listening...", flush=True)

    def callback(indata, frames, time_info, status):
        nonlocal silent_count, speaking
        if status:
            print(f"[MIC] {status}", flush=True)
        pcm = struct.pack(f"{len(indata)}h", *[int(s * 32767) for s in indata[:, 0]])
        is_speech = vad.is_speech(pcm, SAMPLE_RATE)
        if is_speech:
            speech_frames.append(pcm)
            silent_count = 0
            if not speaking and len(speech_frames) >= MIN_SPEECH_FRAMES:
                speaking = True
                stop_speaking()
                print("[MIC] Speech detected...", flush=True)
        else:
            if speaking:
                speech_frames.append(pcm)
                silent_count += 1

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=FRAME_SAMPLES, callback=callback, device="pipewire"):
        while True:
            sd.sleep(FRAME_MS)
            if speaking and silent_count >= SILENCE_FRAMES:
                break
            if not speaking and len(speech_frames) > 300:  # ~9s no speech, reset
                speech_frames.clear()

    if not speaking:
        return None

    print(f"[MIC] Captured {len(speech_frames)} frames", flush=True)
    return b"".join(speech_frames)


def transcribe(pcm_data):
    """Run whisper.cpp on PCM data."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wf = wave.open(f.name, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
        wf.close()

        result = subprocess.run(
            [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", f.name,
             "--no-timestamps", "-np"],
            capture_output=True, text=True, timeout=30
        )
        text = result.stdout.strip()
        # Filter whisper artifacts
        if text.startswith("["):
            text = ""
        return text


def query_fat_man(text, history):
    """Send to Fat Man LLM."""
    context = search_specs(text)
    if context:
        print(f"[SEARCH] {len(context)} chars of context found")
        user_content = f"Question: {text}\n\nContext:\n{context}"
    else:
        user_content = text
    system = {"role": "system", "content": "You are Marley, a concise voice assistant answering from provided documents. Answer in 2-3 sentences. No markdown."}
    messages = [system] + history + [{"role": "user", "content": user_content}]





    try:
        r = requests.post(FAT_MAN_URL, json={
            "model": "qwen",
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "stream": False
        }, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[LLM] Error: {e}", flush=True)
        return "Sorry, I could not get a response."


_tts_procs = []

def stop_speaking():
    """Kill any active TTS processes."""
    global _tts_procs
    for p in _tts_procs:
        try:
            p.kill()
        except:
            pass
    _tts_procs.clear()

def speak(text):
    """TTS via Piper through PipeWire. Interruptible."""
    global _tts_procs
    stop_speaking()
    try:
        piper = subprocess.Popen(
            ["/home/marley1/.local/bin/piper", "-m", PIPER_MODEL, "--output-raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        aplay = subprocess.Popen(
            ["aplay", "-D", "pipewire", "-f", "S16_LE", "-r", "22050", "-c", "1"],
            stdin=piper.stdout, stderr=subprocess.DEVNULL
        )
        _tts_procs = [piper, aplay]
        clean = __import__("re").sub(r"[*#_`\[\]()]", "", text)
        piper.stdin.write(clean.encode())
        piper.stdin.close()
        aplay.wait()
        _tts_procs.clear()
    except Exception as e:
        print(f"[TTS] Error: {e}", flush=True)


def run(lang="en"):
    print(f"[MARLEY1] Voice pipeline active | model=tiny | lang={lang}")
    print(f"[MARLEY1] Fat Man at {FAT_MAN_URL}")
    print(f"[MARLEY1] Speak to begin. Ctrl+C to exit.\n")
    history = []

    while True:
        try:
            pcm = record_utterance()
            if not pcm:
                continue

            t0 = time.time()
            transcript = transcribe(pcm)
            stt_time = time.time() - t0
            print(f"[STT] {stt_time:.1f}s -> {transcript!r}")

            if not transcript:
                continue

            t0 = time.time()
            response = query_fat_man(transcript, history)
            llm_time = time.time() - t0
            print(f"[LLM] {llm_time:.1f}s -> {response!r}")

            t0 = time.time()
            speak(response)
            tts_time = time.time() - t0
            print(f"[TTS] {tts_time:.1f}s")
            print(f"[TOTAL] {stt_time + llm_time + tts_time:.1f}s\n")

            history.append({"role": "user", "content": transcript})
            history.append({"role": "assistant", "content": response})
            if len(history) > 8:
                history = history[-8:]

        except KeyboardInterrupt:
            print("\n[MARLEY1] Shutting down.")
            break
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marley1 Voice Pipeline")
    parser.add_argument("--lang", default="en", choices=["en", "es", "auto"])
    parser.add_argument("--benchmark", action="store_true", help="Run single loop and print timings")
    args = parser.parse_args()

    if args.benchmark:
        print("[BENCHMARK] Running single voice loop...")
        run(lang=args.lang)
    else:
        run(lang=args.lang)
