#!/usr/bin/env python3
"""
server.py
Flask backend for speech-to-text transcription.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import tempfile
import time
from typing import Optional
import inspect
import numpy as np
import soundfile as sf
import librosa
from pywhispercpp.model import Model

app = Flask(__name__)
CORS(app)

MODEL = None
MODEL_NAME = "tiny.en"

def prepare_audio(input_path: str, target_sr: int = 16000) -> str:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    data, sr = sf.read(input_path)
    if data.ndim > 1: data = np.mean(data, axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype("float32") / max(abs(info.min), info.max)
    changed = False
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        changed = True
    max_abs = float(np.max(np.abs(data))) if data.size > 0 else 0.0
    if max_abs > 0 and max_abs < 0.99:
        data = data / max_abs * 0.99
        changed = True
    if not changed and sr == target_sr and (not input_path.endswith("_fixed.wav")):
        return input_path
    base, _ext = os.path.splitext(input_path)
    out_path = f"{base}_fixed.wav"
    sf.write(out_path, data.astype("float32"), sr, subtype="PCM_16")
    return out_path

def _transcribe_try_kwargs(model: Model, audio_path: str, kwargs: dict):
    return model.transcribe(audio_path, **kwargs)

def transcribe_with_model(model: Model, audio_path: str, language: Optional[str] = "en"):
    sig = inspect.signature(model.transcribe)
    accepts_kwargs = any(p.name in ("task", "language") for p in sig.parameters.values()) or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    best_kwargs = {"task": "transcribe", "language": language}
    if accepts_kwargs:
        try:
            return _transcribe_try_kwargs(model, audio_path, best_kwargs)
        except (AttributeError, TypeError):
            pass
    return model.transcribe(audio_path)

def format_segments(segments):
    result = []
    for seg in segments:
        try:
            text = getattr(seg, "text", None)
            if text is None and isinstance(seg, dict):
                text = seg.get("text")
        except Exception:
            text = None
        if text is not None:
            result.append({"text": text.strip() if isinstance(text, str) else str(text)})
    return result

def load_model():
    global MODEL
    if MODEL is None:
        print(f"Loading model: {MODEL_NAME}")
        MODEL = Model(MODEL_NAME)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files: return jsonify({'error': 'No audio'}), 400
    audio_file = request.files['audio']
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            audio_file.save(temp_path)
        t0 = time.perf_counter()
        audio_for_model = prepare_audio(temp_path, target_sr=16000)
        segments = transcribe_with_model(MODEL, audio_for_model, language="en")
        formatted = format_segments(segments)
        full_text = " ".join([seg["text"] for seg in formatted])
        t1 = time.perf_counter()
        try:
            os.unlink(temp_path)
            if audio_for_model != temp_path: os.unlink(audio_for_model)
        except: pass
        return jsonify({'success': True, 'transcription': full_text, 'processing_time': round(t1 - t0, 2)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=False)
