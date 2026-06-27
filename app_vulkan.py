import os
import sys
import json
import numpy as np
import threading
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qwen3_tts_vulkan import Qwen3TTSVulkan

app = FastAPI(title="Qwen3-TTS Vulkan API Server")

# Paths and Config
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf")
THREADS = int(os.environ.get("THREADS", "4"))
VOICES_DIR = os.environ.get("VOICES_DIR", "/voices")

os.makedirs(VOICES_DIR, exist_ok=True)

# Global engine instance
print(f"[API] Initializing Qwen3-TTS Vulkan engine...")
print(f"      Model Dir:  {MODEL_DIR}")
print(f"      Model Name: {MODEL_NAME}")
print(f"      Threads:    {THREADS}")

try:
    engine = Qwen3TTSVulkan(model_dir=MODEL_DIR, model_name=MODEL_NAME, n_threads=THREADS)
    print("[API] Qwen3-TTS Vulkan engine loaded successfully.")
except Exception as e:
    print(f"[API] Error loading Qwen3-TTS Vulkan engine: {e}")
    sys.exit(1)

engine_lock = threading.Lock()


# Helper: resample mono int16 PCM from 24kHz to 16kHz
def resample_24k_to_16k(pcm_data: bytes) -> bytes:
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    num_input_samples = len(samples)
    if num_input_samples == 0:
        return b""
    num_output_samples = int(num_input_samples * (16000 / 24000))
    input_indices = np.arange(num_input_samples)
    output_indices = np.linspace(0, num_input_samples - 1, num_output_samples)
    resampled_samples = np.interp(output_indices, input_indices, samples)
    return resampled_samples.astype(np.int16).tobytes()


# Helper: create standard WAV container
def make_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    import struct
    channels = 1
    bits_per_sample = 16
    data_size = len(pcm_bytes)
    file_size = data_size + 36
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        file_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        sample_rate * channels * (bits_per_sample // 8),
        channels * (bits_per_sample // 8),
        bits_per_sample,
        b'data',
        data_size
    )
    return header + pcm_bytes


# Helper: load cached speaker embedding JSON
def get_voice_embedding(voice_name: str) -> np.ndarray:
    if not voice_name or voice_name.lower() == "default":
        return None
        
    # Search in VOICES_DIR and MODEL_DIR/preset_speakers
    search_paths = [
        os.path.join(VOICES_DIR, f"{voice_name}.json"),
        os.path.join(MODEL_DIR, "preset_speakers", f"{voice_name}.json"),
        os.path.join(MODEL_DIR, f"{voice_name}.json")
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                # Embeddings are typically JSON lists of 1024 floats
                if isinstance(data, list):
                    return np.array(data, dtype=np.float32)
                elif isinstance(data, dict) and "embedding" in data:
                    return np.array(data["embedding"], dtype=np.float32)
            except Exception as e:
                print(f"[API] Error reading voice profile {path}: {e}")
                
    return None


class OpenAIRequest(BaseModel):
    model: str = "qwen3"
    input: str
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0

@app.post("/v1/audio/speech")
def openai_tts(request: OpenAIRequest):
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
        
    try:
        # Check if we have a custom voice embedding
        embedding = get_voice_embedding(request.voice)
        
        # Determine language (default to en)
        if embedding is not None:
            with engine_lock:
                pcm_bytes = engine.synthesize_with_embedding(
                    text=request.input,
                    embedding=embedding,
                    language="en"
                )
        else:
            with engine_lock:
                pcm_bytes = engine.synthesize(
                    text=request.input,
                    language="en"
                )
            
        wav_bytes = make_wav(pcm_bytes, sample_rate=24000)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class VapiTTSRequest(BaseModel):
    text: str
    voice: str = "default"
    language: str = "en"

@app.post("/vapi-tts")
def vapi_tts(request: VapiTTSRequest):
    if not request.text.strip():
        return Response(content=b"", media_type="audio/l16")
        
    try:
        embedding = get_voice_embedding(request.voice)
        
        if embedding is not None:
            with engine_lock:
                pcm_bytes = engine.synthesize_with_embedding(
                    text=request.text,
                    embedding=embedding,
                    language=request.language
                )
        else:
            with engine_lock:
                pcm_bytes = engine.synthesize(
                    text=request.text,
                    language=request.language
                )
            
        # Vapi expects 16kHz L16 (raw 16-bit signed PCM mono)
        resampled_bytes = resample_24k_to_16k(pcm_bytes)
        return Response(content=resampled_bytes, media_type="audio/l16")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}
