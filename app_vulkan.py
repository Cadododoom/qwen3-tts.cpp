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


import re

def clean_tts_text(text: str) -> str:
    if not text:
        return ""
        
    # Strip thinking blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Handle unclosed tags
    if re.search(r'<think>', text, re.IGNORECASE):
        text = re.compile(r'<think>', re.IGNORECASE).split(text)[0]
    if re.search(r'<thought>', text, re.IGNORECASE):
        text = re.compile(r'<thought>', re.IGNORECASE).split(text)[0]
        
    # Strip tool call and response blocks
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<tool_response>.*?</tool_response>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    if re.search(r'<tool_call>', text, re.IGNORECASE):
        text = re.compile(r'<tool_call>', re.IGNORECASE).split(text)[0]
    if re.search(r'<tool_response>', text, re.IGNORECASE):
        text = re.compile(r'<tool_response>', re.IGNORECASE).split(text)[0]
        
    # Strip code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Strip standalone JSON/dictionary objects (e.g. tool call arguments)
    text = re.sub(r'\{"\w+".*?\}', '', text, flags=re.DOTALL)
    
    # Convert markdown links [text](url) to just text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Clean up formatting characters
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = re.sub(r'#+\s*', '', text) # Strip markdown headers
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Cap to 70 words to stay under the 15-second client timeout
    words = text.split()
    if len(words) > 70:
        text = " ".join(words[:70]) + "..."
        
    return text


def make_wav_header(sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    import struct
    data_size = 0x7F000000
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
    return header

def stream_openai_tts(text: str, voice: str):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        yield make_wav(b"", sample_rate=24000)
        return
        
    yield make_wav_header(sample_rate=24000)
    
    embedding = get_voice_embedding(voice)
    
    for sentence in sentences:
        cleaned = clean_tts_text(sentence)
        if not cleaned:
            continue
            
        try:
            with engine_lock:
                if embedding is not None:
                    pcm_bytes = engine.synthesize_with_embedding(
                        text=cleaned,
                        embedding=embedding,
                        language="en"
                    )
                else:
                    pcm_bytes = engine.synthesize(
                        text=cleaned,
                        language="en"
                    )
            yield pcm_bytes
        except Exception as e:
            print(f"[Streaming OpenAI TTS Error] {e}")
            break

def stream_vapi_tts(text: str, voice: str, language: str):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        yield b""
        return
        
    embedding = get_voice_embedding(voice)
    
    for sentence in sentences:
        cleaned = clean_tts_text(sentence)
        if not cleaned:
            continue
            
        try:
            with engine_lock:
                if embedding is not None:
                    pcm_bytes = engine.synthesize_with_embedding(
                        text=cleaned,
                        embedding=embedding,
                        language=language
                    )
                else:
                    pcm_bytes = engine.synthesize(
                        text=cleaned,
                        language=language
                    )
            resampled = resample_24k_to_16k(pcm_bytes)
            yield resampled
        except Exception as e:
            print(f"[Streaming Vapi TTS Error] {e}")
            break


class OpenAIRequest(BaseModel):
    model: str = "qwen3"
    input: str
    voice: str = "default"
    response_format: str = "wav"
    speed: float = 1.0

@app.post("/v1/audio/speech")
def openai_tts(request: OpenAIRequest):
    cleaned_input = clean_tts_text(request.input)
    if not cleaned_input.strip():
        return Response(content=make_wav(b"", sample_rate=24000), media_type="audio/wav")
        
    return StreamingResponse(
        stream_openai_tts(request.input, request.voice),
        media_type="audio/wav"
    )


class VapiTTSRequest(BaseModel):
    text: str
    voice: str = "default"
    language: str = "en"

@app.post("/vapi-tts")
def vapi_tts(request: VapiTTSRequest):
    cleaned_text = clean_tts_text(request.text)
    if not cleaned_text.strip():
        return Response(content=b"", media_type="audio/l16")
        
    return StreamingResponse(
        stream_vapi_tts(request.text, request.voice, request.language),
        media_type="audio/l16"
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_NAME}
