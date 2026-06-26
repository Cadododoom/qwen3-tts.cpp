import ctypes
import os
import sys
import numpy as np

# Find the library
lib_name = "libqwen3tts.so"
lib_paths = [
    os.path.join(os.path.dirname(__file__), "build", lib_name),
    os.path.join(os.path.dirname(__file__), lib_name),
    os.path.join("/app/build", lib_name),
    os.path.join("/usr/local/lib", lib_name),
    lib_name
]

lib = None
for p in lib_paths:
    if os.path.exists(p):
        try:
            lib = ctypes.CDLL(p)
            print(f"[Wrapper] Loaded {lib_name} from: {p}")
            break
        except Exception as e:
            print(f"[Wrapper] Failed to load from {p}: {e}")

if lib is None:
    try:
        lib = ctypes.CDLL(lib_name)
        print(f"[Wrapper] Loaded {lib_name} from system path.")
    except Exception as e:
        print(f"[Wrapper] Critical error: could not load {lib_name}: {e}")
        sys.exit(1)

# Structs
class Qwen3TtsParams(ctypes.Structure):
    _fields_ = [
        ("max_audio_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("repetition_penalty", ctypes.c_float),
        ("language_id", ctypes.c_int32),
        ("instruction", ctypes.c_char_p),
    ]

class Qwen3TtsAudio(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
    ]

# Setup function signatures
lib.qwen3_tts_default_params.argtypes = [ctypes.POINTER(Qwen3TtsParams)]
lib.qwen3_tts_default_params.restype = None

lib.qwen3_tts_create.argtypes = [ctypes.c_char_p, ctypes.c_int32]
lib.qwen3_tts_create.restype = ctypes.c_void_p

lib.qwen3_tts_create_with_model_name.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32]
lib.qwen3_tts_create_with_model_name.restype = ctypes.c_void_p

lib.qwen3_tts_is_loaded.argtypes = [ctypes.c_void_p]
lib.qwen3_tts_is_loaded.restype = ctypes.c_int

lib.qwen3_tts_synthesize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(Qwen3TtsParams)]
lib.qwen3_tts_synthesize.restype = ctypes.POINTER(Qwen3TtsAudio)

lib.qwen3_tts_sample_rate.argtypes = [ctypes.c_void_p]
lib.qwen3_tts_sample_rate.restype = ctypes.c_int32

lib.qwen3_tts_free_audio.argtypes = [ctypes.POINTER(Qwen3TtsAudio)]
lib.qwen3_tts_free_audio.restype = None

lib.qwen3_tts_destroy.argtypes = [ctypes.c_void_p]
lib.qwen3_tts_destroy.restype = None

lib.qwen3_tts_synthesize_with_voice_file.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(Qwen3TtsParams)
]
lib.qwen3_tts_synthesize_with_voice_file.restype = ctypes.POINTER(Qwen3TtsAudio)

lib.qwen3_tts_extract_embedding_file.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32
]
lib.qwen3_tts_extract_embedding_file.restype = ctypes.c_int32

lib.qwen3_tts_synthesize_with_embedding.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.POINTER(Qwen3TtsParams)
]
lib.qwen3_tts_synthesize_with_embedding.restype = ctypes.POINTER(Qwen3TtsAudio)

lib.qwen3_tts_get_error.argtypes = [ctypes.c_void_p]
lib.qwen3_tts_get_error.restype = ctypes.c_char_p


class Qwen3TTSVulkan:
    def __init__(self, model_dir: str, model_name: str = None, n_threads: int = 4):
        self.model_dir = model_dir
        self.model_name = model_name
        self.n_threads = n_threads
        
        m_dir_bytes = model_dir.encode("utf-8")
        if model_name:
            m_name_bytes = model_name.encode("utf-8")
            self.handle = lib.qwen3_tts_create_with_model_name(m_dir_bytes, m_name_bytes, n_threads)
        else:
            self.handle = lib.qwen3_tts_create(m_dir_bytes, n_threads)
            
        if not self.handle or lib.qwen3_tts_is_loaded(self.handle) == 0:
            err = self.get_error()
            raise RuntimeError(f"Failed to create Qwen3TTSVulkan engine: {err}")
            
    def get_error(self) -> str:
        if not self.handle:
            return "No handle"
        err_ptr = lib.qwen3_tts_get_error(self.handle)
        return err_ptr.decode("utf-8") if err_ptr else "Unknown error"
        
    def _get_params(self, max_tokens=300, temperature=0.9, top_p=1.0, top_k=50, repetition_penalty=1.05, language="en", instruction=None):
        params = Qwen3TtsParams()
        lib.qwen3_tts_default_params(ctypes.byref(params))
        
        params.max_audio_tokens = max_tokens
        params.temperature = temperature
        params.top_p = top_p
        params.top_k = top_k
        params.n_threads = self.n_threads
        params.repetition_penalty = repetition_penalty
        
        # Map language strings to IDs
        lang_map = {
            "en": 2050, "english": 2050,
            "ja": 2058, "japanese": 2058,
            "zh": 2055, "chinese": 2055,
            "ru": 2069, "russian": 2069,
            "ko": 2064, "korean": 2064,
            "de": 2053, "german": 2053,
            "fr": 2061, "french": 2061,
            "es": 2054, "spanish": 2054,
        }
        params.language_id = lang_map.get(str(language).lower(), 2050)
        
        if instruction:
            params.instruction = instruction.encode("utf-8")
            
        return params
        
    def _audio_to_numpy_int16(self, audio_ptr) -> bytes:
        if not audio_ptr:
            err = self.get_error()
            raise RuntimeError(f"Synthesis failed: {err}")
            
        try:
            audio_struct = audio_ptr.contents
            n_samples = audio_struct.n_samples
            samples_ptr = audio_struct.samples
            
            # Read floats directly
            float_array = np.ctypeslib.as_array(samples_ptr, shape=(n_samples,))
            # Convert to int16 PCM
            clamped = np.clip(float_array, -1.0, 1.0)
            int16_samples = (clamped * 32767.0).astype(np.int16)
            return int16_samples.tobytes()
        finally:
            lib.qwen3_tts_free_audio(audio_ptr)
            
    def synthesize(self, text: str, **kwargs) -> bytes:
        params = self._get_params(**kwargs)
        audio_ptr = lib.qwen3_tts_synthesize(self.handle, text.encode("utf-8"), ctypes.byref(params))
        return self._audio_to_numpy_int16(audio_ptr)
        
    def synthesize_with_voice_file(self, text: str, reference_audio_path: str, **kwargs) -> bytes:
        params = self._get_params(**kwargs)
        audio_ptr = lib.qwen3_tts_synthesize_with_voice_file(
            self.handle, 
            text.encode("utf-8"), 
            reference_audio_path.encode("utf-8"), 
            ctypes.byref(params)
        )
        return self._audio_to_numpy_int16(audio_ptr)
        
    def extract_embedding(self, reference_audio_path: str) -> np.ndarray:
        embedding = np.zeros(1024, dtype=np.float32)
        emb_ptr = embedding.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = lib.qwen3_tts_extract_embedding_file(self.handle, reference_audio_path.encode("utf-8"), emb_ptr, 1024)
        if ret <= 0:
            err = self.get_error()
            raise RuntimeError(f"Embedding extraction failed: {err}")
        return embedding
        
    def synthesize_with_embedding(self, text: str, embedding: np.ndarray, **kwargs) -> bytes:
        params = self._get_params(**kwargs)
        emb_ptr = embedding.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        audio_ptr = lib.qwen3_tts_synthesize_with_embedding(
            self.handle,
            text.encode("utf-8"),
            emb_ptr,
            len(embedding),
            ctypes.byref(params)
        )
        return self._audio_to_numpy_int16(audio_ptr)
        
    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            lib.qwen3_tts_destroy(self.handle)
            self.handle = None
