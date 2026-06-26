import os
import time
import subprocess
import json

cli_path = "./build/qwen3-tts-cli"
model_dir = "models"
model_name = "qwen3-tts-1.7b-base-q8_0.gguf"
text = "The quick brown fox jumps over the lazy dog. Hello from qwen3-tts-cpp."

configs = [
    {
        "name": "CPU Only (4 threads)",
        "env": {"QWEN3_TTS_DEVICE": "cpu"},
        "threads": 4
    },
    {
        "name": "CPU Only (1 thread)",
        "env": {"QWEN3_TTS_DEVICE": "cpu"},
        "threads": 1
    },
    {
        "name": "Vulkan AMD GPU (4 threads)",
        "env": {"QWEN3_TTS_DEVICE": "vulkan0"},
        "threads": 4
    },
    {
        "name": "Vulkan AMD GPU (1 thread)",
        "env": {"QWEN3_TTS_DEVICE": "vulkan0"},
        "threads": 1
    },
    {
        "name": "Component Routing: Transformer on Vulkan GPU, Vocoder on CPU (1 thread)",
        "env": {"QWEN3_TTS_TRANSFORMER_DEVICE": "vulkan0", "QWEN3_TTS_DECODER_DEVICE": "cpu"},
        "threads": 1
    }
]

print("=" * 80)
print("                    QWEN3-TTS.CPP BACKEND BENCHMARK")
print("=" * 80)

results = []

for config in configs:
    name = config["name"]
    env = {**os.environ, **config["env"]}
    threads = config["threads"]
    
    print(f"\nRunning: {name}...")
    
    cmd = [
        cli_path,
        "-m", model_dir,
        "--model-name", model_name,
        "-t", text,
        "-o", "benchmark_out.wav",
        "-j", str(threads),
        "--max-tokens", "100"
    ]
    
    # Warmup run
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Measured run
    start_time = time.time()
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - start_time
    
    if proc.returncode != 0:
        print(f"Error running benchmark for {name}:")
        print(proc.stderr)
        continue
        
    stdout = proc.stdout
    stderr = proc.stderr
    
    # Parse output to extract times
    gen_time = 0
    dec_time = 0
    total_time = 0
    rtf = 0
    audio_dur = 0
    
    for line in stderr.split('\n'):
        if "  Generate:" in line:
            gen_time = float(line.split(":")[1].strip().replace("ms", ""))
        elif "  Decode:" in line:
            dec_time = float(line.split(":")[1].strip().replace("ms", ""))
        elif "  Total:" in line and "ms" in line:
            total_time = float(line.split(":")[1].strip().replace("ms", ""))
        elif "RTF=" in line:
            parts = line.split("RTF=")
            rtf = float(parts[1].replace(")", "").strip())
        elif "Audio duration:" in line and "seconds" in line:
            audio_dur = float(line.split(":")[1].strip().replace("seconds", ""))
            
    print(f"  -> Total inference: {total_time:.1f} ms")
    print(f"  -> Code generation: {gen_time:.1f} ms")
    print(f"  -> Vocoder decode: {dec_time:.1f} ms")
    print(f"  -> Real-time Factor (RTF): {rtf:.3f}x")
    
    results.append({
        "name": name,
        "total_time": total_time,
        "gen_time": gen_time,
        "dec_time": dec_time,
        "rtf": rtf,
        "audio_dur": audio_dur
    })

print("\n" + "=" * 80)
print("                                SUMMARY TABLE")
print("=" * 80)
print(f"{'Configuration':<60} | {'Total (ms)':<10} | {'RTF':<8}")
print("-" * 84)
for res in results:
    print(f"{res['name']:<60} | {res['total_time']:<10.1f} | {res['rtf']:<8.3f}")
print("=" * 80)
