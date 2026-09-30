import json
import pathlib
import time
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC


def run_benchmark():
    print("=" * 70)
    print("  MedASR Benchmark: CPU (Intel Core i7-13700H) vs XPU (Intel Arc A370M)")
    print("=" * 70)

    # 1. Locate test audio in local cache
    cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "hub" / "models--google--medasr" / "snapshots"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache snapshot directory not found: {cache_dir}")

    snapshot_dir = next(cache_dir.iterdir())
    audio_path = snapshot_dir / "test_audio.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Official test audio not found: {audio_path}")

    # 2. Inspect audio
    data, sample_rate = sf.read(str(audio_path))
    audio_duration = len(data) / sample_rate
    print(f"Audio Path:     {audio_path}")
    print(f"Audio Duration: {audio_duration:.2f} s ({len(data):,} samples @ {sample_rate} Hz)")

    # =========================================================================
    # A. CPU BENCHMARK
    # =========================================================================
    print("\n" + "-" * 40)
    print("  Running CPU Benchmark...")
    print("-" * 40)

    # CPU Model Loading
    t_cpu_load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained("google/medasr", local_files_only=True)
    model_cpu = AutoModelForCTC.from_pretrained("google/medasr", local_files_only=True)
    cpu_load_time = time.perf_counter() - t_cpu_load_start
    cpu_model_device = next(model_cpu.parameters()).device
    print(f"  CPU Model Loaded in:   {cpu_load_time:.4f} s (device: {cpu_model_device})")

    # CPU Preprocessing
    t_cpu_prep_start = time.perf_counter()
    inputs_cpu = processor(data, sampling_rate=sample_rate, return_tensors="pt")
    cpu_prep_time = time.perf_counter() - t_cpu_prep_start
    print(f"  CPU Preprocessing:     {cpu_prep_time:.4f} s")

    # CPU Inference
    t_cpu_inf_start = time.perf_counter()
    with torch.no_grad():
        outputs_cpu = model_cpu(**inputs_cpu)
        logits_cpu = outputs_cpu.logits
        predicted_ids_cpu = torch.argmax(logits_cpu, dim=-1)
    cpu_inf_time = time.perf_counter() - t_cpu_inf_start
    print(f"  CPU Inference Time:    {cpu_inf_time:.4f} s")

    cpu_transcription = processor.batch_decode(predicted_ids_cpu)[0]
    cpu_total_time = cpu_load_time + cpu_prep_time + cpu_inf_time
    cpu_rtf = cpu_inf_time / audio_duration

    # =========================================================================
    # B. XPU (INTEL ARC A370M) BENCHMARK
    # =========================================================================
    print("\n" + "-" * 40)
    print("  Running Intel Arc A370M XPU Benchmark...")
    print("-" * 40)

    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is not available on this machine.")

    xpu_device = torch.device("xpu:0")
    gpu_name = torch.xpu.get_device_name(0)

    # XPU Model Loading
    t_xpu_load_start = time.perf_counter()
    model_xpu = AutoModelForCTC.from_pretrained("google/medasr", local_files_only=True).to(xpu_device)
    torch.xpu.synchronize()
    xpu_load_time = time.perf_counter() - t_xpu_load_start
    xpu_model_device = next(model_xpu.parameters()).device
    print(f"  XPU Model Loaded in:   {xpu_load_time:.4f} s (device: {xpu_model_device})")

    # XPU Preprocessing
    t_xpu_prep_start = time.perf_counter()
    inputs_xpu = processor(data, sampling_rate=sample_rate, return_tensors="pt")
    inputs_xpu = {k: v.to(xpu_device) for k, v in inputs_xpu.items()}
    xpu_prep_time = time.perf_counter() - t_xpu_prep_start
    xpu_input_device = inputs_xpu["input_features"].device
    print(f"  XPU Preprocessing:     {xpu_prep_time:.4f} s (device: {xpu_input_device})")

    # XPU Inference with Synchronization
    torch.xpu.synchronize()
    t_xpu_inf_start = time.perf_counter()
    with torch.no_grad():
        outputs_xpu = model_xpu(**inputs_xpu)
        logits_xpu = outputs_xpu.logits
        predicted_ids_xpu = torch.argmax(logits_xpu, dim=-1)
    torch.xpu.synchronize()
    xpu_inf_time = time.perf_counter() - t_xpu_inf_start
    print(f"  XPU Inference Time:    {xpu_inf_time:.4f} s")

    xpu_transcription = processor.batch_decode(predicted_ids_xpu)[0]
    xpu_total_time = xpu_load_time + xpu_prep_time + xpu_inf_time
    xpu_rtf = xpu_inf_time / audio_duration

    # =========================================================================
    # C. COMPARISON & METRICS
    # =========================================================================
    transcriptions_identical = (cpu_transcription == xpu_transcription)
    gpu_speedup = cpu_inf_time / xpu_inf_time if xpu_inf_time > 0 else float("inf")

    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<28} | {'CPU (i7-13700H)':<18} | {'Intel Arc A370M XPU':<20}")
    print("-" * 72)
    print(f"{'Model loading time':<28} | {cpu_load_time:>16.4f} s | {xpu_load_time:>18.4f} s")
    print(f"{'Preprocessing time':<28} | {cpu_prep_time:>16.4f} s | {xpu_prep_time:>18.4f} s")
    print(f"{'Inference time':<28} | {cpu_inf_time:>16.4f} s | {xpu_inf_time:>18.4f} s")
    print(f"{'Total time':<28} | {cpu_total_time:>16.4f} s | {xpu_total_time:>18.4f} s")
    print(f"{'RTF':<28} | {cpu_rtf:>16.4f}   | {xpu_rtf:>18.4f}  ")
    print(f"{'Transcription identical?':<28} | {'--':>16}   | {str(transcriptions_identical):>18}  ")
    print(f"{'GPU speedup':<28} | {'1.00x':>16}   | {gpu_speedup:>17.2f}x ")
    print("=" * 70)

    print("\nAdditional Diagnostic Verification:")
    print(f"  - CPU Model Device:         {cpu_model_device}")
    print(f"  - XPU Model Device:         {xpu_model_device}")
    print(f"  - GPU Name:                 {gpu_name}")
    print(f"  - Both Inferences Succeeded: True")
    print(f"  - Transcription Length:     {len(cpu_transcription)} characters")

    # Output JSON representation for automated extraction
    benchmark_data = {
        "audio_duration_sec": audio_duration,
        "cpu": {
            "device": str(cpu_model_device),
            "model_load_sec": cpu_load_time,
            "prep_sec": cpu_prep_time,
            "inference_sec": cpu_inf_time,
            "total_sec": cpu_total_time,
            "rtf": cpu_rtf,
            "transcription": cpu_transcription
        },
        "xpu": {
            "device": str(xpu_model_device),
            "gpu_name": gpu_name,
            "model_load_sec": xpu_load_time,
            "prep_sec": xpu_prep_time,
            "inference_sec": xpu_inf_time,
            "total_sec": xpu_total_time,
            "rtf": xpu_rtf,
            "transcription": xpu_transcription
        },
        "identical": transcriptions_identical,
        "speedup": gpu_speedup
    }

    with open(pathlib.Path("tests") / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)


if __name__ == "__main__":
    run_benchmark()
