import time
import pathlib
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC


def run_test():
    print("=" * 60)
    print("  Google MedASR Local Inference Test on Intel Arc (XPU)")
    print("=" * 60)

    # 1. Locate official sample audio in Hugging Face cache
    cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "hub" / "models--google--medasr" / "snapshots"
    if not cache_dir.exists():
        raise FileNotFoundError(f"MedASR cache snapshot directory not found at: {cache_dir}")

    snapshot_dir = next(cache_dir.iterdir())
    audio_path = snapshot_dir / "test_audio.wav"

    if not audio_path.exists():
        raise FileNotFoundError(f"Official test_audio.wav not found at: {audio_path}")

    print(f"Audio Path: {audio_path}")

    # 2. Inspect audio file
    data, sample_rate = sf.read(str(audio_path))
    duration = len(data) / sample_rate
    channels = 1 if len(data.shape) == 1 else data.shape[1]
    sample_count = len(data)

    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Channels: {channels} (Mono)")
    print(f"Sample Count: {sample_count:,}")
    print(f"Audio Duration: {duration:.2f} s")

    # 3. Model & Processor loading from local cache
    print("\nLoading MedASR model & processor from local cache...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained("google/medasr", local_files_only=True)
    model = AutoModelForCTC.from_pretrained("google/medasr", local_files_only=True)
    model_load_time = time.perf_counter() - t0
    print(f"Model Load Time (CPU): {model_load_time:.4f} s")

    # 4. Device verification and placement
    if not torch.xpu.is_available():
        raise RuntimeError("Intel XPU is not available on this system.")

    device = torch.device("xpu:0")
    gpu_name = torch.xpu.get_device_name(0)
    print(f"Target Device: {device} ({gpu_name})")

    model = model.to(device)
    model_device = next(model.parameters()).device
    print(f"Verified Model Device: {model_device}")

    # 5. Preprocessing
    t1 = time.perf_counter()
    inputs = processor(data, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prep_time = time.perf_counter() - t1
    input_device = inputs["input_features"].device
    print(f"Preprocessing Time: {prep_time:.4f} s")
    print(f"Verified Input Tensor Device: {input_device}")

    # 6. Inference on XPU with synchronization
    print("\nRunning inference on Intel Arc GPU...")
    torch.xpu.synchronize()
    t2 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)
    torch.xpu.synchronize()
    inference_time = time.perf_counter() - t2

    # 7. Postprocessing / Decoding
    transcription = processor.batch_decode(predicted_ids)[0]
    total_time = model_load_time + prep_time + inference_time
    rtf = inference_time / duration

    # 8. Print formatted results
    print("\n" + "=" * 60)
    print("  INFERENCE RESULTS SUMMARY")
    print("=" * 60)
    print(f"GPU Name:               {gpu_name}")
    print(f"Model Device:           {model_device}")
    print(f"Input Tensor Device:    {input_device}")
    print(f"Audio Duration:         {duration:.2f} s")
    print(f"Model Loading Time:     {model_load_time:.4f} s")
    print(f"Preprocessing Time:     {prep_time:.4f} s")
    print(f"XPU Inference Time:     {inference_time:.4f} s")
    print(f"Total Time:             {total_time:.4f} s")
    print(f"Real-Time Factor (RTF): {rtf:.4f} (Inference is {1/rtf:.2f}x faster than real-time)")
    print("\nTranscription Output:")
    print("-" * 60)
    print(transcription)
    print("-" * 60)
    print("\nXPU Inference Succeeded: True")


if __name__ == "__main__":
    run_test()
