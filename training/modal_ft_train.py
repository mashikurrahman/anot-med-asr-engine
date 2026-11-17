#!/usr/bin/env python3
"""
LoRA fine-tune of Whisper large-v3 on verbatim clinical clips (Modal, ONE GPU).

Data and model weights live ONLY in a private Modal volume ("asr-ft"); nothing
patient-derived is written to git. Steps (run from the project root):

    .venv/Scripts/modal.exe volume create asr-ft
    .venv/Scripts/modal.exe volume put asr-ft scratch_ft_data /data
    .venv/Scripts/modal.exe run training/modal_ft_train.py

Trains on the train split, selects the epoch with the best content-WER on the
val split (early stopping), merges the adapter, and converts to a CTranslate2
model for faster-whisper at /vol/models/ft-ct2.
"""
import json

import modal

app = modal.App("asr-ft-train")
vol = modal.Volume.from_name("asr-ft", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.1", "transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1",
        "ctranslate2==4.5.0", "soundfile", "numpy<2", "sentencepiece",
    )
    .add_local_file("evaluation/content_norm.py", "/root/content_norm.py")
)

BASE = "openai/whisper-large-v3"
HP = dict(epochs=6, lr=3e-4, batch=4, accum=4, rank=16, alpha=32, dropout=0.05)


@app.function(image=image, gpu=["A10G", "L4"], volumes={"/vol": vol}, timeout=3 * 3600)
def train(name: str = "ft2", use_synth: bool = True, real_repeat: int = 3,
          epochs: int = 4, synth_limit: int = 0, manifest: str = "manifest.jsonl") -> dict:
    import dataclasses
    from typing import Any, Dict, List

    import soundfile as sf
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    from content_norm import clean_for_content_wer

    processor = WhisperProcessor.from_pretrained(BASE, language="english", task="transcribe")
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="english", task="transcribe", predict_timestamps=False)

    hp = {**HP, "epochs": epochs}
    rows = [json.loads(line) for line in open(f"/vol/data/{manifest}", encoding="utf-8")]
    train_rows = [r for r in rows if r["split"] == "train"] * real_repeat
    if use_synth:
        synth = [json.loads(line) for line in open("/vol/data/manifest_synth.jsonl", encoding="utf-8")]
        train_rows += synth[:synth_limit] if synth_limit else synth
    val_rows = [r for r in rows if r["split"] == "val"]
    print(f"train clips: {len(train_rows)} (real x{real_repeat}"
          f"{', + synthetic' if use_synth else ''}), val clips: {len(val_rows)}", flush=True)

    class Clips(torch.utils.data.Dataset):
        def __init__(self, split_rows):
            self.rows = split_rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            audio, sr = sf.read(f"/vol/data/{r['path']}", dtype="float32")
            feats = processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
            return {"input_features": feats, "labels": tok(r["text"]).input_ids}

    @dataclasses.dataclass
    class Collator:
        start_id: int

        def __call__(self, feats: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            batch = processor.feature_extractor.pad(
                [{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
            # Model weights are bfloat16; features must match (conv1 rejects float32).
            batch["input_features"] = batch["input_features"].to(torch.bfloat16)
            labels = tok.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
            ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
            if (ids[:, 0] == self.start_id).all():
                ids = ids[:, 1:]
            batch["labels"] = ids
            return batch

    def wer(refs, hyps):
        errs = total = 0
        for r, h in zip(refs, hyps):
            r, h = clean_for_content_wer(r).split(), clean_for_content_wer(h).split()
            d = list(range(len(h) + 1))
            for i, rw in enumerate(r, 1):
                prev, d[0] = d[0], i
                for j, hw in enumerate(h, 1):
                    prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            errs, total = errs + d[len(h)], total + len(r)
        return errs / max(total, 1)

    def compute_metrics(pred):
        ids = pred.predictions
        lab = pred.label_ids
        lab[lab == -100] = tok.pad_token_id
        return {"wer": wer(tok.batch_decode(lab, skip_special_tokens=True),
                           tok.batch_decode(ids, skip_special_tokens=True))}

    model = WhisperForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, apply_spec_augment=True, mask_time_prob=0.05)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=hp["rank"], lora_alpha=hp["alpha"], lora_dropout=hp["dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none"))
    model.print_trainable_parameters()

    args = Seq2SeqTrainingArguments(
        output_dir=f"/vol/runs/{name}", per_device_train_batch_size=hp["batch"],
        per_device_eval_batch_size=hp["batch"], gradient_accumulation_steps=hp["accum"],
        learning_rate=hp["lr"], warmup_steps=10, num_train_epochs=hp["epochs"],
        lr_scheduler_type="cosine", bf16=True, gradient_checkpointing=True,
        # Non-reentrant checkpointing keeps encoder LoRA weights in the autograd graph
        # (the encoder input, conv features, has no requires_grad).
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=225, logging_steps=5,
        remove_unused_columns=False, label_names=["labels"], report_to=[], dataloader_num_workers=2,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=Clips(train_rows), eval_dataset=Clips(val_rows),
        data_collator=Collator(model.config.decoder_start_token_id),
        compute_metrics=compute_metrics, processing_class=processor.feature_extractor)
    base_val = trainer.evaluate()
    print("BASE val content-WER:", base_val["eval_wer"], flush=True)
    trainer.train()
    best_val = trainer.evaluate()
    print("FT   val content-WER:", best_val["eval_wer"], flush=True)

    model.save_pretrained(f"/vol/models/{name}-adapter")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(f"/vol/models/{name}-merged", safe_serialization=True)
    processor.save_pretrained(f"/vol/models/{name}-merged")
    # faster-whisper / CTranslate2 needs the fast-tokenizer file (tokenizer.json).
    from transformers import WhisperTokenizerFast
    WhisperTokenizerFast.from_pretrained(BASE).save_pretrained(f"/vol/models/{name}-merged")
    summary = {"base_val_wer": base_val["eval_wer"], "ft_val_wer": best_val["eval_wer"],
               "hyperparameters": hp, "use_synth": use_synth, "real_repeat": real_repeat,
               "train_clips": len(train_rows), "val_clips": len(val_rows)}
    json.dump(summary, open(f"/vol/models/{name}-summary.json", "w"))
    vol.commit()  # persist the trained model before the (separate) conversion step
    print("SUMMARY:", json.dumps(summary), flush=True)

    from ctranslate2.converters import TransformersConverter
    TransformersConverter(f"/vol/models/{name}-merged",
                          copy_files=["tokenizer.json", "preprocessor_config.json"]
                          ).convert(f"/vol/models/{name}-ct2", quantization="float16", force=True)
    vol.commit()
    return summary


@app.local_entrypoint()
def main(name: str = "ft2", use_synth: bool = True, real_repeat: int = 3,
         epochs: int = 4, synth_limit: int = 0, manifest: str = "manifest.jsonl"):
    print(json.dumps(train.remote(name, use_synth, real_repeat, epochs, synth_limit, manifest), indent=2))
