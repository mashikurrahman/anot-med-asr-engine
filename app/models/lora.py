#!/usr/bin/env python3
"""
Anot Medical ASR — Conformer LoRA Adapter Module
=================================================
Pure PyTorch implementation of Parameter-Efficient Fine-Tuning (LoRA)
for Conformer attention projections. Used in production deployment
to apply clinical adaptation weights without third-party dependencies.
"""

import hashlib
import math
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper around an existing frozen `nn.Linear` module.
    Computes: y = W_0(x) + (alpha / r) * B(A(dropout(x)))
    where A ~ N(0, sigma^2) and B = 0 at initialization.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {r}")
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)

        # Freeze base weights and bias
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty((r, self.in_features), dtype=base_linear.weight.dtype, device=base_linear.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((self.out_features, r), dtype=base_linear.weight.dtype, device=base_linear.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return base_out + self.scaling * lora_out


def inject_lora_adapters(
    model: nn.Module,
    target_suffixes: Tuple[str, ...] = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    freeze_early_layers: int = 12,
) -> Dict[str, Any]:
    """
    Freeze all base model parameters and replace matching `nn.Linear` submodules
    with `LoRALinear` adapters.
    `freeze_early_layers`: layers with index < freeze_early_layers remain 100% frozen
    base weights, protecting acoustic feature extraction from drift.
    """
    for param in model.parameters():
        param.requires_grad = False

    injected_names: List[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(sfx) for sfx in target_suffixes):
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "encoder" and parts[1] == "layers" and parts[2].isdigit():
                layer_idx = int(parts[2])
                if layer_idx < freeze_early_layers:
                    continue  # Keep early acoustic layers completely frozen

            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]
            lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_layer)
            injected_names.append(name)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "injected_layers_count": len(injected_names),
        "injected_layer_names": injected_names,
        "freeze_early_layers": freeze_early_layers,
        "lora_r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio_percent": round((trainable_params / total_params) * 100, 4) if total_params else 0.0,
    }


def save_lora_checkpoint(
    model: nn.Module,
    output_path: pathlib.Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save only the trainable LoRA adapter state_dict and governance metadata (< 1 MB)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lora_state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    payload = {
        "format": "anot_medasr_lora_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "lora_state_dict": lora_state,
    }
    torch.save(payload, str(output_path))
    sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "checkpoint_path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha,
        "tensors_saved": len(lora_state),
    }


def load_lora_checkpoint(model: nn.Module, checkpoint_path: pathlib.Path) -> Dict[str, Any]:
    """Load LoRA adapter weights into a LoRA-injected model."""
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    lora_state = payload["lora_state_dict"]
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    unexpected_lora = [k for k in unexpected if "lora_" in k]
    if unexpected_lora:
        raise RuntimeError(f"Unexpected LoRA keys in checkpoint: {unexpected_lora}")
    return payload.get("metadata", {})
