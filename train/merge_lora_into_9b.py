"""Merge 104 backbone LoRA layers and 122 JointSchemaHead tensors in-place into 9B safetensors shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def merge_checkpoint_in_place(
    base_model_dir: Path,
    adapter_safetensors: Path,
    head_safetensors: Path,
    output_dir: Path,
    lora_alpha: float = 16.0,
    lora_r: float = 8.0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_state = load_file(str(adapter_safetensors))
    head_state = load_file(str(head_safetensors))

    lora_pairs = {}
    for k, v in adapter_state.items():
        if k.endswith(".lora_A.weight"):
            prefix = k[:-len(".lora_A.weight")]
            base_key = prefix[len("base_model.model."):] + ".weight" if prefix.startswith("base_model.model.") else prefix + ".weight"
            lora_pairs[base_key] = (v, adapter_state[prefix + ".lora_B.weight"])

    scale = float(lora_alpha) / float(lora_r)

    for item in base_model_dir.iterdir():
        if item.name.startswith(".") and item.name != ".gitattributes":
            continue
        if item.is_dir():
            continue
        if item.name == "joint_head.safetensors":
            orig_jh = load_file(str(item))
            jh_merged = {k: head_state[k].to(dtype=orig_jh[k].dtype) for k in orig_jh}
            save_file(jh_merged, str(output_dir / "joint_head.safetensors"))
        elif item.name.endswith(".safetensors") and item.name.startswith("model"):
            continue
        else:
            shutil.copy2(item, output_dir / item.name)

    index_data = json.loads((base_model_dir / "model.safetensors.index.json").read_text())
    shard_names = sorted(set(index_data["weight_map"].values()))

    merged_layers = 0
    max_delta = 0.0
    for sname in shard_names:
        tensors = load_file(str(base_model_dir / sname))
        out_tensors = {}
        for k, w in tensors.items():
            match_key = k if k in lora_pairs else ("model." + k if ("model." + k) in lora_pairs else None)
            if match_key is not None:
                lora_A, lora_B = lora_pairs[match_key]
                delta = scale * (lora_B.float() @ lora_A.float())
                max_delta = max(max_delta, float(delta.abs().max()))
                out_tensors[k] = (w.float() + delta).to(dtype=w.dtype)
                merged_layers += 1
            else:
                out_tensors[k] = w
        save_file(out_tensors, str(output_dir / sname))

    return {
        "lora_layers_merged_in_place": merged_layers,
        "lora_scale_alpha_over_r": scale,
        "max_weight_delta_abs": max_delta,
        "joint_head_sha256": sha256_file(output_dir / "joint_head.safetensors"),
    }


def main():
    parser = argparse.ArgumentParser(description="Merge FastBrain-9B LoRA and JointSchemaHead in-place into 9B shards.")
    parser.add_argument("--base-dir", type=Path, required=True, help="Path to base 9B directory")
    parser.add_argument("--adapter", type=Path, required=True, help="Path to adapter_model.safetensors")
    parser.add_argument("--head", type=Path, required=True, help="Path to head.safetensors")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for LoRA-free release")
    args = parser.parse_args()
    receipt = merge_checkpoint_in_place(args.base_dir, args.adapter, args.head, args.output_dir)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
