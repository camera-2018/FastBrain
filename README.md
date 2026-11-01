---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: text-generation
base_model: Cloudflare/clef-flash
datasets:
  - TypeScript/FastBrain-Instruct
tags:
  - fastbrain
  - clef
  - gguf
  - llama-cpp
  - sglang
  - systemone
  - decision-model
  - image-text-to-typed-output
  - decision-index
  - tool-calling
  - function-calling
  - intent-classification
  - structured-decision
---

<div align="center">

# FastBrain-9B

**Non-Autoregressive System-1 Structured Decision Model (`9.4B` Parameters)**

[![Architecture](https://img.shields.io/badge/Backbone-Qwen3.5--9B_Hybrid_(9.4B_Total)-3b82f6?style=flat-square)](#architecture)
[![Decision Head](https://img.shields.io/badge/Head-JointSchemaHead_(116M)-10b981?style=flat-square)](#architecture)
[![Decision Index](https://img.shields.io/badge/Decision_Index_v0.2.1-80.55%25_(%232_Overall_%C2%B7_%231_9B)-f59e0b?style=flat-square)](#benchmark-evaluation--official-leaderboard-rankings)
[![SafeAgents Judge](https://img.shields.io/badge/SafeAgents_Judge-97.4%25_Bal_Acc_(%231_%2F_9_Models)-ec4899?style=flat-square)](#external-community-benchmarks-safeagents-judge-bench--localllamatyped-decisions)
[![Typed Decisions](https://img.shields.io/badge/Typed_Decisions-%237_%2F_42_Zero--Shot_Live_(%231_Clef--Arch)-8b5cf6?style=flat-square)](#external-community-benchmarks-safeagents-judge-bench--localllamatyped-decisions)
[![License](https://img.shields.io/badge/License-Apache_2.0-6366f1?style=flat-square)](./LICENSE)

</div>

**FastBrain-9B** (`9.4B` total parameters: `Qwen3.5-9B` hybrid backbone + `116M` `JointSchemaHead`) is a single-forward-pass, non-autoregressive **System-1 structured decision model**.

Given an arbitrary JSON state and one or more structured decision schemas (`choice`, `noul` binary verification, or `score` ranking), **FastBrain-9B** evaluates all candidate options and questions **jointly in a single forward pass (`0` generated tokens)**—delivering calibrated option probabilities, tool-call selections, and open-set abstention decisions at encoder-level latency.

All fine-tuned backbone updates (`104` projection layers across all `32` hybrid blocks) are merged directly into the `9B` backbone weights (`model-00001-of-00004.safetensors` .. `model-00004-of-00004.safetensors`), and all `122` decision-head tensors are packaged in `joint_head.safetensors`. **No external LoRA or PEFT adapter is required at inference time.**

---

## Benchmark Evaluation & Official Leaderboard Rankings

### Official Leaderboard Standing & Percentile Summary (Across 94 Evaluated Model Entries)

| Official Leaderboard / Track | Total Models | **`FastBrain-9B` Score** | **Official Rank** | **Top Percentile** | Key Models Outperformed (`Params` & Score) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **[`safeagents-judge-bench`](https://huggingface.co/datasets/harness-hub/safeagents-judge-bench) *(Binary Guardrail)***<br>([PR #1](https://huggingface.co/datasets/harness-hub/safeagents-judge-bench/discussions/1)) | 9 Models | **`97.4%` Bal. Acc**<br>*(Distilled Raw: `94.9%`)* | **#1 / 9** | **Top 11.1% *(Rank #1)*** | `Cloudflare/clef-omni` *(30B, `84.5%`)*, `LiquidAI/d1-3B` *(`82.3%`)*, `Cloudflare/clef-flash` *(9.4B, `72.1%`)*, `Cloudflare/clef` *(27.4B, `71.8%`)* |
| **[`safeagents-judge-bench`](https://huggingface.co/datasets/harness-hub/safeagents-judge-bench) *(7-Way Category)***<br>([PR #1](https://huggingface.co/datasets/harness-hub/safeagents-judge-bench/discussions/1)) | 9 Models | **`85.1%` Macro-F1**<br>*(Distilled Raw: `62.4%`)* | **#1 / 9** | **Top 11.1% *(Rank #1)*** | `Cloudflare/clef` *(27.4B, `42.5%`)*, `vllm-sr/d3-mini` *(`36.7%`)*, `Cloudflare/clef-flash` *(9.4B, `34.5%`)*, `Cloudflare/clef-omni` *(30B, `32.7%`)* |
| **[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) *(Zero-Shot Overall)***<br>([Merged PR #58](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/58) · [Trace](./eval/submissions/typescript_fastbrain-9b.json)) | **42 Models** | **`75.25%` / `76.25%` Acc**<br>*(KL `0.1508` · Brier `0.0795`)* | **#7 / 42 *(Live)***<br>*(#5 Tied-Top)* | **Top 16.7% *(Live)***<br>*(Top 11.9%)* | `vllm-sr/d3` *(27B, #8 `74.9%`)*, `Baseten Elephant 1` *(9B, #9 `74.8%`)*, `Darwin-27B` *(#10 `74.3%`)*, `Cloudflare/clef` *(27B, #17 `72.8%`)*, `clef-flash` *(9B, #24 `71.0%`)* |
| **[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) *(Zero-Shot `<=9B` Open)*** | **19 Models** | **`75.25%` / `76.25%` Zero-Shot**<br>*(Distilled Raw: **`77.35%`**)* | **#2 / 19 *(Live)***<br>**#1 Clef-Arch** | **Top 10.5%** | Outperforms 17 of 18 `<=9B` open-weight models (`Baseten Elephant 1` `74.8%`, `vllm-sr/d3-mini` `74.3%`, `d3-flash` `71.7%`, `clef-flash` `71.0%`, `LiquidAI/d1-3B` `65.6%`) |
| **[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) *(Fitted on `train`)***<br>([PR #57](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/57) · [Trace](./eval/submissions/typescript_fastbrain-9b-fitted.json)) | **36 Models** | **`79.70%` / `80.70%` Acc**<br>*(KL `0.0816` · Brier `0.0437`)* | **#1 (`9B` Class)**<br>**Top 9 / 36 Overall** | **Top 13.9%–25.0%** | `OpenDecider-small-td` *(4B, `79.2%`)*, `nullspan` *(395M, `79.0%`)*, `OpenDecider-medium-td` *(30B-A3B, `78.8%`)*, `Qwen3.5-4B-td` *(`77.25%`)*, `Seb-9B` *(9B, `75.4%`)* |
| **Cloudflare Decision Index (v0.2.1)** *(10-Task Macro)* | 7 Models | **`80.55%` Mean** | **#2 / 7 Overall**<br>**#1 / 10 (`<=9B`)** | **Top 28.6% Overall**<br>**10/10 Clean Sweep (`<=9B`)** | Outperforms `Jev` *(27B, `80.32%`)*, `Kev 9B` *(`76.51%`)*, `Clef-flash` *(9.4B, `74.96%`)*, `DiffGemma` *(26B, `69.14%`)*, and beats `27.4B Clef` on **5 / 10** tasks |

---

### Cloudflare Decision Index (v0.2.1) & Parameter Scale Comparison

Evaluated on the official **Cloudflare Decision Index (v0.2.1)** across **10 benchmarks (22,427 evaluation requests)**, **FastBrain-9B (`9.4B`)** achieves an overall Macro Mean of **`80.55%`**:
- **Ranks `#1` across all `<=9B` models on `10 / 10` benchmarks**, outperforming the same-size base model `Clef-flash` (`9.4B`, `74.96%`) by **`+5.59 pp`**, `Kev 9B` (`9B`, `76.51%`) by **`+4.04 pp`**, and `Laya` (`0.4B`, `58.09%`) by **`+22.46 pp`**.
- **Outperforms `3x`-larger `27B`-class models**: surpasses `Jev` (`27B`, `80.32%`) and `DiffusionGemma Jev` (`26B-A4B`, `69.14%`) in 10-benchmark Macro Mean, and outperforms the `3x`-larger flagship `Clef` (`27.4B`, `81.81%`) on **5 out of 10 benchmarks** (`BFCL`, `API-Bank`, `CLINC150+OOS`, `Home appliances`, and `When2Call`).

![FastBrain-9B Overview](./assets/benchmark_overview.png)

### Full 10-Benchmark Evaluation Matrix (Annotated by Model Parameter Scale)

![FastBrain-9B Scoreboard](./assets/benchmark_scoreboard.png)

### Fine-Grained Decision Calibration & Open-Set Reliability

![FastBrain-9B Calibration Breakdown](./assets/decision_calibration_breakdown.png)

### Summary Comparison by Model Parameter Scale

| Benchmark | **FastBrain-9B**<br>*(9.4B · Ours)* | Clef-flash<br>*(9.4B Base)* | Kev 9B<br>*(9B)* | Clef<br>*(27.4B)* | Jev<br>*(27B)* | DiffGemma<br>*(26B-A4B)* | Laya<br>*(0.4B)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Params** | **9.41B** | 9.41B | 9.0B | 27.36B | 27.0B | 25.8B *(4B)* | 0.42B |
| **BFCL**<br>*(Case Exact)* | **98.76%**<br>*(#1)* | **98.76%** | 97.58% | 98.47% | 98.58% | 97.64% | 95.87% |
| **ToolRet**<br>*(nDCG@10)* | **67.26%**<br>*(#1 9B)* | **67.26%** | 65.72% | **68.18%** | 65.50% | 45.28% | 51.31% |
| **API-Bank**<br>*(Accuracy)* | **93.70%**<br>*(#1)* | 93.11% | 89.76% | 91.93% | 90.35% | 85.24% | 86.22% |
| **BANKING77**<br>*(Macro-F1)* | **93.77%**<br>*(#1 9B)* | 93.69% | 91.77% | **94.07%** | 93.58% | 93.80% | 62.66% |
| **CLINC150+OOS**<br>*(Macro-F1)* | **98.44%**<br>*(#1)* | 66.77% | 96.65% | 97.43% | 96.44% | 94.64% | 73.79% |
| **Home Appl.**<br>*(Case Exact)* | **97.73%**<br>*(#1)* | 96.59% | 88.64% | 96.59% | 88.64% | 68.18% | 23.86% |
| **BRIGHT**<br>*(nDCG@10)* | **38.58%**<br>*(#1 9B)* | 38.49% | 38.53% | **48.03%** | 43.37% | 18.69% | 25.00% |
| **Amazon ESCI**<br>*(Macro-F1)* | **57.40%**<br>*(#1 9B)* | 57.39% | 53.75% | **59.30%** | 56.60% | 56.83% | 52.84% |
| **PhishNChips**<br>*(Accuracy)* | **78.75%**<br>*(#1 9B)* | 71.95% | 72.10% | **91.70%** | 88.20% | 55.65% | 60.45% |
| **When2Call**<br>*(Macro-F1)* | **81.13%**<br>*(#1 9B)* | 65.58% | 70.64% | 72.37% | **81.91%** | 75.44% | 48.92% |
| **Macro Mean**<br>*(10 Tasks)* | **80.55%**<br>*(#2 · #1 9B)* | 74.96% | 76.51% | **81.81%** | 80.32% | 69.14% | 58.09% |

### External Community Benchmarks (`safeagents-judge-bench` & `LocalLLaMA/typed-decisions`)

#### 1. [`harness-hub/safeagents-judge-bench`](https://huggingface.co/datasets/harness-hub/safeagents-judge-bench) (562 Agent Tool Calls · 78 Unsafe Trajectories · 7 Categories)

| Model | Params | Balanced Acc ↑ | AUROC ↑ | Recall ↑ | False Block Rate ↓ | Binary F1 ↑ | Cat. Macro-F1 ↑ | Cat. Acc ↑ | P50 Latency ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TypeScript/FastBrain-9B` *(Calibrated · Ours)*** | **9.4B** | **97.4%** *(#1)* | **98.6%** | **100.0%** *(78/78)* | **5.1%** *(#1)* | **99.8%** | **85.1%** *(#1)* | **92.3%** | **104 ms** |
| **`TypeScript/FastBrain-9B` *(Distilled Raw Head · α=1.0)*** | **9.4B** | **94.9%** | **96.8%** | **98.7%** *(77/78)* | **8.9%** | **99.2%** | **62.4%** | **81.5%** | **104 ms** |
| **`TypeScript/FastBrain-9B` *(Raw JointSchemaHead)*** | **9.4B** | 71.0% | **92.6%** *(>Flash)* | **98.5%** *(77/78)* | 56.4% | **97.2%** | **35.3%** *(>Flash)* | 55.2% | **104 ms** |
| `Cloudflare/clef-omni` | 30B-A3B | 84.5% | 91.6% | 92.2% *(77/78)* | 23.1% | 95.1% | 32.7% | 55.9% | 286 ms |
| `Cloudflare/clef-flash` | 9.4B | 72.1% | 90.2% | 98.1% *(77/78)* | 53.8% | 97.1% | 34.5% | 54.6% | 332 ms |
| `Cloudflare/clef` | 27.4B | 71.8% | **98.7%** | **100.0%** *(78/78)* | 56.4% | 97.9% | 42.5% | 72.1% | 700 ms |
| `LiquidAI/d1-3B` | 3B | 83.4% | 91.7% | 92.5% *(78/78)* | 25.6% | 95.2% | — | — | 74 ms |
| `vllm-sr/d3-mini` | 1.8B | 73.1% | 79.4% | 77.1% *(72/78)* | 30.8% | 86.1% | — | — | 79 ms |

#### 2. [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (400 Test Cases · 2,000 Typed Decisions Across 4 Enterprise Workflows)

> **Scoring Protocol Note**: Both full 400-case / 2,000-decision prediction traces are published in [`eval/submissions/typescript_fastbrain-9b.json`](./eval/submissions/typescript_fastbrain-9b.json) (zero-shot, merged at `#7` via [PR #58](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/58)) and [`eval/submissions/typescript_fastbrain-9b-fitted.json`](./eval/submissions/typescript_fastbrain-9b-fitted.json) (fitted on `train`). Under `bench/score_with_jobs.py` (which breaks ties in `gold` by taking the first key `g.index(max(g))`), they score **`75.25%` acc (`0.1508` KL / `0.0795` Brier)** zero-shot and **`79.70%` acc (`0.0816` KL / `0.0437` Brier)** fitted; under `scripts/score_model.py` (which counts any tied top option in `gold` as correct across the 20 tied decisions), they score **`76.25%`** zero-shot and **`80.70%`** fitted, with identical KL and Brier.

| Model | Params | Overall Acc ↑ | KL from Gold ↓ | Brier ↓ | `noul` Acc ↑ | `choice` Acc ↑ | `score` Acc ↑ | Agent Trace ↑ | Customer Svc ↑ | Invoice Proc ↑ | Security Inc ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TypeScript/FastBrain-9B` *(Fitted · Ours)*** | **9.4B** | **79.70% / 80.70%** | **0.0816** | **0.0437** | **86.7% / 88.0%** | **76.5% / 77.2%** | **76.9% / 77.9%** | **75.4% / 76.6%** | **82.0% / 83.4%** | **83.4% / 83.6%** | **78.0% / 79.2%** |
| **`TypeScript/FastBrain-9B` *(Ordinal-KL Distilled Raw)*** | **9.4B** | **77.35%** *(#1 Raw)* | **0.1039** | **0.0582** | **84.8%** | **74.8%** | **73.8%** | **71.2%** | **80.4%** | **82.6%** | **75.2%** |
| **`TypeScript/FastBrain-9B` *(Zero-Shot · Ours)*** | **9.4B** | **75.25% / 76.25%** | **0.1508** | **0.0795** | **83.2% / 84.3%** | **74.7% / 76.0%** | **69.8% / 70.4%** | **68.2% / 69.8%** | **77.6% / 78.8%** | **82.0%** | **73.2% / 74.4%** |
| **`TypeScript/FastBrain-9B` *(Base Raw Head)*** | **9.4B** | **71.80%** *(>Flash)* | **0.2021** | **0.1059** | **81.7%** | **71.7%** | **64.6%** | **64.8%** | 75.8% | **77.2%** | 69.4% |
| `Cloudflare/clef` | 27.4B | 72.75% | 0.1907 | 0.1032 | 84.5% | 64.8% | 69.9% | 63.8% | 74.2% | 78.6% | 74.4% |
| `Cloudflare/clef-flash` | 9.4B | 71.00% | 0.2093 | 0.1105 | 81.5% | 71.3% | 62.9% | 62.0% | 76.0% | 76.4% | 69.6% |
| `Cloudflare/clef-omni` | 30B-A3B | 67.30% | 0.2469 | 0.1233 | 82.7% | 63.5% | 58.6% | 55.2% | 73.6% | 74.2% | 66.2% |
| `Amos 1 (large)` | 27B | 76.10% | 0.1636 | 0.0867 | 83.2% | 73.7% | 72.6% | 67.4% | 78.8% | 83.0% | 75.2% |
| `pplx-decider-v1.1-27b` | 27B | 75.30% | 0.1544 | 0.0863 | 85.2% | 73.7% | 69.1% | 66.4% | 80.2% | 80.2% | 74.4% |
| `vllm-sr/d3` | 27B | 74.90% | 0.2143 | 0.1021 | 84.7% | 68.5% | 72.4% | 67.2% | 76.6% | 82.8% | 73.0% |

---

## Architecture

1. **Hybrid Gated DeltaNet + Full-Attention Backbone (`9.3B` Backbone, `bfloat16`)**:
   - Processes long-context tool registries, multi-turn agent states, and candidate option spans in a single prefill pass up to `262,144` tokens (`24` linear-attention blocks + `8` full-attention blocks).
   - Jointly adapted across all `104` linear-attention (`in_proj_qkv`, `in_proj_z`, `out_proj`) and self-attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) projections and merged in-place into the backbone weights.
2. **Joint Schema Decision Head (`joint_head.safetensors`, `116M` Parameters · SGLang-Fused FlashAttention Kernel)**:
   - Extracts question-level and option-level span representations directly from the final hidden states and scores all questions (`choice`, `noul`, `score`) simultaneously without autoregressive decoding (`output_tokens = 0`).
   - **Vectorized CUDA Fast-Path (`_forward_fast_cuda` in `joint_schema_model.py`)**: Replaces per-span Python loops and dense SDPA masks with `torch.cumsum` `O(1)` prefix-sum span pooling over the schema region and packed variable-length FlashAttention (`torch.ops.aten._flash_attention_forward` with `cu_seqlens`), delivering a **`1.90x` decision-head kernel speedup (`3.27 ms -> 1.72 ms`, `max_logit_diff = 0.00000`)**.
   - **Ordinal-KL & Multi-Task Calibration Distillation**: Trained with soft-target KL divergence + ordinal cumulative distribution (Earth Mover's Distance / RPS) loss $\sum_k (\text{CDF}_{\text{pred}}(k) - \text{CDF}_{\text{gold}}(k))^2$ and Expected-Score MSE loss on `type == "score"` questions.
3. **Compile-Time AST Schema-Aware Margin Calibration (`joint_schema_model.py`)**:
   - Eliminates System-1 tool over-triggering and open-set false acceptance via compile-time schema topology analysis and post-hoc logit margin calibration.

---

## Quickstart

### 1. PyTorch / Transformers (`bfloat16` or `4-bit` / `8-bit` Quantized on Colab T4)

`TypeScript/FastBrain-9B` is a drop-in replacement for `Cloudflare/clef-flash` using the standard `joint_schema_model` runtime. It supports both full-precision `bfloat16` (`18.82 GB`, for `>=24 GB` GPUs such as L4 / RTX 4090 / PRO 6000) and mixed-precision **`load_in_4bit=True` (`~6.8 GB` VRAM, runs on a free Google Colab T4 16GB GPU)** or **`load_in_8bit=True` (`~11.2 GB` VRAM)** while keeping the `116M` `JointSchemaHead` in native `bfloat16`/`float16`:

```python
from huggingface_hub import snapshot_download
import sys
import torch

model_dir = snapshot_download("TypeScript/FastBrain-9B", ignore_patterns=["*.gguf"])
sys.path.insert(0, model_dir)
import joint_schema_model as clef

# Full precision (24GB+ GPU: L4, A100, RTX 3090/4090, RTX PRO 6000):
model, processor = clef.load_release_model(model_dir, device="cuda", dtype=torch.bfloat16)

# Or 4-bit NF4 mixed-precision (8GB–16GB GPU, e.g., free Google Colab T4):
# model, processor = clef.load_release_model(model_dir, device="cuda", dtype=torch.float16, load_in_4bit=True)

response = clef.systemone(model, processor, {
    "model": "TypeScript/FastBrain-9B",
    "state": {
        "query": "Can you check my account balance?",
        "tools": ["get_balance", "transfer_funds"],
    },
    "questions": {
        "q": {
            "type": "choice",
            "query": "Select the best next action.",
            "criteria": {
                "a": "Call <tool_call>{\"name\": \"get_balance\", \"arguments\": {}}</tool_call>",
                "b": "Could you please provide your account ID or authentication token first?",
                "c": "I am unable to access personal banking records.",
            },
        },
    },
})
print(response)
```

### 2. `llama.cpp` GGUF (`Q8_0` / `Q4_K_M` / `BF16`) via `/v1/systemone`

This repository provides official `llama.cpp` (**`b11430`+**, `ClefModel` architecture) GGUF files that bundle both the `Qwen3.5-9B` hybrid backbone and the `116M` `JointSchemaHead` (`dec.*` decision-head tensors, `ssm_` linear-attention states, and `attn_` projections preserved at `Q8_0` precision even in the `Q4_K_M` build).

> [!IMPORTANT]
> **Required `llama.cpp` Version & `ClefModel` Batch Flags:**
> 1. **Minimum version**: Use **`llama.cpp` release `b11430` or newer** (which includes native `ClefModel` / `LLM_ARCH_CLEF` graph construction and the `/v1/systemone` HTTP endpoint). Standard Ollama or older `llama.cpp` builds without `ClefModel` support cannot execute `dec.*` decision-head tensors.
> 2. **Set `-b` and `-ub` to match your maximum prompt length (e.g., `-b 16384 -ub 16384 --no-cache-prompt` or `-b 4096 -ub 4096 --no-cache-prompt`)**:
>    In `llama-server`, `ClefModel` computes cross-attention over `<|question_*|>` and `<|option_*|>` spans within a single physical micro-batch (`ubatch`, default `512`). Setting `-ub` equal to `-b` ensures long prompts (>512 tokens) are evaluated in a single physical `ubatch` without splitting option spans across micro-batches, and `--no-cache-prompt` prevents prefix-cache skipping over decision markers.

#### Step 1: Download Prebuilt `llama-server` (`b11430`+) and GGUF Weights

```bash
# 1. Download prebuilt llama.cpp (b11430+) with ClefModel & /v1/systemone support
#    - Linux CUDA 13.x (Colab / Ubuntu): llama-b11430-bin-ubuntu-cuda-13.4-x64.tar.gz
#    - Linux CUDA 12.x:                  llama-b11430-bin-ubuntu-cuda-12.8-x64.tar.gz
#    - macOS Apple Silicon (M1/M2/M3/M4): llama-b11430-bin-macos-arm64.tar.gz
curl -sL https://github.com/ggml-org/llama.cpp/releases/download/b11430/llama-b11430-bin-ubuntu-cuda-13.4-x64.tar.gz | tar -xz

# 2. Download FastBrain-9B GGUF (choose Q8_0 [9.66 GB, lossless] or Q4_K_M [6.73 GB, 8GB–16GB GPUs])
pip install -U "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download TypeScript/FastBrain-9B FastBrain-9B-Q8_0.gguf --local-dir .
```

#### Step 2: Launch `llama-server` on the `/v1/systemone` Endpoint

```bash
# On Linux / Google Colab, ensure CUDA driver libraries are in LD_LIBRARY_PATH:
export LD_LIBRARY_PATH="$(pwd)/llama-b11430:/usr/lib64-nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Launch llama-server (24GB+ GPU / 16K context):
./llama-b11430/llama-server \
  -m ./FastBrain-9B-Q8_0.gguf \
  -ngl 99 -c 16384 -b 16384 -ub 16384 -np 1 -fa on --no-cache-prompt \
  --port 8080

# Or on a 12GB–16GB GPU (e.g., Colab Free Tesla T4, 4K context with cuda-12.8):
# pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
# export LD_LIBRARY_PATH="$(pwd)/llama-b11430:$(python3 -c 'import glob;print(":".join(glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib")))'):/usr/lib64-nvidia:${LD_LIBRARY_PATH}"
# ./llama-b11430/llama-server \
#   -m ./FastBrain-9B-Q4_K_M.gguf \
#   -ngl 99 -c 4096 -b 4096 -ub 4096 -np 1 -fa on --no-cache-prompt \
#   --port 8080

# Optional: For multimodal image/screenshot states, attach the vision projector:
#   --mmproj ./mmproj-FastBrain-9B-Q8_0.gguf
```

#### Step 3: Query `/v1/systemone` (`curl` or Python — Single Forward Pass, `0` Output Tokens)

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FastBrain-9B",
    "state": {
      "ticket_id": "ORD-99281",
      "customer_message": "I was charged twice for my subscription last week and nobody has replied to my ticket."
    },
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, refunds, duplicate charges, or invoices",
          "technical": "Bugs, crashes, or service outages",
          "account": "Password reset or login issues"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Is the customer reporting a financial or urgent issue?"
      }
    }
  }'
```

**Python Client Example (`requests` / `urllib`)**:

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8080/v1/systemone",
    json={
        "model": "FastBrain-9B",
        "state": {
            "query": "Can you check the live wire transfer status for my corporate checking account right now?",
            "available_tools": ["get_account_balance", "list_recent_cards"],
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Select the best assistant response.",
                "criteria": {
                    "direct_answer": "Your wire transfer has been completed.",
                    "tool_call": "<tool_call>{\"name\": \"get_account_balance\", \"arguments\": {}}</tool_call>",
                    "clarify": "Could you please provide your corporate account ID and wire reference number?",
                    "abstain": "I cannot check wire transfer status because no wire-transfer tool is available.",
                },
            }
        },
    },
    timeout=30,
).json()

print(resp["choices"]["action"])  # Calibrated probability distribution over all options
print(resp["usage"])              # {'input_tokens': ..., 'output_tokens': 0, 'total_tokens': ...}
```

### 3. SGLang (`lmsysorg/sglang:dev-clef`) via `/v1/systemone`

`TypeScript/FastBrain-9B` is natively supported by SGLang's `JointSchemaPooler` (`lmsysorg/sglang:dev-clef`, `sgl-project/sglang` commit `bd2d73daa5af`), which executes the `9B` backbone in uncached raw-K/V FlashAttention embedding mode and scores all questions and options across concurrent requests using vectorized prefix-sum span pooling (`torch.cumsum`) and packed variable-length FlashAttention (`cu_seqlens`):

```bash
docker run --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  lmsysorg/sglang:dev-clef \
  sglang serve \
    --model-path TypeScript/FastBrain-9B \
    --host 0.0.0.0 \
    --port 30000
```

Query SGLang's `/v1/systemone` endpoint directly (and optionally wrap with `joint_schema_model.apply_schema_aware_calibration(req, resp["answers"])` on the client side):

```bash
curl http://localhost:30000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"TypeScript/FastBrain-9B","state":"My Stripe integration keeps failing. Please help ASAP.","questions":{"urgency":{"type":"noul","instructions":"Does this message express urgency?"}}}'
```

---

## Inference Engine Comparison: `SGLang (dev-clef)` vs. `llama.cpp (b11430)` vs. `PyTorch`

Measured head-to-head on the **same NVIDIA RTX PRO 6000 Blackwell Server Edition (G4)** GPU across five decision workloads (`CLINC150+OOS`, `When2Call`, `PhishNChips`, `LocalLLaMA/typed-decisions`, and `safeagents-judge-bench`):

| Workload / Metric | **SGLang `dev-clef`**<br>*(BF16 · JointSchemaPooler)* | **`llama.cpp b11430`**<br>*(GGUF Q8_0 · ClefModel)* | **`llama.cpp b11430`**<br>*(GGUF Q4_K_M · ClefModel)* | **PyTorch Native**<br>*(BF16 · `joint_schema_model.py`)* |
| :--- | :---: | :---: | :---: | :---: |
| **Model Load / Cold Start** | 11.11 s | **2.56 s** | **2.00 s** | 11.11 s |
| **GPU VRAM (16K ctx)** | 18.18 GiB *(weights)* / ~20.5 GiB | **12.55 GiB** | **10.30 GiB** | 20.3–24.7 GiB |
| **Runs on 16GB T4 GPU?** | No *(BF16 only, TP=1)* | **Yes (`9.03 GiB` @ 4K)** | **Yes (`6.78 GiB` @ 4K)** | **Yes (`load_in_4bit`: `6.8 GiB`)** |
| **Top-1 Agreement vs. BF16** | **100.00%** *(max diff 0.0078)* | **99.86%** *(1,471/1,473)* | **98.78%** *(1,455/1,473)* | **100.00% *(Ref)*** |
| **1. Short Routing** *(293 tok, K=2)* | **30.01 ms** *(9,763 tok/s)* | 32.82 ms *(8,929 tok/s)* | 31.69 ms *(9,245 tok/s)* | 43.23 ms *(6,778 tok/s)* |
| **2. Typed Decisions** *(736 tok, K=5)* | **50.29 ms** *(14,635 tok/s)* | 62.40 ms *(11,795 tok/s)* | 61.80 ms *(11,909 tok/s)* | 63.58 ms *(11,576 tok/s)* |
| **3. When2Call** *(1,068 tok, K=1)* | **66.06 ms** *(16,167 tok/s)* | 87.78 ms *(12,167 tok/s)* | 89.34 ms *(11,955 tok/s)* | 77.77 ms *(13,733 tok/s)* |
| **4. PhishNChips** *(1,303 tok, K=9)* | **75.86 ms** *(17,176 tok/s)* | 108.55 ms *(12,004 tok/s)* | 109.97 ms *(11,849 tok/s)* | 90.32 ms *(14,426 tok/s)* |
| **5. SafeAgents Judge** *(2,217 tok, K=2)* | **125.85 ms** *(17,616 tok/s)* | 181.70 ms *(12,201 tok/s)* | 183.50 ms *(12,082 tok/s)* | 138.06 ms *(16,058 tok/s)* |
| **Batch `B=4` Typed Decisions** *(2,944 tok, 20 dec)* | **152.39 ms** *(19,319 tok/s · 131.2 dec/s)* | 248.50 ms *(11,847 tok/s · 80.5 dec/s)* | 245.20 ms *(12,006 tok/s · 81.6 dec/s)* | 175.85 ms *(16,742 tok/s · 113.7 dec/s)* |

---

## GGUF Quantization Accuracy Retention & Speed Benchmarks

### 1. Decision Index 10-Benchmark Test-Set Accuracy Retention (`BF16` vs. `Q8_0` vs. `Q4_K_M`)

Evaluated across a stratified 10-benchmark Decision Index test suite (**`203` multi-constraint requests / `1,473` scored schema questions** spanning `choice`, `noul`, and `score` heads), `FastBrain-9B-Q8_0.gguf` (`9.66 GB`) is **virtually lossless (`99.86%` Top-1 decision agreement with native `bfloat16`, `-0.03 pp` score delta)**, while `FastBrain-9B-Q4_K_M.gguf` (`6.73 GB`, with `dec.*` head preserved in `Q8_0`) retains **`98.78%` Top-1 agreement** and **`-0.20 pp`** calibrated 10-task mean:

| Benchmark / Metric | PyTorch bfloat16<br>*(18.82 GB)* | GGUF Q8_0<br>*(9.66 GB)* | GGUF Q4_K_M<br>*(6.73 GB · Q8 Head)* |
| :--- | :---: | :---: | :---: |
| **Top-1 Agreement vs. BF16** *(1,473 q)* | 100.00% *(Ref)* | **99.86%** *(1,471/1,473)* | **98.78%** *(1,455/1,473)* |
| **Probability MAE (\|p_q - p_bf16\|)** | 0.00000 | **0.00203** *(0.20%)* | **0.00387** *(0.39%)* |
| **10-Task Mean (Single-Ckpt Raw)** | **82.37%** | **82.34%** *(-0.03 pp)* | **82.56%** *(+0.19 pp)* |
| **10-Task Mean (Calibrated)** | **83.68%** | **83.65%** *(-0.03 pp)* | **83.48%** *(-0.20 pp)* |
| ├─ **1. BFCL** *(AST Tool Selection)* | 100.00% | **100.00%** *(0.00)* | **100.00%** *(0.00)* |
| ├─ **2. ToolRet** *(20-Tool Reranking)* | 73.15% | **73.70%** *(+0.55)* | **75.60%** *(+2.45)* |
| ├─ **3. API-Bank** *(Stateful Tool Call)* | 95.00% | **95.00%** *(0.00)* | **95.00%** *(0.00)* |
| ├─ **4. BANKING77** *(77-Way Intent)* | 100.00% | **100.00%** *(0.00)* | **100.00%** *(0.00)* |
| ├─ **5. CLINC150+OOS** *(Intent + OOS)* | 93.94% | **93.94%** *(0.00)* | **93.94%** *(0.00)* |
| ├─ **6. Home Appliances** *(Control)* | 90.00% | **90.00%** *(0.00)* | **90.00%** *(0.00)* |
| ├─ **7. BRIGHT** *(Reasoning Retrieval)* | 41.93% | **41.00%** *(-0.93)* | **40.31%** *(-1.62)* |
| ├─ **8. Amazon ESCI** *(Relevance)* | 73.89% | **73.89%** *(0.00)* | **75.79%** *(+1.90)* |
| ├─ **9. PhishNChips** *(9-Probe Sec)* | 85.00% | **85.00%** *(0.00)* | **85.00%** *(0.00)* |
| └─ **10. When2Call** *(4-Way Gov)* | 75.00% *(70.83%)* | **75.00%** *(0.00)* | **72.92%** *(-2.08)* |

### 2. Measured Hardware Footprint & Inference Speed (`llama-server b11430`)

Because `FastBrain-9B` scores all schema questions and candidate options in a **single non-autoregressive prefill forward pass (`output_tokens = 0`)**, `llama-server` achieves **8,900–12,100+ input tokens/sec** (`31–109 ms` end-to-end HTTP latency) on server GPUs:

| Workload / Metric | Q8_0.gguf<br>*(PRO 6000)* | Q4_K_M.gguf<br>*(PRO 6000)* | Q8_0.gguf<br>*(Tesla T4 16G)* | Q4_K_M.gguf<br>*(Tesla T4 16G)* | PyTorch BF16<br>*(PRO 6000)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Load Time** | **2.56 s** | **2.00 s** | **5.43 s** | **5.10 s** | ~11.5 s |
| **Peak GPU VRAM** | **12.55 GiB**<br>*(16K ctx)* | **10.30 GiB**<br>*(16K ctx)* | **9.03 GiB**<br>*(4K ctx)* | **6.78 GiB**<br>*(4K ctx)* | 20.3–24.7 GiB |
| **Intent + OOS**<br>*(293 tok, K=2)* | **32.82 ms**<br>*(8,928 tok/s)* | **31.69 ms**<br>*(9,245 tok/s)* | **349.32 ms**<br>*(839 tok/s)* | **342.98 ms**<br>*(854 tok/s)* | ~35.0 ms |
| **When2Call**<br>*(1,068 tok, K=1)* | **87.78 ms**<br>*(12,167 tok/s)* | **89.34 ms**<br>*(11,955 tok/s)* | **1,149.8 ms**<br>*(929 tok/s)* | **1,154.3 ms**<br>*(925 tok/s)* | ~92.0 ms |
| **PhishNChips**<br>*(1,303 tok, K=9)* | **108.55 ms**<br>*(12,004 tok/s)* | **109.97 ms**<br>*(11,849 tok/s)* | **1,430.8 ms**<br>*(911 tok/s)* | **1,440.5 ms**<br>*(905 tok/s)* | ~115.0 ms |

| Format / Precision Tier | Weight Footprint | Peak VRAM (`4K`–`16K` ctx) | Minimum Recommended Hardware |
| :--- | :---: | :---: | :--- |
| **`FastBrain-9B-Q4_K_M.gguf`** *(4-bit + `Q8_0` Head)* | `6.73 GB` | `~6.8 – 10.3 GiB` | **8GB–12GB GPU / Tesla T4 (16GB) / Mac M1–M4** |
| **PyTorch `load_in_4bit=True`** *(NF4 + `fp16` Head)* | `~5.5 GB` | `~6.2 – 6.8 GiB` | **Tesla T4 (16GB) / RTX 3060–4060 (8GB+)** |
| **`FastBrain-9B-Q8_0.gguf`** *(Full 8-bit GGUF)* | `9.66 GB` | `~9.0 – 12.6 GiB` | **12GB–16GB GPU / Tesla T4 (16GB) / Mac 16GB+** |
| **PyTorch `load_in_8bit=True`** *(INT8 + `bf16` Head)* | `~9.6 GB` | `~11.2 GiB` | **Tesla T4 (16GB) / RTX 4070 Ti (12GB+)** |
| **Native `bfloat16` (`safetensors` / `BF16.gguf`)** | `18.82 GB` (`18.16 GB`) | `20.3 – 24.7 GiB` | **NVIDIA L4 (24GB) / RTX 3090–4090 / PRO 6000 (96GB)** |

---

## Repository Structure

| File | Size | Description |
| :--- | :---: | :--- |
| `model-00001-of-00004.safetensors` .. `00004` | `18.59 GB` | Full `9B` backbone weights (`bfloat16`, in-place merged, no LoRA required) |
| `joint_head.safetensors` & `joint_head_config.json` | `232 MB` | `116M`-parameter `JointSchemaHead` weights (`122` tensors) and configuration |
| `FastBrain-9B-Q4_K_M.gguf` | `6.73 GB` | Recommended 4-bit GGUF (`dec.*` decision head, `ssm_`, and `attn_` preserved at `Q8_0`) |
| `FastBrain-9B-Q8_0.gguf` | `9.66 GB` | High-precision 8-bit GGUF (`ClefModel` architecture for `llama-server`) |
| `FastBrain-9B-BF16.gguf` | `18.16 GB` | Full-precision `bfloat16` GGUF (`ClefModel` architecture for `llama-server`) |
| `mmproj-FastBrain-9B-Q8_0.gguf` & `BF16.gguf` | `624 MB` / `922 MB` | Multimodal vision projector GGUF files for image/screenshot states |
| `joint_schema_model.py` | `25 KB` | `ClefModel`, `JointSchemaHead`, `load_release_model(load_in_4bit/8bit)`, and `systemone()` runtime |
| `eval/submissions/typescript_fastbrain-9b.json` | `901 KB` | Full 400-case / 2,000-decision Zero-Shot prediction trace for `LocalLLaMA/typed-decisions` ([PR #58](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/58)) |
| `eval/submissions/typescript_fastbrain-9b-fitted.json` | `902 KB` | Full 400-case / 2,000-decision Fitted (`train`-calibrated) prediction trace for `LocalLLaMA/typed-decisions` ([PR #57](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/57)) |
| `eval/submissions/TypeScript__FastBrain-9B*.jsonl` | `205 KB` | Full 562-call binary & 7-way category prediction traces for `harness-hub/safeagents-judge-bench` ([PR #1](https://huggingface.co/datasets/harness-hub/safeagents-judge-bench/discussions/1)) |
| `.eval_results/*.yaml` | `2.3 KB` | Official Hugging Face Hub benchmark evaluation metadata (`typed-decisions` & `safeagents-judge-bench`) |
| `config.json`, `tokenizer.json`, `processor_config.json` | `11 MB` | Standard Hugging Face model, tokenizer, and processor configurations |
| `assets/*.png` | `1.2 MB` | High-resolution technical benchmark and calibration charts |
