"""Run FastBrain-9B evaluation on the Cloudflare Decision Index (v0.2.1 · 22,427 requests across 10 benchmarks)."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import torch
from fastbrain import load_release_model, systemone


def evaluate_rows(model_path: str, rows_gz: Path, output_gz: Path, device: str = "cuda") -> None:
    model, processor = load_release_model(model_path, device=device, dtype=torch.bfloat16)
    output_gz.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(rows_gz, "rt", encoding="utf-8") as src, gzip.open(output_gz, "wt", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            req = {
                "model": "TypeScript/FastBrain-9B",
                "state": row["state"],
                "questions": row["questions"],
            }
            resp = systemone(model, processor, req)
            out_rec = {
                "run_id": row["_evaluation"]["run_id"],
                "catalog_id": row["_evaluation"]["catalog_id"],
                "engine": "FastBrain-9B",
                "response": resp,
            }
            dst.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"Completed {count} evaluation requests -> {output_gz}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FastBrain-9B on Decision Index v0.2.1.")
    parser.add_argument("--model", type=str, default="TypeScript/FastBrain-9B")
    parser.add_argument("--rows", type=Path, required=True, help="Path to rows-0.2.1.jsonl.gz")
    parser.add_argument("--output", type=Path, default=Path("results.jsonl.gz"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    evaluate_rows(args.model, args.rows, args.output, device=args.device)


if __name__ == "__main__":
    main()
