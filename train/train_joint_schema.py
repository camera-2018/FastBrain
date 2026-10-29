"""Joint backbone LoRA + JointSchemaHead training pipeline with calibrated Brier & margin objectives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def compute_calibrated_decision_loss(
    logits_list: list[torch.Tensor],
    targets_list: list[int],
    label_smoothing: float = 0.03,
    brier_weight: float = 0.25,
    margin_weight: float = 0.15,
    target_margin: float = 0.35,
) -> torch.Tensor:
    """Compute joint Cross-Entropy + Brier calibration + Decision Margin loss across all questions in a batch."""
    losses = []
    for q_logits, target_idx in zip(logits_list, targets_list):
        num_opts = q_logits.shape[-1]
        target = torch.tensor([target_idx], device=q_logits.device, dtype=torch.long)
        ce = F.cross_entropy(q_logits.unsqueeze(0).float(), target, label_smoothing=label_smoothing)

        probs = q_logits.float().softmax(dim=-1)
        one_hot = F.one_hot(target, num_classes=num_opts).float().squeeze(0)
        brier = torch.mean((probs - one_hot) ** 2)

        gold_logit = q_logits[target_idx].float()
        if num_opts > 1:
            mask = torch.ones(num_opts, dtype=torch.bool, device=q_logits.device)
            mask[target_idx] = False
            runner_up = q_logits[mask].float().max()
            margin_loss = F.relu(target_margin - (gold_logit - runner_up))
        else:
            margin_loss = torch.tensor(0.0, device=q_logits.device)

        losses.append(ce + brier_weight * brier + margin_weight * margin_loss)
    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description="Train FastBrain-9B JointSchemaHead + hybrid backbone projections.")
    parser.add_argument("--model-path", type=str, default="TypeScript/FastBrain-9B")
    parser.add_argument("--dataset", type=str, default="TypeScript/FastBrain-Instruct")
    parser.add_argument("--lr-head", type=float, default=2e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--brier-weight", type=float, default=0.25)
    parser.add_argument("--margin-weight", type=float, default=0.15)
    args = parser.parse_args()
    print("Training configuration:", json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
