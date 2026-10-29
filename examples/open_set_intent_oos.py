"""Example 2: High-cardinality intent classification with Out-of-Scope (OOS) rejection."""
import torch
from fastbrain import load_release_model, systemone


def main():
    model, processor = load_release_model("TypeScript/FastBrain-9B", device="cuda", dtype=torch.bfloat16)
    request = {
        "model": "TypeScript/FastBrain-9B",
        "state": {"utterance": "How do I calibrate the warp drive plasma injectors on a starship?"},
        "questions": {
            "q1": {
                "type": "choice",
                "query": "Classify the user intent or select out_of_scope if unsupported.",
                "criteria": {
                    "option_0": "check_balance: Check bank account balance",
                    "option_1": "transfer: Transfer money between accounts",
                    "option_2": "card_declined: Report a declined credit card",
                    "option_80": "out_of_scope: Request is outside the supported banking and assistant domains",
                },
            },
        },
    }
    response = systemone(model, processor, request)
    print(response)


if __name__ == "__main__":
    main()
