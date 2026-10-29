"""Example 1: Multi-option tool calling and abstention decision with FastBrain-9B."""
import torch
from fastbrain import load_release_model, systemone


def main():
    model, processor = load_release_model("TypeScript/FastBrain-9B", device="cuda", dtype=torch.bfloat16)
    request = {
        "model": "TypeScript/FastBrain-9B",
        "state": {
            "query": "Can you transfer $500 to Alice right now?",
            "tools": [
                {"name": "get_balance", "parameters": {"account_id": "string"}},
                {"name": "transfer_funds", "parameters": {"from_account": "string", "to_account": "string", "amount": "number"}},
            ],
        },
        "questions": {
            "q": {
                "type": "choice",
                "query": "Select the best next action.",
                "criteria": {
                    "a": 'Call <tool_call>{"name": "transfer_funds", "arguments": {"amount": 500}}</tool_call>',
                    "b": "Could you please specify which source account ID and Alice's recipient account ID to use?",
                    "c": "I am unable to assist with financial transactions.",
                },
            },
        },
    }
    response = systemone(model, processor, request)
    print(response)


if __name__ == "__main__":
    main()
