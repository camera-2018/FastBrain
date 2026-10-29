"""Example 3: Multi-question security inspection combining choice and binary (noul) verification."""
import torch
from fastbrain import load_release_model, systemone


def main():
    model, processor = load_release_model("TypeScript/FastBrain-9B", device="cuda", dtype=torch.bfloat16)
    request = {
        "model": "TypeScript/FastBrain-9B",
        "state": {
            "subject": "URGENT: Your payroll direct deposit has been suspended",
            "sender": "hr-update@corp-security-verify.net",
            "body": "Click https://corp-security-verify.net/login within 2 hours to re-enter your SSO password.",
        },
        "questions": {
            "verdict": {
                "type": "choice",
                "query": "Classify the email as legitimate or phishing.",
                "criteria": {"legitimate": "Legitimate email", "phishing": "Phishing or malicious email"},
            },
            "verdict_alt_click": {
                "type": "choice",
                "query": "Should the user click the link?",
                "criteria": {"safe_to_click": "Safe to click", "do_not_click": "Do not click"},
            },
            "verdict_alt_minimal": {
                "type": "choice",
                "query": "Minimal security verdict.",
                "criteria": {"legitimate": "Legitimate", "phishing": "Phishing"},
            },
        },
    }
    response = systemone(model, processor, request)
    print(response)


if __name__ == "__main__":
    main()
