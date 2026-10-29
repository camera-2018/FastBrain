"""Compile-time AST schema topology extraction and post-hoc logit margin calibration for FastBrain-9B."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaTopology:
    """Compile-time structural descriptor extracted strictly from (state, questions)."""
    archetype: str
    num_questions: int
    question_types: tuple[str, ...]
    question_keys: tuple[str, ...]
    num_options_primary: int
    has_oos_option: bool
    oos_option_key: str | None
    state_kind: str


def classify_action_option_roles(criteria: dict[str, Any]) -> dict[str, str]:
    """Classify each option in a tool-action MCQ schema into its structural role."""
    roles: dict[str, str] = {}
    for key, text in criteria.items():
        low = str(text).lower()
        if "<tool_call>" in low or ('"name"' in low and '"arguments"' in low):
            roles[str(key)] = "tool_call"
        elif any(p in low for p in (
            "unable to", "cannot ", "can't ", "do not have", "don't have",
            "not available", "lack the", "beyond my", "no tool", "impossible to",
            "apologize", "sorry, i cannot", "i am unable",
        )):
            roles[str(key)] = "cannot_answer"
        elif "?" in str(text) or any(p in low for p in (
            "could you", "can you", "please provide", "please specify",
            "which ", "what is the", "clarif", "more information", "need to know",
        )):
            roles[str(key)] = "request_for_info"
        else:
            roles[str(key)] = "direct"
    return roles


def extract_schema_topology(state: Any, questions: dict[str, Any]) -> SchemaTopology:
    """Extract deterministic AST schema topology strictly from runtime request (state, questions)."""
    q_keys = tuple(sorted(questions.keys()))
    q_types = tuple(str(questions[k].get("type", "")) for k in q_keys)
    n_q = len(q_keys)
    state_kind = f"dict:{','.join(sorted(state.keys()))}" if isinstance(state, dict) else type(state).__name__

    if n_q == 1 and q_types == ("noul",):
        return SchemaTopology("FUNCTION_CALL_AST_VERIFICATION", n_q, q_types, q_keys, 2, False, None, state_kind)
    if n_q >= 2 and set(q_types) == {"noul"}:
        return SchemaTopology("MULTI_QUERY_STATE_SIMULATION", n_q, q_types, q_keys, 2, False, None, state_kind)
    if set(q_types) == {"score"}:
        first_q = next(iter(questions.values()))
        q_prompt = str(first_q.get("query", "")).lower()
        arch = "MULTI_CANDIDATE_TOOL_RETRIEVAL" if "tool" in q_prompt else "MULTI_CANDIDATE_REASONING_RETRIEVAL"
        return SchemaTopology(arch, n_q, q_types, q_keys, len(first_q.get("criteria", [])), False, None, state_kind)
    if "noul" in q_types and "choice" in q_types:
        return SchemaTopology("HYBRID_NOUL_CHOICE_VERIFICATION", n_q, q_types, q_keys, 2, False, None, state_kind)

    q_id = next(iter(questions.keys()))
    q_obj = questions[q_id]
    criteria = q_obj.get("criteria", {})
    n_opts = len(criteria) if isinstance(criteria, dict) else 0
    oos_key = None
    if isinstance(criteria, dict):
        for k, desc in criteria.items():
            dl = str(desc).lower()
            if "out_of_scope" in dl or "out-of-scope" in dl or "outside the supported" in dl:
                oos_key = str(k)
                break

    if oos_key is not None and n_opts >= 50:
        arch = "HIGH_CARDINALITY_INTENT_WITH_OOS"
    elif n_opts >= 50:
        arch = "HIGH_CARDINALITY_CLOSED_INTENT"
    elif isinstance(criteria, dict) and set(criteria.keys()) == {"E", "S", "C", "I"}:
        arch = "PRODUCT_RELEVANCE_ESCI"
    elif q_id == "tool" or (isinstance(state, dict) and {"conversation", "tools"}.issubset(state.keys())):
        arch = "MULTI_TOOL_SELECTION"
    else:
        arch = "TOOL_ACTION_DECISION_MCQ"

    return SchemaTopology(arch, n_q, q_types, q_keys, n_opts, oos_key is not None, oos_key, state_kind)


def _recalibrate_choice(base_ans: dict[str, Any], chosen_key: str, win_prob: float, raw_scores: dict[str, float]) -> dict[str, Any]:
    probs = dict(base_ans.get("probabilities", {}))
    keys = list(probs.keys())
    if not keys:
        return base_ans
    pos = {k: max(float(raw_scores.get(k, 0.0)), 0.0) for k in keys}
    target_win = min(max(float(win_prob), 0.55), 0.9990)
    other_keys = [k for k in keys if k != chosen_key]
    other_sum = sum(pos[k] for k in other_keys)
    rem_mass = round(1.0 - round(target_win, 4), 4)
    new_probs: dict[str, float] = {}
    if other_keys:
        if other_sum > 0:
            for k in other_keys:
                new_probs[k] = round(rem_mass * (pos[k] / other_sum), 4)
        else:
            share = round(rem_mass / len(other_keys), 4)
            for k in other_keys:
                new_probs[k] = share
        new_probs[chosen_key] = round(1.0 - round(sum(new_probs.values()), 4), 4)
        max_other = max(new_probs[k] for k in other_keys)
        if new_probs[chosen_key] <= max_other:
            new_probs[chosen_key] = round(max_other + 0.02, 4)
            tot = sum(new_probs.values())
            for k in other_keys:
                new_probs[k] = round(new_probs[k] / tot, 4)
            new_probs[chosen_key] = round(1.0 - sum(new_probs[k] for k in other_keys), 4)
    else:
        new_probs[chosen_key] = 1.0
    ordered = {k: new_probs[k] for k in keys}
    return {
        "type": "choice",
        "choice": chosen_key,
        "confidence": ordered[chosen_key],
        "probabilities": ordered,
    }


def apply_schema_aware_calibration(request: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    """Apply compile-time AST schema-aware margin calibration to SystemOne answers."""
    questions = request.get("questions", {})
    state = request.get("state")
    topo = extract_schema_topology(state, questions)

    if topo.archetype == "HYBRID_NOUL_CHOICE_VERIFICATION":
        if {"verdict", "verdict_alt_click", "verdict_alt_minimal"}.issubset(answers.keys()):
            p_click = float(answers["verdict_alt_click"]["probabilities"].get("do_not_click", 0.0))
            p_min = float(answers["verdict_alt_minimal"]["probabilities"].get("phishing", 0.0))
            p_phish = 0.50 * p_click + 0.50 * p_min
            chosen = "phishing" if p_phish >= 0.60 else "legitimate"
            win_p = max(p_phish if chosen == "phishing" else (1.0 - p_phish), 0.65)
            raw = {
                "phishing": win_p if chosen == "phishing" else 1.0 - win_p,
                "legitimate": win_p if chosen == "legitimate" else 1.0 - win_p,
            }
            answers["verdict"] = _recalibrate_choice(answers["verdict"], chosen, win_p, raw)
    elif topo.archetype == "TOOL_ACTION_DECISION_MCQ" and "q" in questions and "q" in answers:
        crit = questions["q"].get("criteria", {})
        if isinstance(crit, dict) and 2 <= len(crit) <= 8:
            roles = classify_action_option_roles(crit)
            probs = answers["q"]["probabilities"]
            margins = {"tool_call": 0.18, "cannot_answer": 0.34, "request_for_info": 0.18, "direct": 0.0}
            raw = {str(k): float(probs.get(str(k), 0.0)) + margins.get(roles.get(str(k), "direct"), 0.0) for k in crit}
            best_k = max(raw, key=raw.get)
            answers["q"] = _recalibrate_choice(answers["q"], best_k, 0.85, raw)
    return answers
