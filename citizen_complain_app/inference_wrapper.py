#!/usr/bin/env python
# coding: utf-8
# citizen_complain_app/inference_wrapper.py

from typing import Dict, Any, List, Optional
from collections import Counter
from citizen_complain_app import model_core as mc

# -------------------------------------------------------------------
# v2 passthrough (preferred)
# -------------------------------------------------------------------
def run_full_inference(text: str, k_sim: int = 5) -> Dict[str, Any]:
    """
    Pure passthrough to model_core v2.
    Returns:
      {
        "keywords": [...],
        "intents": {"의도": "..."},
        "department": "...",
        "subdepartment": "...",
        "urgency": <float|None>,
        "emotion": <float|None>,
        "model_version": "chatbot_v1",
        "extra": {
          "router": {...},
          "cause": {...},
          "similarity": [...],
          "priority": {...|None},
        }
      }
    """
    return mc.run_full_inference(text, k_sim=k_sim)


# -------------------------------------------------------------------
# Legacy-compatible behavior (adds KW-vote checks, label strings, reasons)
# Uses only public functions from model_core v2 (no manual loaders).
# -------------------------------------------------------------------
def _heuristic_emotion_label(emotion_norm: Optional[float]) -> str:
    if emotion_norm is None:
        return "중립"
    return "불만" if emotion_norm >= 0.6 else "중립"

def _urgency_label_from_norm(urg_norm: Optional[float]) -> str:
    if urg_norm is None:
        return "보통"
    if urg_norm >= 0.75: return "매우높음"
    if urg_norm >= 0.50: return "높음"
    if urg_norm >= 0.25: return "보통"
    return "낮음"

def _kw_vote_reasons(keywords: List[str]) -> List[str]:
    reasons: List[str] = []
    try:
        kw_votes = mc.load_kw_votes(mc.CSV_PATH)
        if keywords:
            votes = Counter()
            for kw in keywords:
                for dept, c in kw_votes.get(kw, {}).items():
                    votes[dept] += c
            if votes:
                tot = sum(votes.values())
                top2 = votes.most_common(2)
                # if close contest add a reason (same heuristic you had)
                if len(top2) == 1 or (top2[0][1] - top2[1][1]) / max(tot, 1) < 0.15:
                    reasons.append("kw vote conflict")
    except Exception:
        # voting is optional; fail quiet
        pass
    return reasons

def run_full_inference_legacy(text: str, k_sim: int = 5) -> Dict[str, Any]:
    """
    Recreates the older wrapper's enriched output using v2 model_core.
    Adds: 'reasons', human-readable 'urgency'/'emotion' labels,
    and 'similar' list (alias of similarity).
    """
    # Use v2 single-call
    out_v2 = mc.run_full_inference(text, k_sim=k_sim)

    # Pull core parts
    keywords   = out_v2.get("keywords") or []
    router     = out_v2.get("extra", {}).get("router", {}) or {}
    dept       = router.get("상위부서") or out_v2.get("department") or "공통확인"
    subdept    = router.get("부서") or out_v2.get("subdepartment") or "공통확인"
    intent_val = router.get("의도") or out_v2.get("intents", {}).get("의도") or "미정"

    # KW vote reasons (optional)
    reasons = _kw_vote_reasons(keywords)

    # Numeric priority -> readable labels (legacy behavior)
    pr = out_v2.get("extra", {}).get("priority")
    urg_norm = pr.get("urgency_norm") if pr else None
    emo_norm = pr.get("emotion_norm") if pr else None
    urgency_txt = _urgency_label_from_norm(urg_norm)
    emotion_txt = _heuristic_emotion_label(emo_norm)

    # Similar (alias) for legacy callers
    similar = out_v2.get("extra", {}).get("similarity", [])

    # Intent mapping (legacy dict)
    intents_dict = {"공통확인": 1.0} if intent_val in ("", None, "미정") else {intent_val: 1.0}

    # Cause + sentence
    cause = out_v2.get("extra", {}).get("cause", {}) or {}
    if "sentence" not in cause and hasattr(mc, "format_cause_sentence"):
        # add readable sentence for convenience
        cause["sentence"] = mc.format_cause_sentence(cause)

    # Compose legacy-like structure
        # Compose legacy-like structure
    return {
        "keywords": keywords or ["공통확인"],
        "intents": intents_dict,
        "department": dept,
        "subdepartment": subdept,
        "urgency": urgency_txt,
        "emotion": emotion_txt,
        "model_version": "app_v1",
        "extra": {
            "priority": pr,
            "cause": cause,
            "similar": similar,          # alias
            "similarity": similar,       # keep both keys
            "reasons": reasons,
            "router": router,
            "상위부서Top2": router.get("상위부서Top2", []),
            "상위부서_후보TopK": router.get("상위부서_후보TopK", []),
            "부서_후보TopK": router.get("부서_후보TopK", []),
            "공통확인_사유": router.get("공통확인_사유", ""),
        },
    }
