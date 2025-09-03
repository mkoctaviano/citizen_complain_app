#!/usr/bin/env python
# coding: utf-8
# citizen_complain_app/inference_wrapper.py

from typing import Dict, Any, List, Optional
from pathlib import Path
import os
from collections import Counter

# -------------------------------------------------------------------
# 1) Ensure KEI booster file exists BEFORE importing model_core
# -------------------------------------------------------------------

def _ensure_local_file(local_path: str, url: str) -> str:
    """
    Ensure the model file exists at local_path.
    Tries plain HTTP(S) first; if that fails and GCS creds exist, tries GCS SDK.
    """
    p = Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size > 0:
        return str(p)

    # Try unauthenticated HTTP(S)
    try:
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
        return str(p)
    except Exception:
        # Fallback: authenticated GCS download
        try:
            from google.cloud import storage
            from google.oauth2 import service_account
            import json
            try:
                import streamlit as st
                sa_info = dict(st.secrets["gcp_sa"])
                project = st.secrets.get("GCP_PROJECT")
            except Exception:
                sa_info = json.loads(os.environ["GCP_SA_JSON"])
                project = os.environ.get("GCP_PROJECT")

            parts = url.split("/", 4)
            if len(parts) < 5 or parts[2] != "storage.googleapis.com":
                raise ValueError("Unsupported GCS URL format: " + url)
            bucket = parts[3]
            blob_name = parts[4]

            creds = service_account.Credentials.from_service_account_info(sa_info)
            client = storage.Client(credentials=creds, project=project)
            client.bucket(bucket).blob(blob_name).download_to_filename(str(p))
            return str(p)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch KEI booster from {url} -> {local_path}: {e}")

def _get_paths():
    """Read KEI paths from Streamlit secrets first, then env; provide sane defaults."""
    url, path = None, None
    try:
        import streamlit as st
        url  = st.secrets.get("KEI_BOOSTER_URL", url)
        path = st.secrets.get("KEI_BOOSTER_PATH", path)
    except Exception:
        pass
    url  = url  or os.getenv("KEI_BOOSTER_URL")
    path = path or os.getenv("KEI_BOOSTER_PATH")
    path = path or "/mnt/data/kei_booster.pkl"
    if not url:
        raise RuntimeError("KEI_BOOSTER_URL not set in secrets or environment.")
    return url, path

# Ensure the file is present before model_core import
_KEI_URL, _KEI_PATH = _get_paths()
_LOCAL_KEI = _ensure_local_file(_KEI_PATH, _KEI_URL)

# Expose to env
os.environ.setdefault("KEI_BOOSTER_PATH", _LOCAL_KEI)

# -------------------------------------------------------------------
# 2) Import model_core and patch its KEI_PKL
# -------------------------------------------------------------------
from citizen_complain_app import model_core as mc

# ✅ Force model_core to use our downloaded booster file
try:
    mc.KEI_PKL = Path(_LOCAL_KEI)
except Exception:
    pass

# ✅ Optional: also copy the file to the legacy location
try:
    legacy = mc.BASE_DIR / "kei_booster.pkl"
    if Path(_LOCAL_KEI).resolve() != legacy.resolve():
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(Path(_LOCAL_KEI).read_bytes())
except Exception:
    pass

# -------------------------------------------------------------------
# v2 passthrough (preferred)
# -------------------------------------------------------------------
def run_full_inference(text: str, k_sim: int = 5) -> Dict[str, Any]:
    return mc.run_full_inference(text, k_sim=k_sim)

# -------------------------------------------------------------------
# Legacy-compatible behavior (adds KW-vote checks, label strings, reasons)
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
        # model_core doesn’t expose CSV_PATH in your paste, so guard it
        csv_path = getattr(mc, "CSV_PATH", None)
        if csv_path:
            kw_votes = mc.load_kw_votes(csv_path)
            if keywords:
                votes = Counter()
                for kw in keywords:
                    for dept, c in kw_votes.get(kw, {}).items():
                        votes[dept] += c
                if votes:
                    tot = sum(votes.values())
                    top2 = votes.most_common(2)
                    if len(top2) == 1 or (top2[0][1] - top2[1][1]) / max(tot, 1) < 0.15:
                        reasons.append("kw vote conflict")
    except Exception:
        pass
    return reasons

def run_full_inference_legacy(text: str, k_sim: int = 5) -> Dict[str, Any]:
    out_v2 = mc.run_full_inference(text, k_sim=k_sim)

    keywords   = out_v2.get("keywords") or []
    router     = out_v2.get("extra", {}).get("router", {}) or {}
    dept       = router.get("상위부서") or out_v2.get("department") or "공통확인"
    subdept    = router.get("부서") or out_v2.get("subdepartment") or "공통확인"
    intent_val = router.get("의도") or out_v2.get("intents", {}).get("의도") or "미정"

    reasons = _kw_vote_reasons(keywords)

    pr = out_v2.get("extra", {}).get("priority")
    urg_norm = pr.get("urgency_norm") if pr else None
    emo_norm = pr.get("emotion_norm") if pr else None
    urgency_txt = _urgency_label_from_norm(urg_norm)
    emotion_txt = _heuristic_emotion_label(emo_norm)

    similar = out_v2.get("extra", {}).get("similarity", [])

    intents_dict = {"공통확인": 1.0} if intent_val in ("", None, "미정") else {intent_val: 1.0}

    cause = out_v2.get("extra", {}).get("cause", {}) or {}
    if "sentence" not in cause and hasattr(mc, "format_cause_sentence"):
        cause["sentence"] = mc.format_cause_sentence(cause)

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
            "similar": similar,
            "similarity": similar,
            "reasons": reasons,
            "router": router,
            "상위부서Top2": router.get("상위부서Top2", []),
            "상위부서_후보TopK": router.get("상위부서_후보TopK", []),
            "부서_후보TopK": router.get("부서_후보TopK", []),
            "공통확인_사유": router.get("공통확인_사유", ""),
        },
    }

__all__ = [
    "run_full_inference",
    "run_full_inference_legacy",
]
