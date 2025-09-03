#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import os
from typing import List, Optional
from collections import Counter

# Streamlit is optional (worker won't have it)
try:
    import streamlit as st
    HAS_ST = True
except Exception:
    HAS_ST = False

from google.cloud import storage
from google.oauth2 import service_account


# -----------------------------
# Secrets / env helpers
# -----------------------------
def _get_secret(key: str, default=None):
    if HAS_ST:
        val = st.secrets.get(key, None)
        if val is not None:
            return val
    return os.getenv(key, default)


def _gcs_client():
    # Prefer st.secrets["gcp_sa"], else env var GCP_SA_JSON, else ADC
    sa_info = None
    if HAS_ST and "gcp_sa" in st.secrets:
        sa_info = dict(st.secrets["gcp_sa"])
    elif os.getenv("GCP_SA_JSON"):
        import json
        sa_info = json.loads(os.environ["GCP_SA_JSON"])

    project = _get_secret("GCP_PROJECT")
    if sa_info:
        creds = service_account.Credentials.from_service_account_info(sa_info)
        return storage.Client(credentials=creds, project=project)
    # Fallback: ADC (only works if the environment provides it)
    return storage.Client(project=project)


def _download_blob(bucket: str, blob: str, dest: Path):
    client = _gcs_client()
    b = client.bucket(bucket).blob(blob)
    dest.parent.mkdir(parents=True, exist_ok=True)
    b.download_to_filename(str(dest))


def _download_prefix(bucket: str, prefix: str, root: Path) -> List[str]:
    client = _gcs_client()
    b = client.bucket(bucket)
    files = []
    for bl in b.list_blobs(prefix=prefix):
        if bl.name.endswith("/"):
            continue
        rel = bl.name[len(prefix):].lstrip("/")
        local = root / prefix.strip("/") / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        bl.download_to_filename(str(local))
        files.append(str(local))
    return files


# -----------------------------
# Bootstrap: download models to /tmp and set envs
# -----------------------------
def bootstrap_models():
    """
    Download all required artifacts to /tmp and set envs.
    Safe to call from Streamlit OR a background worker.
    """
    bucket = _get_secret("GCS_BUCKET_MODELS")  # e.g., "kds-hackathon-models"
    if not bucket:
        raise RuntimeError("Missing GCS_BUCKET_MODELS (secrets or env).")

    base_tmp = Path("/tmp")

    # 1) KEI booster
    kei_dest = Path(_get_secret("KEI_BOOSTER_PATH", str(base_tmp / "kei_booster.pkl")))
    if not (kei_dest.exists() and kei_dest.stat().st_size > 0):
        _download_blob(bucket, "kei_booster.pkl", kei_dest)
    if not kei_dest.exists() or kei_dest.stat().st_size == 0:
        raise FileNotFoundError(f"Failed to fetch gs://{bucket}/kei_booster.pkl")
    os.environ["KEI_BOOSTER_PATH"] = str(kei_dest)

    # 2) Model folders
    for prefix in ["main_model/", "child_models/", "priority_model/", "retriever_bert/", "cause_tagger/"]:
        _download_prefix(bucket, prefix, base_tmp)

    # 3) Tell model_core where to look (envs)
    os.environ.setdefault("MAIN_MODEL_DIR", str(base_tmp / "main_model"))
    os.environ.setdefault("CHILD_MODEL_DIR", str(base_tmp / "child_models"))
    os.environ.setdefault("CHILD_REG_PATH",  str(base_tmp / "child_models" / "child_registry.json"))
    os.environ.setdefault("RETR_MODEL_DIR",  str(base_tmp / "retriever_bert"))
    os.environ.setdefault("RETR_INDEX_DIR",  str(base_tmp / "retriever_bert"))
    os.environ.setdefault("CAUSE_MODEL_DIR", str(base_tmp / "cause_tagger"))
    os.environ.setdefault("PRIORITY_DIR",    str(base_tmp / "priority_model"))

    return {"base_tmp": base_tmp, "kei_path": kei_dest}


# ---- IMPORTANT: bootstrap BEFORE importing model_core ----
_paths = bootstrap_models()
BASE_TMP = _paths["base_tmp"]
KEI_DEST = _paths["kei_path"]

# Now import model_core; it reads KEI_BOOSTER_PATH and dirs from env
import citizen_complain_app.model_core as mc

# Belt & suspenders: force the resolved paths
try:
    mc.KEI_PKL = KEI_DEST
except Exception:
    pass
try:
    mc.BASE_DIR     = BASE_TMP
    mc.PARENT_DIR   = mc.BASE_DIR / "main_model"
    mc.CHILD_DIR    = mc.BASE_DIR / "child_models"
    mc.CAUSE_DIR    = mc.BASE_DIR / "cause_tagger"
    mc.PRIORITY_DIR = mc.BASE_DIR / "priority_model"
    mc.RETR_DIR     = mc.BASE_DIR / "retriever_bert"
    mc.RETR_INDEX   = mc.BASE_DIR / "retriever_bert"
except Exception:
    pass


# -----------------------------
# Public API
# -----------------------------
def run_full_inference(text: str, k_sim: int = 5):
    return mc.run_full_inference(text, k_sim=k_sim)


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


def run_full_inference_legacy(text: str, k_sim: int = 5):
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


__all__ = ["run_full_inference", "run_full_inference_legacy"]
