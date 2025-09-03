#!/usr/bin/env python
# coding: utf-8
# citizen_complain_app/inference_wrapper.py

from typing import Dict, Any, List, Optional
from pathlib import Path
import os
from collections import Counter

import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

# =============================================================================
# 0) Config
# =============================================================================
MODELS_BUCKET = st.secrets["GCS_BUCKET_MODELS"]  # e.g., "kds-hackathon-models"
GCP_PROJECT   = st.secrets["GCP_PROJECT"]
GCP_SA        = st.secrets["gcp_sa"]
KEI_DEST      = Path(st.secrets.get("KEI_BOOSTER_PATH", "/tmp/kei_booster.pkl"))
KEI_FALLBACK  = st.secrets.get("KEI_BOOSTER_URL")  # optional HTTP(s) fallback

MODEL_PREFIXES = [
    "main_model/",
    "child_models/",
    "priority_model/",
    "retriever_bert/",
    "cause_tagger/",
]

BASE_TMP = Path("/tmp")  # we will set mc.BASE_DIR = /tmp

# =============================================================================
# 1) GCS helpers
# =============================================================================
@st.cache_resource(show_spinner="Authorizing Google Cloud…")
def _gcs_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_info(GCP_SA)
    return storage.Client(credentials=creds, project=GCP_PROJECT)

def _download_blob(bucket_name: str, blob_name: str, local_path: Path) -> None:
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))

@st.cache_resource(show_spinner="Downloading model folders from GCS…")
def download_gcs_folder(bucket_name: str, prefix: str, destination_dir: str) -> List[str]:
    """
    Syncs all files under 'prefix' from GCS to 'destination_dir' (keeps the relative structure).
    Returns list of downloaded file paths.
    """
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    downloaded_files: List[str] = []

    dest_root = Path(destination_dir)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue  # Skip folder markers
        rel_path = blob.name[len(prefix):].lstrip("/")
        local_path = dest_root / prefix.strip("/") / rel_path if prefix else dest_root / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        downloaded_files.append(str(local_path))

    return downloaded_files

# =============================================================================
# 2) Ensure KEI booster in /tmp
# =============================================================================
def ensure_kei_booster() -> str:
    # If already present and non-empty, done
    if KEI_DEST.exists() and KEI_DEST.stat().st_size > 0:
        return str(KEI_DEST)

    # Try the exact bucket path first
    try:
        _download_blob(MODELS_BUCKET, "kei_booster.pkl", KEI_DEST)
        if KEI_DEST.exists() and KEI_DEST.stat().st_size > 0:
            return str(KEI_DEST)
    except Exception as e:
        st.warning(f"GCS download of KEI failed: {e}")

    # Optional fallback: HTTP(S)
    if KEI_FALLBACK:
        try:
            import requests
            r = requests.get(KEI_FALLBACK, timeout=60)
            r.raise_for_status()
            KEI_DEST.parent.mkdir(parents=True, exist_ok=True)
            KEI_DEST.write_bytes(r.content)
            if KEI_DEST.exists() and KEI_DEST.stat().st_size > 0:
                return str(KEI_DEST)
        except Exception as e:
            st.warning(f"HTTP fallback for KEI failed: {e}")

    raise FileNotFoundError("Could not fetch kei_booster.pkl from GCS or fallback URL.")

# =============================================================================
# 3) Download all required folders to /tmp
# =============================================================================
for prefix in MODEL_PREFIXES:
    # Downloads into /tmp/<prefix>/... to match mc.BASE_DIR = /tmp
    download_gcs_folder(bucket_name=MODELS_BUCKET, prefix=prefix, destination_dir=str(BASE_TMP))

# Ensure KEI before importing model_core
os.environ["KEI_BOOSTER_PATH"] = ensure_kei_booster()

# =============================================================================
# 4) Patch model_core paths (import after artifacts exist)
# =============================================================================
from citizen_complain_app import model_core as mc

# Point model_core to /tmp locations (so it loads local_files_only=True cleanly)
mc.BASE_DIR     = BASE_TMP
mc.PARENT_DIR   = mc.BASE_DIR / "main_model"
mc.CHILD_DIR    = mc.BASE_DIR / "child_models"
mc.CAUSE_DIR    = mc.BASE_DIR / "cause_tagger"
mc.PRIORITY_DIR = mc.BASE_DIR / "priority_model"
mc.RETR_DIR     = mc.BASE_DIR / "retriever_bert"
mc.RETR_INDEX   = mc.BASE_DIR / "retriever_bert"

# Patch KEI booster path
try:
    mc.KEI_PKL = KEI_DEST
except Exception:
    pass

# Optional legacy copy (some code might look for BASE_DIR/kei_booster.pkl)
try:
    legacy = mc.BASE_DIR / "kei_booster.pkl"
    if KEI_DEST.resolve() != legacy.resolve():
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(KEI_DEST.read_bytes())
except Exception:
    pass

# =============================================================================
# 5) Export inference functions
# =============================================================================
def run_full_inference(text: str, k_sim: int = 5) -> Dict[str, Any]:
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
