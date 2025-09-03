#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import os, json, time
from pathlib import Path
from typing import List, Optional, Dict
from collections import Counter

# --------------------------------
# Optional Streamlit compatibility
# --------------------------------
try:
    import streamlit as st
    HAS_ST = True
except Exception:
    HAS_ST = False

from google.cloud import storage
from google.oauth2 import service_account


# =============================
# Secrets / env helper functions
# =============================
def _get_secret(key: str, default=None):
    """st.secrets first (if available), else env var."""
    if HAS_ST:
        val = st.secrets.get(key, None)
        if val is not None:
            return val
    return os.getenv(key, default)


def _gcs_client():
    """
    Prefer st.secrets["gcp_sa"], else env var GCP_SA_JSON, else ADC.
    Works both on Streamlit Cloud and local.
    """
    sa_info = None
    if HAS_ST and "gcp_sa" in st.secrets:
        sa_info = dict(st.secrets["gcp_sa"])
    elif os.getenv("GCP_SA_JSON"):
        sa_info = json.loads(os.environ["GCP_SA_JSON"])

    project = _get_secret("GCP_PROJECT")
    if sa_info:
        creds = service_account.Credentials.from_service_account_info(sa_info)
        return storage.Client(credentials=creds, project=project)
    # Fallback: Application Default Credentials
    return storage.Client(project=project)


# =============================
# Local FS helpers / idempotency
# =============================
def _exists_nonempty(p: Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def _download_blob(bucket: str, blob: str, dest: Path, max_retries: int = 3):
    client = _gcs_client()
    b = client.bucket(bucket).blob(blob)
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            b.download_to_filename(str(dest))
            if _exists_nonempty(dest):
                return
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    raise FileNotFoundError(f"Failed to fetch gs://{bucket}/{blob} -> {dest}")


def _download_prefix_if_missing(bucket: str, prefix: str, root: Path) -> List[str]:
    """
    Download *only if* the target folder looks empty. Returns list of files downloaded.
    Layout: <root>/<prefix>/<relative_files>
    """
    target_root = root / prefix.strip("/")
    # quick sentinel: if any config.json under this prefix exists locally, skip listing/downloading
    sentinel = list(target_root.rglob("config.json"))
    if sentinel:
        return []  # assume folder already present

    client = _gcs_client()
    bucket_obj = client.bucket(bucket)

    files = []
    for bl in bucket_obj.list_blobs(prefix=prefix):
        if bl.name.endswith("/"):
            continue
        rel = bl.name[len(prefix):].lstrip("/")
        local = target_root / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        bl.download_to_filename(str(local))
        files.append(str(local))
    return files


# =============================
# Bootstrap: resolve models
# =============================
def bootstrap_models() -> Dict[str, Path]:
    """
    Resolve runtime artifacts. Preference order:
    1) Use LOCAL dirs if already set via env (and exist).
    2) Else use /tmp and optionally download missing parts from GCS (if bucket is configured).

    Expects GCS layout:
      main_model/
      child_models/
      priority_model/
      retriever_bert/
      cause_tagger/
      kei_booster.pkl   (file)
    """
    # Allow caller to fully short-circuit by setting these envs to local paths
    base_tmp = Path("/tmp")

    # ----- KEI booster (single file) -----
    kei_dest = Path(_get_secret("KEI_BOOSTER_PATH", str(base_tmp / "kei_booster.pkl")))
    if not _exists_nonempty(kei_dest):
        bucket = _get_secret("GCS_BUCKET_MODELS", "")
        if not bucket:
            raise RuntimeError("Missing KEI booster locally and no GCS_BUCKET_MODELS provided.")
        _download_blob(bucket, "kei_booster.pkl", kei_dest)
    os.environ["KEI_BOOSTER_PATH"] = str(kei_dest)

    # ----- Model folders: only download if missing -----
    bucket = _get_secret("GCS_BUCKET_MODELS", "")  # optional (if local already present)
    prefixes = ["main_model/", "child_models/", "priority_model/", "retriever_bert/", "cause_tagger/"]
    for pfx in prefixes:
        tgt = base_tmp / pfx.strip("/")
        # If caller already mounted or set local paths with contents, we respect them via envs below.
        # Otherwise, place (or ensure) under /tmp.
        if not any(tgt.rglob("config.json")) and bucket:
            _download_prefix_if_missing(bucket, pfx, base_tmp)

    # ----- Tell model_core where to look (set defaults if not provided) -----
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

# Belt & suspenders: force the resolved paths (don’t crash if attributes missing)
for name, value in {
    "KEI_PKL": KEI_DEST,
    "BASE_DIR": BASE_TMP,
    "PARENT_DIR": BASE_TMP / "main_model",
    "CHILD_DIR": BASE_TMP / "child_models",
    "CAUSE_DIR": BASE_TMP / "cause_tagger",
    "PRIORITY_DIR": BASE_TMP / "priority_model",
    "RETR_DIR": BASE_TMP / "retriever_bert",
    "RETR_INDEX": BASE_TMP / "retriever_bert",
}.items():
    try:
        setattr(mc, name, value)
    except Exception:
        pass


# =============================
# Public API (thin passthroughs)
# =============================
def run_full_inference(text: str, k_sim: int = 5):
    """
    Primary entrypoint used by Streamlit page(s).
    Calls mc.run_full_inference (which handles parent→child→priority→cause→retriever).
    """
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
    """Optional: surface ambiguous KEI vote as a reason for 공통확인."""
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
    """
    Legacy formatter that keeps the older output shape
    while using mc.run_full_inference under the hood.
    """
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
