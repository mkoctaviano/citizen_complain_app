#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import os, json, time
from pathlib import Path
from typing import List, Optional, Dict
from collections import Counter

# -------------------------------
# Environment & perf preferences
# -------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ.get("HF_HOME", ""))

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

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
    if HAS_ST:
        val = st.secrets.get(key, None)
        if val is not None:
            return val
    return os.getenv(key, default)

def _gcs_client():
    sa_info = None
    if HAS_ST and "gcp_sa" in st.secrets:
        sa_info = dict(st.secrets["gcp_sa"])
    elif os.getenv("GCP_SA_JSON"):
        sa_info = json.loads(os.environ["GCP_SA_JSON"])
    project = _get_secret("GCP_PROJECT")
    if sa_info:
        creds = service_account.Credentials.from_service_account_info(sa_info)
        return storage.Client(credentials=creds, project=project)
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
                (dest.parent / ".download_ok").write_text("ok", encoding="utf-8")
                return
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    raise FileNotFoundError(f"Failed to fetch gs://{bucket}/{blob} -> {dest}")

def _prefix_has_content(root: Path, prefix: str) -> bool:
    target_root = root / prefix.strip("/")
    if (target_root / ".download_ok").exists():
        return True
    return any(target_root.rglob("config.json"))

def _download_prefix_if_missing(bucket: str, prefix: str, root: Path) -> List[str]:
    target_root = root / prefix.strip("/")
    if _prefix_has_content(root, prefix):
        return []
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
    try:
        (target_root / ".download_ok").write_text("ok", encoding="utf-8")
    except Exception:
        pass
    return files

# =============================
# Bootstrap: resolve models
# =============================
def bootstrap_models() -> Dict[str, Path]:
    base_tmp = Path("/tmp")
    kei_dest = Path(_get_secret("KEI_BOOSTER_PATH", str(base_tmp / "kei_booster.pkl")))
    if not _exists_nonempty(kei_dest):
        bucket = _get_secret("GCS_BUCKET_MODELS", "")
        if not bucket:
            raise RuntimeError("Missing KEI booster locally and no GCS_BUCKET_MODELS provided.")
        _download_blob(bucket, "kei_booster.pkl", kei_dest)
    os.environ["KEI_BOOSTER_PATH"] = str(kei_dest)
    bucket = _get_secret("GCS_BUCKET_MODELS", "")
    prefixes = ["main_model/", "child_models/", "priority_model/", "retriever_bert/", "cause_tagger/"]
    for pfx in prefixes:
        tgt = base_tmp / pfx.strip("/")
        if not _prefix_has_content(base_tmp, pfx) and bucket:
            _download_prefix_if_missing(bucket, pfx, base_tmp)
    os.environ.setdefault("MAIN_MODEL_DIR", str(base_tmp / "main_model"))
    os.environ.setdefault("CHILD_MODEL_DIR", str(base_tmp / "child_models"))
    os.environ.setdefault("CHILD_REG_PATH",  str(base_tmp / "child_models" / "child_registry.json"))
    os.environ.setdefault("RETR_MODEL_DIR",  str(base_tmp / "retriever_bert"))
    os.environ.setdefault("RETR_INDEX_DIR",  str(base_tmp / "retriever_bert"))
    os.environ.setdefault("CAUSE_MODEL_DIR", str(base_tmp / "cause_tagger"))
    os.environ.setdefault("PRIORITY_DIR",    str(base_tmp / "priority_model"))
    return {"base_tmp": base_tmp, "kei_path": kei_dest}

def _apply_path_overrides(mc, base_tmp: Path, kei_dest: Path) -> None:
    mapping = {
        "KEI_PKL": kei_dest,
        "BASE_DIR": base_tmp,
        "PARENT_DIR": base_tmp / "main_model",
        "CHILD_DIR": base_tmp / "child_models",
        "CAUSE_DIR": base_tmp / "cause_tagger",
        "PRIORITY_DIR": base_tmp / "priority_model",
        "RETR_DIR": base_tmp / "retriever_bert",
        "RETR_INDEX": base_tmp / "retriever_bert",
    }
    for name, value in mapping.items():
        try:
            setattr(mc, name, value)
        except Exception:
            pass

# =============================
# Cached bootstrap + import
# =============================

# Always define module globals so callers can import safely
_mc: Optional[object] = None
_init_error: Optional[BaseException] = None

def _bootstrap_and_import_uncached():
    """Attempt to import model_core and apply path overrides once."""
    paths = bootstrap_models()
    base_tmp = paths["base_tmp"]
    kei_dest = paths["kei_path"]

    import citizen_complain_app.model_core as _mc_local
    _apply_path_overrides(_mc_local, base_tmp, kei_dest)

    # Optional warmup (safe best-effort)
    try:
        _ = _mc_local.classify("워밍업 문장입니다.")
        try:
            booster, p_tok, p_mdl, p_classes, c_map = _mc_local._router_artifacts()
            if p_classes:
                demo_parent = p_classes[0]
                c_path = c_map.get(demo_parent)
                if c_path:
                    _ = _mc_local.predict_child_topk(c_path, "버스 정류장이 파손됐어요", k=1)
        except Exception:
            pass
        try: _ = _mc_local.run_cause("보도블럭 파손으로 보행이 불편합니다.")
        except Exception: pass
        try: _ = _mc_local.retriever_search("가로등 고장 신고", k=1)
        except Exception: pass
        try: _ = _mc_local.predict_priority_single("민원 처리 지연")
        except Exception: pass
    except Exception as e:
        # Warmup failing shouldn't block import
        print("[warmup skipped]", e)

    return _mc_local

# A cached getter that never throws at import time
if HAS_ST:
    @st.cache_resource(show_spinner="Initializing models… (first time only)")
    def _cached_loader():
        return _bootstrap_and_import_uncached()
else:
    def _cached_loader():
        return _bootstrap_and_import_uncached()

def _ensure_mc():
    """Ensure _mc is initialized or capture the init error; never raise here."""
    global _mc, _init_error
    if _mc is not None or _init_error is not None:
        return
    try:
        _mc = _cached_loader()
    except BaseException as e:
        _init_error = e

# Try to initialize once at import, but swallow failures so imports succeed
try:
    _ensure_mc()
except Exception as e:
    _init_error = e

# =============================
# Public API (always defined)
# =============================

def run_full_inference(text: str, k_sim: int = 5, fast: bool = False):
    """
    Stable entrypoint. Always importable.
    If models failed to init, returns an error-shaped payload instead of raising ImportError.
    """
    _ensure_mc()
    if _mc is None:
        # Return a friendly error object so the UI can render something graceful
        return {
            "keywords": [],
            "intents": {"의도": "미정"},
            "department": "",
            "subdepartment": "",
            "urgency": None,
            "emotion": None,
            "model_version": "chatbot_v1",
            "error": {
                "type": "init_failure",
                "message": "모델 초기화에 실패했습니다. 로그를 확인하세요.",
                "detail": str(_init_error) if _init_error else "",
            },
            "extra": {"router": {}, "cause": {"cause_span": "", "cause_score": 0.0}, "similarity": [], "priority": None},
        }

    # Delegate to real implementation
    try:
        return _mc.run_full_inference(text, k_sim=k_sim, fast=fast)  # newer signature
    except TypeError:
        return _mc.run_full_inference(text, k_sim=k_sim)             # legacy signature

def run_full_inference_legacy(text: str, k_sim: int = 5):
    """Older output shape, but never raises on import/init failure."""
    out_v2 = run_full_inference(text, k_sim=k_sim)
    # If init failed above, just return the error payload as-is
    if out_v2.get("error"):
        return out_v2

    keywords   = out_v2.get("keywords") or []
    router     = out_v2.get("extra", {}).get("router", {}) or {}
    dept       = router.get("상위부서") or out_v2.get("department") or "공통확인"
    subdept    = router.get("부서") or out_v2.get("subdepartment") or "공통확인"
    intent_val = router.get("의도") or out_v2.get("intents", {}).get("의도") or "미정"

    # helper mappers (safe even if _mc is None because we reached here only on success)
    reasons = []
    try:
        csv_path = getattr(_mc, "CSV_PATH", None)
        if csv_path and keywords:
            kw_votes = _mc.load_kw_votes(csv_path)
            from collections import Counter as _Counter
            votes = _Counter()
            for kw in keywords:
                for dept2, c in kw_votes.get(kw, {}).items():
                    votes[dept2] += c
            if votes:
                tot = sum(votes.values())
                top2 = votes.most_common(2)
                if len(top2) == 1 or (top2[0][1] - top2[1][1]) / max(tot, 1) < 0.15:
                    reasons.append("kw vote conflict")
    except Exception:
        pass

    pr = out_v2.get("extra", {}).get("priority")
    urg_norm = pr.get("urgency_norm") if pr else None
    emo_norm = pr.get("emotion_norm") if pr else None
    def _heuristic_emotion_label(s): return "중립" if s is None else ("불만" if s >= 0.6 else "중립")
    def _urgency_label_from_norm(s):
        if s is None: return "보통"
        if s >= 0.75: return "매우높음"
        if s >= 0.50: return "높음"
        if s >= 0.25: return "보통"
        return "낮음"

    urgency_txt = _urgency_label_from_norm(urg_norm)
    emotion_txt = _heuristic_emotion_label(emo_norm)
    similar = out_v2.get("extra", {}).get("similarity", [])
    intents_dict = {"공통확인": 1.0} if intent_val in ("", None, "미정") else {intent_val: 1.0}

    cause = out_v2.get("extra", {}).get("cause", {}) or {}
    try:
        if "sentence" not in cause and _mc and hasattr(_mc, "format_cause_sentence"):
            cause["sentence"] = _mc.format_cause_sentence(cause)
    except Exception:
        pass

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

