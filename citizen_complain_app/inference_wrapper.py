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
def _bootstrap_and_import_uncached():
    paths = bootstrap_models()
    base_tmp = paths["base_tmp"]
    kei_dest = paths["kei_path"]
    import citizen_complain_app.model_core as mc
    _apply_path_overrides(mc, base_tmp, kei_dest)

    # 🔸 full warmup: hit all submodels once
    try:
        _ = mc.classify("간단한 워밍업 문장입니다.")
        try:
            booster, p_tok, p_mdl, p_classes, c_map = mc._router_artifacts()
            if p_classes:
                demo_parent = p_classes[0]
                c_path = c_map.get(demo_parent)
                if c_path:
                    _ = mc.predict_child_topk(c_path, "워밍업 문장", k=1)
        except Exception: pass
        try: _ = mc.run_cause("버스 정류장 파손으로 불편합니다.") 
        except Exception: pass
        try: _ = mc.retriever_search("가로등 고장 신고", k=1)
        except Exception: pass
        try: _ = mc.predict_priority_single("민원 처리 지연으로 불만이 큽니다.")
        except Exception: pass
    except Exception as e:
        print("[warmup skipped]", e)
    return mc

if HAS_ST:
    @st.cache_resource(show_spinner="Initializing models… (first time only)")
    def _bootstrap_and_import():
        return _bootstrap_and_import_uncached()
    mc = _bootstrap_and_import()
else:
    mc = _bootstrap_and_import_uncached()

# =============================
# Public API
# =============================
def run_full_inference(text: str, k_sim: int = 5, fast: bool = False):
    try:
        return mc.run_full_inference(text, k_sim=k_sim, fast=fast)
    except TypeError:
        return mc.run_full_inference(text, k_sim=k_sim)

def _heuristic_emotion_label(emotion_norm: Optional[float]) -> str:
    if emotion_norm is None: return "중립"
    return "불만" if emotion_norm >= 0.6 else "중립"

def _urgency_label_from_norm(urg_norm: Optional[float]) -> str:
    if urg_norm is None: return "보통"
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
    out_v2 = run_full_inference(text, k_sim=k_sim)
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
    try:
        if "sentence" not in cause and hasattr(mc, "format_cause_sentence"):
            cause["sentence"] = mc.format_cause_sentence(cause)
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
