# #!/usr/bin/env python
# # coding: utf-8
# # citizen_complain_app/inference_wrapper.py

# from typing import Dict, Any, List, Optional
# from pathlib import Path
# import os
# from collections import Counter
# import streamlit as st

# from google.cloud import storage
# from google.oauth2 import service_account

# # -------------------------------------------------------------------
# # 1) Download full model folders from GCS to /tmp/
# # -------------------------------------------------------------------

# @st.cache_resource(show_spinner="📦 모델 로드 중... 잠시만 기다려주세요.")
# def download_gcs_folder(bucket_name: str, prefix: str, destination_dir: str) -> list[str]:
#     credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_sa"])
#     client = storage.Client(credentials=credentials, project=st.secrets["GCP_PROJECT"])
#     bucket = client.bucket(bucket_name)

#     blobs = bucket.list_blobs(prefix=prefix)
#     downloaded_files = []

#     for blob in blobs:
#         if blob.name.endswith("/"):
#             continue  # Skip folders
#         rel_path = blob.name[len(prefix):].lstrip("/")
#         local_path = Path(destination_dir) / rel_path
#         local_path.parent.mkdir(parents=True, exist_ok=True)
#         blob.download_to_filename(str(local_path))
#         downloaded_files.append(str(local_path))

#     return downloaded_files

# # 🔽 Download all required folders
# model_folders = [
#     "main_model/",
#     "child_models/",
#     "priority_model/",
#     "retriever_bert/",
#     "cause_tagger/",
# ]

# for folder in model_folders:
#     download_gcs_folder(
#         bucket_name=st.secrets["GCS_BUCKET"],
#         prefix=folder,
#         destination_dir="/tmp"
#     )

# # -------------------------------------------------------------------
# # 2) Patch KEI booster file (GCS or URL)
# # -------------------------------------------------------------------

# def _ensure_local_file(local_path: str, url: str) -> str:
#     p = Path(local_path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     if p.exists() and p.stat().st_size > 0:
#         return str(p)

#     try:
#         import requests
#         r = requests.get(url, timeout=60)
#         r.raise_for_status()
#         p.write_bytes(r.content)
#         return str(p)
#     except Exception:
#         from google.cloud import storage
#         from google.oauth2 import service_account
#         import json
#         try:
#             sa_info = dict(st.secrets["gcp_sa"])
#             project = st.secrets.get("GCP_PROJECT")
#         except Exception:
#             sa_info = json.loads(os.environ["GCP_SA_JSON"])
#             project = os.environ.get("GCP_PROJECT")

#         parts = url.split("/", 4)
#         if len(parts) < 5 or parts[2] != "storage.googleapis.com":
#             raise ValueError("Unsupported GCS URL format: " + url)
#         bucket = parts[3]
#         blob_name = parts[4]

#         creds = service_account.Credentials.from_service_account_info(sa_info)
#         client = storage.Client(credentials=creds, project=project)
#         client.bucket(bucket).blob(blob_name).download_to_filename(str(p))
#         return str(p)

# def _get_paths():
#     url  = st.secrets.get("KEI_BOOSTER_URL")
#     path = st.secrets.get("KEI_BOOSTER_PATH", "/tmp/kei_booster.pkl")
#     if not url:
#         raise RuntimeError("KEI_BOOSTER_URL not set in secrets or environment.")
#     return url, path

# # 🔽 Download and patch
# _KEI_URL, _KEI_PATH = _get_paths()
# _LOCAL_KEI = _ensure_local_file(_KEI_PATH, _KEI_URL)
# os.environ.setdefault("KEI_BOOSTER_PATH", _LOCAL_KEI)

# # -------------------------------------------------------------------
# # 3) Patch model_core paths
# # -------------------------------------------------------------------

# from citizen_complain_app import model_core as mc

# # ✅ Set BASE_DIR to match downloaded model folders
# mc.BASE_DIR = Path("/tmp")
# mc.PARENT_DIR = mc.BASE_DIR / "main_model"
# mc.CHILD_DIR = mc.BASE_DIR / "child_models"
# mc.CAUSE_DIR = mc.BASE_DIR / "cause_tagger"
# mc.PRIORITY_DIR = mc.BASE_DIR / "priority_model"
# mc.RETR_DIR = mc.BASE_DIR / "retriever_bert"
# mc.RETR_INDEX = mc.BASE_DIR / "retriever_bert"

# # ✅ Patch KEI booster path
# try:
#     mc.KEI_PKL = Path(_LOCAL_KEI)
# except Exception:
#     pass

# # Optional legacy fallback
# try:
#     legacy = mc.BASE_DIR / "kei_booster.pkl"
#     if Path(_LOCAL_KEI).resolve() != legacy.resolve():
#         legacy.parent.mkdir(parents=True, exist_ok=True)
#         legacy.write_bytes(Path(_LOCAL_KEI).read_bytes())
# except Exception:
#     pass

# # -------------------------------------------------------------------
# # 4) Export inference functions
# # -------------------------------------------------------------------

# def run_full_inference(text: str, k_sim: int = 5) -> Dict[str, Any]:
#     return mc.run_full_inference(text, k_sim=k_sim)

# def _heuristic_emotion_label(emotion_norm: Optional[float]) -> str:
#     if emotion_norm is None:
#         return "중립"
#     return "불만" if emotion_norm >= 0.6 else "중립"

# def _urgency_label_from_norm(urg_norm: Optional[float]) -> str:
#     if urg_norm is None:
#         return "보통"
#     if urg_norm >= 0.75: return "매우높음"
#     if urg_norm >= 0.50: return "높음"
#     if urg_norm >= 0.25: return "보통"
#     return "낮음"

# def _kw_vote_reasons(keywords: List[str]) -> List[str]:
#     reasons: List[str] = []
#     try:
#         csv_path = getattr(mc, "CSV_PATH", None)
#         if csv_path:
#             kw_votes = mc.load_kw_votes(csv_path)
#             if keywords:
#                 votes = Counter()
#                 for kw in keywords:
#                     for dept, c in kw_votes.get(kw, {}).items():
#                         votes[dept] += c
#                 if votes:
#                     tot = sum(votes.values())
#                     top2 = votes.most_common(2)
#                     if len(top2) == 1 or (top2[0][1] - top2[1][1]) / max(tot, 1) < 0.15:
#                         reasons.append("kw vote conflict")
#     except Exception:
#         pass
#     return reasons

# def run_full_inference_legacy(text: str, k_sim: int = 5) -> Dict[str, Any]:
#     out_v2 = mc.run_full_inference(text, k_sim=k_sim)

#     keywords   = out_v2.get("keywords") or []
#     router     = out_v2.get("extra", {}).get("router", {}) or {}
#     dept       = router.get("상위부서") or out_v2.get("department") or "공통확인"
#     subdept    = router.get("부서") or out_v2.get("subdepartment") or "공통확인"
#     intent_val = router.get("의도") or out_v2.get("intents", {}).get("의도") or "미정"

#     reasons = _kw_vote_reasons(keywords)

#     pr = out_v2.get("extra", {}).get("priority")
#     urg_norm = pr.get("urgency_norm") if pr else None
#     emo_norm = pr.get("emotion_norm") if pr else None
#     urgency_txt = _urgency_label_from_norm(urg_norm)
#     emotion_txt = _heuristic_emotion_label(emo_norm)

#     similar = out_v2.get("extra", {}).get("similarity", [])

#     intents_dict = {"공통확인": 1.0} if intent_val in ("", None, "미정") else {intent_val: 1.0}

#     cause = out_v2.get("extra", {}).get("cause", {}) or {}
#     if "sentence" not in cause and hasattr(mc, "format_cause_sentence"):
#         cause["sentence"] = mc.format_cause_sentence(cause)

#     return {
#         "keywords": keywords or ["공통확인"],
#         "intents": intents_dict,
#         "department": dept,
#         "subdepartment": subdept,
#         "urgency": urgency_txt,
#         "emotion": emotion_txt,
#         "model_version": "app_v1",
#         "extra": {
#             "priority": pr,
#             "cause": cause,
#             "similar": similar,
#             "similarity": similar,
#             "reasons": reasons,
#             "router": router,
#             "상위부서Top2": router.get("상위부서Top2", []),
#             "상위부서_후보TopK": router.get("상위부서_후보TopK", []),
#             "부서_후보TopK": router.get("부서_후보TopK", []),
#             "공통확인_사유": router.get("공통확인_사유", ""),
#         },
#     }

# __all__ = [
#     "run_full_inference",
#     "run_full_inference_legacy",
# ]


# ====TRIAL RUN

#!/usr/bin/env python
# coding: utf-8

from typing import Dict, Any, List, Optional
from pathlib import Path
import os
from collections import Counter

from google.cloud import storage
from google.oauth2 import service_account
import streamlit as st

# -----------------------------
# 1. Download main_model folder from GCS
# -----------------------------
@st.cache_resource
def download_main_model() -> str:
    bucket_name = st.secrets["GCS_BUCKET_MODELS"]
    prefix = "main_model/"
    destination_dir = "/tmp/main_model"

    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_sa"])
    client = storage.Client(credentials=credentials, project=st.secrets["GCP_PROJECT"])
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel_path = blob.name[len(prefix):].lstrip("/")
        local_path = Path(destination_dir) / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

    return destination_dir

# Trigger download
download_main_model()

# -----------------------------
# 2. Set KEI_BOOSTER_PATH from downloaded main_model
import os
import requests

def _find_kei_booster_path() -> str:
    """
    Try loading KEI booster from cloud URL (defined in secrets).
    If not downloaded yet, fetch and save locally.
    """
    try:
        import streamlit as st
        booster_url = st.secrets["KEI_BOOSTER_URL"]
        local_path = st.secrets["KEI_BOOSTER_PATH"]
    except Exception:
        booster_url = os.getenv("KEI_BOOSTER_URL", "")
        local_path = os.getenv("KEI_BOOSTER_PATH", "/tmp/kei_booster.pkl")

    if not os.path.exists(local_path):
        try:
            r = requests.get(booster_url, timeout=30)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            raise RuntimeError(f"Failed to download KEI Booster: {e}")

    return local_path

# -----------------------------
# 3. Import model_core and patch its KEI_PKL
# -----------------------------
from citizen_complain_app import model_core as mc

try:
    mc.KEI_PKL = Path(_LOCAL_KEI)
except Exception:
    pass

# -----------------------------
# 4. Main run functions
# -----------------------------
def run_full_inference(text: str, k_sim: int = 5) -> Dict[str, Any]:
    return mc.run_full_inference(text, k_sim=k_sim)

def run_full_inference_legacy(text: str, k_sim: int = 5) -> Dict[str, Any]:
    out_v2 = mc.run_full_inference(text, k_sim=k_sim)
    keywords = out_v2.get("keywords") or []
    router = out_v2.get("extra", {}).get("router", {}) or {}
    dept = router.get("상위부서") or out_v2.get("department") or "공통확인"
    subdept = router.get("부서") or out_v2.get("subdepartment") or "공통확인"
    intent_val = router.get("의도") or out_v2.get("intents", {}).get("의도") or "미정"

    def _urgency_label(norm): return "매우높음" if norm >= 0.75 else "높음" if norm >= 0.5 else "보통" if norm >= 0.25 else "낮음"
    def _emotion_label(norm): return "불만" if norm is not None and norm >= 0.6 else "중립"

    pr = out_v2.get("extra", {}).get("priority")
    urg = _urgency_label(pr.get("urgency_norm")) if pr else "보통"
    emo = _emotion_label(pr.get("emotion_norm")) if pr else "중립"

    cause = out_v2.get("extra", {}).get("cause", {})
    if "sentence" not in cause and hasattr(mc, "format_cause_sentence"):
        cause["sentence"] = mc.format_cause_sentence(cause)

    return {
        "keywords": keywords or ["공통확인"],
        "intents": {"공통확인": 1.0} if intent_val == "미정" else {intent_val: 1.0},
        "department": dept,
        "subdepartment": subdept,
        "urgency": urg,
        "emotion": emo,
        "model_version": "trial_main_model_only",
        "extra": {
            "priority": pr,
            "cause": cause,
            "similar": out_v2.get("extra", {}).get("similarity", []),
            "router": router,
        },
    }

__all__ = [
    "run_full_inference",
    "run_full_inference_legacy",
]
