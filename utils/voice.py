# utils/voice.py
import io
import json
from typing import Optional, Tuple

import streamlit as st
from pydub import AudioSegment

from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech


# -------------------------------------------------------------------
# Recorder UI
# -------------------------------------------------------------------
def record_voice(just_once: bool = True) -> Optional[Tuple[bytes, int]]:
    """
    Simple recorder using streamlit-mic-recorder if available.
    Returns (wav_bytes, sample_rate) or None.
    """
    try:
        from streamlit_mic_recorder import mic_recorder
    except Exception:
        st.info("음성 녹음을 사용할 수 없습니다. (streamlit-mic-recorder 미설치)")
        return None

    rec = mic_recorder(
        start_prompt="녹음 시작",
        stop_prompt="녹음 중지",
        just_once=just_once,
        format="webm",   # browser-friendly
        key="mic_recorder",
    )
    if not rec or not rec.get("bytes"):
        return None

    webm_bytes = rec["bytes"]

    # Convert WEBM -> WAV (mono/16k) using pydub/ffmpeg
    audio = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    wav_bytes = buf.getvalue()
    return wav_bytes, 16000


# -------------------------------------------------------------------
# GCP Credentials loading (robust)
# -------------------------------------------------------------------
def _normalize_sa(info: dict) -> dict:
    """
    Ensure private_key has real newlines so google oauth parser accepts it.
    Handles cases where TOML kept '\\n' literally.
    """
    pk = info.get("private_key", "")
    # If it still contains literal backslash-n and no real newlines, convert:
    if "\\n" in pk and "\n" not in pk:
        pk = pk.replace("\\n", "\n")
    info["private_key"] = pk
    return info


def _load_gcp_creds_from_secrets() -> dict:
    """
    Returns a service-account dict from st.secrets.

    1) If 'GCP_SERVICE_ACCOUNT' exists (string or dict), use and parse it.
    2) Else, try to reconstruct from flattened keys (type, project_id, private_key, etc.)
       This supports the case where a repo .streamlit/secrets.toml defined raw fields.

    Raises RuntimeError with a clear message if neither path works.
    """
    # Path 1: Single JSON blob under GCP_SERVICE_ACCOUNT
    if "GCP_SERVICE_ACCOUNT" in st.secrets:
        info = st.secrets["GCP_SERVICE_ACCOUNT"]
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except Exception as e:
                raise RuntimeError(f"GCP_SERVICE_ACCOUNT exists but could not be parsed as JSON: {e}")
        elif not isinstance(info, dict):
            raise RuntimeError("GCP_SERVICE_ACCOUNT exists but is neither a JSON string nor a dict.")
        return _normalize_sa(info)

    # Path 2: flattened keys (repo secrets.toml scenario)
    needed = [
        "type", "project_id", "private_key", "client_email", "client_id",
        "auth_uri", "token_uri", "auth_provider_x509_cert_url", "client_x509_cert_url"
    ]
    missing = [k for k in needed if k not in st.secrets]
    if missing:
        raise RuntimeError(
            "st.secrets has no key 'GCP_SERVICE_ACCOUNT' and flattened keys are incomplete. "
            f"Missing: {missing}. Prefer defining a single GCP_SERVICE_ACCOUNT = \"\"\"{{...}}\"\"\" in secrets."
        )

    # Some setups accidentally used 'project_key_id' instead of 'private_key_id' — tolerate both.
    private_key_id = st.secrets.get("private_key_id") or st.secrets.get("project_key_id")

    info = {
        "type": st.secrets["type"],
        "project_id": st.secrets["project_id"],
        "private_key_id": private_key_id,
        "private_key": st.secrets["private_key"],
        "client_email": st.secrets["client_email"],
        "client_id": st.secrets["client_id"],
        "auth_uri": st.secrets["auth_uri"],
        "token_uri": st.secrets["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["client_x509_cert_url"],
        "universe_domain": st.secrets.get("universe_domain", "googleapis.com"),
    }
    return _normalize_sa(info)


def _make_speech_client() -> speech.SpeechClient:
    """
    Build a SpeechClient from secrets with strong validation errors.
    """
    try:
        info = _load_gcp_creds_from_secrets()

        # Debug checks (uncomment temporarily if you need to inspect)
        # pk = info.get("private_key", "")
        # st.write("PK starts with:", repr(pk[:30]))
        # st.write("PK ends with:", repr(pk[-30:]))
        # assert pk.startswith("-----BEGIN PRIVATE KEY-----"), "private_key BEGIN line missing/invalid"
        # assert pk.rstrip().endswith("-----END PRIVATE KEY-----"), "private_key END line missing/invalid"

        creds = service_account.Credentials.from_service_account_info(info)
        return speech.SpeechClient(credentials=creds)
    except Exception as e:
        # Keep the message explicit so UI shows a clear reason
        raise RuntimeError(f"GCP 자격 증명 로드 실패: {e}")


# -------------------------------------------------------------------
# Google STT
# -------------------------------------------------------------------
def transcribe_google(
    wav_bytes: bytes,
    sample_rate: int,
    language_code: str = "ko-KR",
    phrase_hints=None,
    timeout_sec: int = 60,
) -> str:
    """
    Send PCM16 mono WAV bytes to Google STT with a finite timeout.
    Returns transcript string ('' if nothing).
    """
    client = _make_speech_client()

    config = speech.RecognitionConfig(
        language_code=language_code,
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        model="latest_long",  # or "default"
        enable_automatic_punctuation=True,
        speech_contexts=[speech.SpeechContext(phrases=phrase_hints or [])],
    )
    audio = speech.RecognitionAudio(content=wav_bytes)

    resp = client.recognize(config=config, audio=audio, timeout=timeout_sec)

    pieces = []
    for r in resp.results:
        if r.alternatives:
            pieces.append(r.alternatives[0].transcript.strip())
    return " ".join([p for p in pieces if p]).strip()
