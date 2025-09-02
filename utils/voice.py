#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils/voice.py
from __future__ import annotations
from typing import Optional, Tuple, Sequence
import os, json

from streamlit_mic_recorder import mic_recorder

from google.cloud import speech
from google.oauth2 import service_account

# Optional hard-coded fallback (leave "" if you don't want a hard-coded path)
SERVICE_ACCOUNT_PATH = r"C:\Users\user\Downloads\kds-hackathon-e9a639a38935.json"


def _load_credentials() -> Optional[service_account.Credentials]:
    """
    Build credentials in the safest order, without using st.secrets:
      1) GCP_SERVICE_ACCOUNT_JSON env (JSON string)
      2) GOOGLE_APPLICATION_CREDENTIALS env (path to JSON file)
      3) SERVICE_ACCOUNT_PATH (hard-coded fallback, if provided & exists)
      4) None -> let google library use ADC (Application Default Credentials)
    """
    # 1) JSON string in env (useful for CI or when no file path available)
    info_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if info_json:
        try:
            info = json.loads(info_json)
            return service_account.Credentials.from_service_account_info(info)
        except Exception:
            pass  # fall through

    # 2) Path from env
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if path and os.path.exists(path):
        try:
            return service_account.Credentials.from_service_account_file(path)
        except Exception:
            pass

    # 3) Hard-coded fallback (optional)
    if SERVICE_ACCOUNT_PATH and os.path.exists(SERVICE_ACCOUNT_PATH):
        try:
            return service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
        except Exception:
            pass

    # 4) None -> let SpeechClient() pick up ADC if available
    return None


def _speech_client() -> speech.SpeechClient:
    creds = _load_credentials()
    if creds is not None:
        return speech.SpeechClient(credentials=creds)
    return speech.SpeechClient()


def record_voice(just_once: bool = True) -> Optional[Tuple[bytes, int]]:
    audio = mic_recorder(
        start_prompt="녹음 시작",
        stop_prompt="녹음 종료",
        just_once=just_once,
        format="wav",   # LINEAR16 PCM wrapped in WAV (what Google expects)
        key="voice_recorder",
    )
    if audio and audio.get("bytes"):
        return audio["bytes"], int(audio.get("sample_rate", 16000))
    return None


def transcribe_google(
    wav_bytes: bytes,
    sample_rate: int,
    language_code: str = "ko-KR",
    alt_langs: Sequence[str] = ("en-US",),
) -> str:
    client = _speech_client()
    audio = speech.RecognitionAudio(content=wav_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        alternative_language_codes=list(alt_langs),
        enable_automatic_punctuation=True,
        use_enhanced=True,
        model="default",
    )
    resp = client.recognize(config=config, audio=audio)
    parts = [r.alternatives[0].transcript for r in resp.results if r.alternatives]
    return " ".join(parts).strip()

