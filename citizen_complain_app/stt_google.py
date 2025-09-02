#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# stt_google.py
import io
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account

# Streamlit import is optional (works without Streamlit too)
try:
    import streamlit as st
except Exception:
    st = None

def _to_wav_mono16k(b: bytes) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(b))
    audio = audio.set_channels(1).set_frame_rate(16000)
    out = io.BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()

def _speech_client():
    # 1) Prefer Streamlit secrets (Cloud/local with secrets.toml)
    if st and "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        return speech.SpeechClient(credentials=creds)
    # 2) Else rely on GOOGLE_APPLICATION_CREDENTIALS env var
    return speech.SpeechClient()

def transcribe_bytes(audio_bytes: bytes, language_code: str = "ko-KR", phrase_hints=None, timeout_sec: int = 1200) -> str:
    wav = _to_wav_mono16k(audio_bytes)
    client = _speech_client()
    cfg = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_automatic_punctuation=True,
        speech_contexts=[speech.SpeechContext(phrases=phrase_hints or [])] if phrase_hints else None,
    )
    op = client.long_running_recognize(config=cfg, audio=speech.RecognitionAudio(content=wav))
    resp = op.result(timeout=timeout_sec)
    return " ".join(r.alternatives[0].transcript for r in resp.results if r.alternatives).strip()

