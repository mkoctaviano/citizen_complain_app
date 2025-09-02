# utils/voice.py
import io
import json
import time
from typing import Optional, Tuple

import streamlit as st
from pydub import AudioSegment

from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech

# ---------- record (UI) ----------
def record_voice(just_once: bool = True):
    """
    Simple recorder using streamlit-mic-recorder if available.
    Returns (wav_bytes, sample_rate) or None.
    """
    try:
        from streamlit_mic_recorder import mic_recorder
    except Exception:
        st.info("음성 녹음을 사용할 수 없습니다. (streamlit-mic-recorder 미설치)")
        return None

    # show recorder widget
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


# ---------- google STT ----------
def _make_speech_client():
    # Build credentials from Secrets (no file path!)
    try:
        info = st.secrets["GCP_SERVICE_ACCOUNT"]
        if isinstance(info, str):
            info = json.loads(info)
        creds = service_account.Credentials.from_service_account_info(info)
        return speech.SpeechClient(credentials=creds)
    except Exception as e:
        # Surface a clear error (don’t hang silently)
        raise RuntimeError(f"GCP 자격 증명 로드 실패: {e}")

def transcribe_google(wav_bytes: bytes, sample_rate: int, language_code="ko-KR", phrase_hints=None, timeout_sec: int = 60) -> str:
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

    # Hard timeout so UI won’t spin forever
    resp = client.recognize(config=config, audio=audio, timeout=timeout_sec)

    pieces = []
    for r in resp.results:
        if r.alternatives:
            pieces.append(r.alternatives[0].transcript.strip())
    return " ".join([p for p in pieces if p]).strip()
