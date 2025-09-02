# utils/voice.py
import io, json, base64
from typing import Optional, Tuple
import streamlit as st
from pydub import AudioSegment
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech


# ---------------- Recorder ----------------
def record_voice(just_once: bool = True) -> Optional[Tuple[bytes, int]]:
    try:
        from streamlit_mic_recorder import mic_recorder
    except Exception:
        st.info("음성 녹음을 사용할 수 없습니다. (streamlit-mic-recorder 미설치)")
        return None

    rec = mic_recorder(
        start_prompt="녹음 시작",
        stop_prompt="녹음 중지",
        just_once=just_once,
        format="webm",
        key="mic_recorder",
    )
    if not rec or not rec.get("bytes"):
        return None

    webm_bytes = rec["bytes"]
    audio = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue(), 16000


# ---------------- Secrets loading (robust) ----------------
def _pem_fix_and_validate(pem: str) -> str:
    """
    Ensure correct newline handling and base64 body for a PEM private key.
    Raises a clear error if invalid.
    """
    # Normalize line endings and escape sequences
    pem = pem.replace("\r\n", "\n").replace("\r", "\n")
    pem = pem.replace("\\n", "\n")  # if literal backslash-n remained
    pem = pem.strip()

    begin = "-----BEGIN PRIVATE KEY-----"
    end   = "-----END PRIVATE KEY-----"

    if begin not in pem or end not in pem:
        raise RuntimeError("private_key must include BEGIN/END PRIVATE KEY lines.")

    # Ensure exactly one newline after BEGIN and before END
    # and ensure the text ends with a newline
    if not pem.startswith(begin):
        # sometimes there is BOM/whitespace; trim handled above, but check anyway
        idx = pem.find(begin)
        pem = pem[idx:]
    if not pem.endswith("\n"):
        pem = pem + "\n"

    # Validate base64 body
    lines = pem.split("\n")
    try:
        bidx = lines.index(begin)
        eidx = lines.index(end)
    except ValueError:
        # in case END is last line with trailing '', adjust
        bidx = 0
        eidx = len(lines) - 1 if lines[-1] == end else lines.index(end)

    body_lines = [ln for ln in lines[bidx + 1 : eidx] if ln.strip() != ""]
    body = "".join(body_lines)

    # padding fix (if needed)
    pad = (-len(body)) % 4
    if pad:
        body += "=" * pad

    try:
        base64.b64decode(body, validate=True)
    except Exception as e:
        # show minimal, safe diagnostics
        raise RuntimeError(f"private_key base64 invalid ({type(e).__name__}: {e})")

    return "\n".join([begin] + [*body_lines] + [end, ""])


def _normalize_sa(info: dict) -> dict:
    pk = info.get("private_key", "")
    info["private_key"] = _pem_fix_and_validate(pk)
    return info


def _load_gcp_creds_from_secrets() -> dict:
    """
    Load service account from secrets in any of these forms:
      1) GCP_SERVICE_ACCOUNT (JSON string or dict)
      2) [gcp_sa] TOML section (preferred for Cloud; preserves newlines)
      3) flattened keys at root (type, project_id, private_key, ...)
    """
    # 1) Single JSON blob
    if "GCP_SERVICE_ACCOUNT" in st.secrets:
        val = st.secrets["GCP_SERVICE_ACCOUNT"]
        if isinstance(val, str):
            try:
                info = json.loads(val)
            except Exception as e:
                raise RuntimeError(f"GCP_SERVICE_ACCOUNT JSON parse error: {e}")
        elif isinstance(val, dict):
            info = dict(val)
        else:
            raise RuntimeError("GCP_SERVICE_ACCOUNT must be JSON string or dict.")
        return _normalize_sa(info)

    # 2) TOML section: [gcp_sa]
    if "gcp_sa" in st.secrets:
        info = dict(st.secrets["gcp_sa"])
        return _normalize_sa(info)

    # 3) Flattened keys
    needed = [
        "type", "project_id", "private_key", "client_email", "client_id",
        "auth_uri", "token_uri", "auth_provider_x509_cert_url", "client_x509_cert_url"
    ]
    missing = [k for k in needed if k not in st.secrets]
    if missing:
        raise RuntimeError(
            "서비스 계정 시크릿을 찾지 못했습니다. "
            "GCP_SERVICE_ACCOUNT (JSON 문자열) 또는 [gcp_sa] 섹션을 사용하세요. "
            f"부족한 키: {missing}"
        )
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
    try:
        info = _load_gcp_creds_from_secrets()
        creds = service_account.Credentials.from_service_account_info(info)
        return speech.SpeechClient(credentials=creds)
    except Exception as e:
        raise RuntimeError(f"GCP 자격 증명 로드 실패: {e}")


# ---------------- Google STT ----------------
def transcribe_google(
    wav_bytes: bytes,
    sample_rate: int,
    language_code: str = "ko-KR",
    phrase_hints=None,
    timeout_sec: int = 60,
) -> str:
    client = _make_speech_client()

    config = speech.RecognitionConfig(
        language_code=language_code,
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        model="latest_long",
        enable_automatic_punctuation=True,
        speech_contexts=[speech.SpeechContext(phrases=phrase_hints or [])],
    )
    audio = speech.RecognitionAudio(content=wav_bytes)
    resp = client.recognize(config=config, audio=audio, timeout=timeout_sec)

    pieces = []
    for r in resp.results:
        if r.alternatives:
            pieces.append(r.alternatives[0].transcript.strip())
    return " ".join(pieces).strip()
