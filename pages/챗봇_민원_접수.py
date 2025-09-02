#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pages/챗봇_민원_접수.py
import re
import time
from pathlib import Path
import streamlit as st
import utils.env  # ensures .env is loaded
import streamlit as st
from utils.voice import record_voice, transcribe_google  # or long_transcribe_google

from utils.ui import hide_multipage_nav_css
from storage import init_db, 민원_등록

# ---------------- Page config ----------------
st.set_page_config(
    page_title="챗봇 민원 접수",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)
hide_multipage_nav_css()

# back to Home
import streamlit as st

# Only works inside multipage apps
if st.button("🏠 홈으로"):
    st.switch_page("streamlit_app.py")  # filename of the page script

st.title("🤖 챗봇 민원 접수")
st.caption("대화형으로 정보를 입력하시면 민원이 접수됩니다. 담당 부서가 확인 후 처리합니다.")

# ---------------- Init ----------------
init_db()

# -------- Validation helpers --------
def validate_name(x: str):
    x = x.strip()
    return (len(x) >= 1, x, "이름을 다시 입력해 주세요.")

def validate_phone(x: str):
    digits = re.sub(r"\D", "", x)
    if 9 <= len(digits) <= 11:
        if len(digits) == 11:
            formatted = f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
        elif len(digits) == 10:
            formatted = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        else:
            formatted = digits
        return (True, formatted, "")
    return (False, None, "연락처 형식이 올바르지 않습니다. 예) 010-1234-5678")

def validate_address(x: str):
    x = x.strip()
    if x == "" or x in {"건너뛰기", "skip"}:
        return (True, None, "")
    return (True, x, "")

def validate_content(x: str):
    text = x.strip()
    if len(text) >= 5:
        return (True, text, "")
    return (False, None, "민원 내용을 5자 이상 입력해 주세요.")

# -------- Conversation steps --------
STEPS = [
    {"key": "name", "prompt": "이름을 입력해 주세요.", "validator": validate_name},
    {"key": "phone", "prompt": "연락처를 입력해 주세요. 예) 010-1234-5678", "validator": validate_phone},
    {"key": "address", "prompt": "주소를 입력해 주세요. (없으면 '건너뛰기')", "validator": validate_address},
    {"key": "content", "prompt": "민원 내용을 입력해 주세요.", "validator": validate_content},
]
CONTENT_STEP_IDX = next(i for i, s in enumerate(STEPS) if s["key"] == "content")

# ---------------- Session state ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "answers" not in st.session_state:
    st.session_state.answers = {"name": None, "phone": None, "address": None, "content": None}
if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "voice" not in st.session_state:   # holds utils.voice.VoiceCapture
    st.session_state.voice = None

def bot_say(msg: str):
    st.session_state.chat_history.append({"role": "assistant", "content": msg})

def user_say(msg: str):
    st.session_state.chat_history.append({"role": "user", "content": msg})

# First bot message
if not st.session_state.chat_history:
    bot_say(STEPS[0]["prompt"])

# ---------------- Render chat ----------------
for m in st.session_state.chat_history:
    with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
        st.write(m["content"])

# ---------------- Submitted state ----------------
if st.session_state.submitted:
    with st.chat_message("assistant"):
        st.success(
            "민원이 정상적으로 접수되었습니다.\n\n"
            "소중한 의견을 남겨주셔서 감사합니다.\n"
            "신속하고 성실하게 처리하겠습니다."
        )
    time.sleep(2)
    st.session_state.clear()
    st.rerun()


# ---------------- Voice input (content step only) ----------------
import os
VOICE_ON = os.getenv("ENABLE_VOICE", "1") == "1"

if VOICE_ON and st.session_state.get("step_idx") is not None:
    if st.session_state.step_idx == CONTENT_STEP_IDX:
        st.markdown("**음성으로 내용을 입력하실 수 있습니다.**")

        rec = record_voice(just_once=True)  # <-- rec is defined here
        if rec is not None:
            wav_bytes, sr = rec
            st.audio(wav_bytes, format="audio/wav")

            with st.spinner("음성 인식 중..."):
                try:
                    transcript = transcribe_google(wav_bytes, sr)
                except Exception as e:
                    st.error(f"음성 인식 오류: {e}")
                    transcript = ""

            if transcript:
                user_say(transcript)
                st.session_state.answers["content"] = transcript

                # (optional silent hook)
                # try: _ = ComplaintRouter.predict(transcript)
                # except Exception: pass

                st.session_state.step_idx += 1
                if st.session_state.step_idx < len(STEPS):
                    bot_say(STEPS[st.session_state.step_idx]["prompt"])
                st.rerun()


# ---------------- Chat input ----------------
msg = st.chat_input("메시지를 입력하세요…")

# Ensure session state is initialized safely
if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

# Prevent index error before accessing STEPS
if st.session_state.step_idx >= len(STEPS):
    st.session_state.step_idx = len(STEPS) - 1  # Clamp to last valid index

if msg:
    user_say(msg)
    step = STEPS[st.session_state.step_idx]
    ok, val, err = step["validator"](msg)

    if not ok:
        bot_say(err)
    else:
        st.session_state.answers[step["key"]] = val

        # ----- Silent model call when the content step is filled -----
        if step["key"] == "content":
            try:
                text = st.session_state.answers["content"]
                # _pred = ComplaintRouter.predict(text)   # silent
                # st.session_state["model_pred"] = _pred  # optional
            except Exception:
                pass
        # ------------------------------------------------------------

        st.session_state.step_idx += 1

        if st.session_state.step_idx < len(STEPS):
            bot_say(STEPS[st.session_state.step_idx]["prompt"])
        else:
            # save and route as you already do
            try:
                cap = st.session_state.get("voice", None)
                기타 = {"voice": {"gs_uri": cap.gs_uri, "duration_sec": cap.duration_sec}} if cap else None

                민원번호 = 민원_등록(
                    접수경로="웹",
                    연락처=st.session_state.answers["phone"],
                    내용=st.session_state.answers["content"],
                    첨부경로목록=[],
                    이름=st.session_state.answers["name"],
                    주소=st.session_state.answers["address"],
                    기타=기타,
                )

                #  Clear everything after submission
                for k in ("answers", "chat_history", "step_idx", "submitted", "voice"):
                    st.session_state.pop(k, None)

                st.switch_page("pages/complaint_submitted.py")

            except Exception as e:
                bot_say(f"죄송합니다. 접수 중 오류가 발생했습니다: {e}")

# Keep chat loop going safely
if st.session_state.get("step_idx", 0) < len(STEPS):
    st.rerun()
