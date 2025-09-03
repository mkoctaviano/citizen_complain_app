#!/usr/bin/env python
# coding: utf-8

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import re
import time

import streamlit as st
import utils.env  # ensures .env is loaded
from utils.voice import record_voice, transcribe_google
from utils.ui import hide_multipage_nav_css
from storage import init_db, 민원_등록

# ---------------- Page config ----------------
st.set_page_config(
    page_title="민원 접수",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)
hide_multipage_nav_css()

st.markdown(
    """
    <style>
    /* Reset Streamlit's default chat bubble look */
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    /* === Assistant messages (left aligned, avatar left) === */
    [data-testid="stChatMessage"][data-testid="assistant"] {
        flex-direction: row !important;         /* avatar left, text right */
        justify-content: flex-start !important;
    }
    [data-testid="stChatMessage"][data-testid="assistant"] [data-testid="stChatMessageContent"] p {
        background: #E9F2FF;                    /* light blue */
        border-radius: 12px;
        padding: 8px 12px;
        margin: 0;
        display: inline-block;
        max-width: 70%;
        color: #000;                            /* dark text */
    }

    /* === User messages (right aligned, avatar right) === */
    [data-testid="stChatMessage"][data-testid="user"] {
        flex-direction: row-reverse !important; /* avatar right, text left */
        justify-content: flex-end !important;
    }
    [data-testid="stChatMessage"][data-testid="user"] [data-testid="stChatMessageContent"] p {
        background: #0B2F59;                    /* brand blue */
        color: #FFFFFF;                         /* white text */
        border-radius: 12px;
        padding: 8px 12px;
        margin: 0;
        display: inline-block;
        max-width: 70%;
    }

    /* === General spacing === */
    [data-testid="stChatMessage"] {
        gap: 6px !important;   /* tighten gap between avatar and bubble */
        margin: 6px 0 !important;
        align-items: flex-end; /* bottom align avatars with bubbles */
    }

    /* === Avatar sizing === */
    [data-testid="stChatMessageAvatar"] img {
        width: 32px; height: 32px; border-radius: 50%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Session state init ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.answers = {"name": None, "phone": None, "address": None, "content": None}
    st.session_state.step_idx = 0
    st.session_state.submitted = False
    st.session_state.voice = None
    st.session_state.ready_to_submit = False

# ---------------- Home 버튼 ----------------
if st.button("🏠 홈으로"):
    st.session_state.clear()
    st.switch_page("streamlit_app.py")
    st.stop()

# ---------------- Validation helpers ----------------
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

# ---------------- Conversation steps ----------------
STEPS = [
    {"key": "name", "prompt": "민원인 분의 성함을 알려주실 수 있을까요?", "validator": validate_name},
    {"key": "phone", "prompt": "연락 가능한 전화번호를 입력해 주시겠어요? 예) 010-1234-5678", "validator": validate_phone},
    {"key": "address", "prompt": "민원이 발생한 주소를 입력해 주세요. 보다 정확한 안내를 위해 필요합니다.", "validator": validate_address},
    {"key": "content", "prompt": "어떤 민원을 접수하고 싶으신가요? 자세히 말씀해 주시면 빠르게 도와드릴 수 있어요!", "validator": validate_content},
]

# ---------------- Init DB ----------------
init_db()

# ---------------- Chat rendering helpers ----------------
def bot_say(msg: str):
    st.session_state.chat_history.append({"role": "assistant", "content": msg})

def user_say(msg: str):
    st.session_state.chat_history.append({"role": "user", "content": msg})

# ---------------- First prompt ----------------
if not st.session_state.chat_history:
    bot_say(STEPS[0]["prompt"])

# ---------------- Render chat history ----------------
chat_box = st.container(border=True)   # this is the outlined window

with chat_box:
    for m in st.session_state.chat_history:
        role = "assistant" if m["role"] == "assistant" else "user"
        with st.chat_message(role):
            st.write(m["content"])

# ---------------- Voice input (content step only) ----------------
VOICE_ON = True
CONTENT_STEP_IDX = 3

if VOICE_ON and st.session_state.step_idx == CONTENT_STEP_IDX:
    with st.expander("🎤 음성으로 내용 입력 (선택)"):
        use_voice = st.checkbox("음성 입력 사용하기", value=False)

        if use_voice:
            st.markdown("**녹음 버튼을 누른 뒤 말씀해 주세요.**")

            try:
                rec = record_voice(just_once=True)
            except Exception as e:
                st.error(f"🎤 마이크 오류: {e}")
                rec = None

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
                    st.session_state.voice = "used"
                    st.session_state.step_idx += 1

                    if st.session_state.step_idx < len(STEPS):
                        bot_say(STEPS[st.session_state.step_idx]["prompt"])
                    else:
                        st.session_state.ready_to_submit = True
                    st.rerun()

# ---------------- Chat input (guard against double input) ----------------
if st.session_state.voice == "used":
    st.session_state.voice = None  # Reset voice flag after rerun
else:
    msg = st.chat_input("메시지를 입력하세요…")
    if msg:
        step = STEPS[st.session_state.step_idx]
        ok, val, err = step["validator"](msg)

        if not ok:
            bot_say(err)
        else:
            user_say(msg)
            st.session_state.answers[step["key"]] = val
            st.session_state.step_idx += 1

            if st.session_state.step_idx < len(STEPS):
                bot_say(STEPS[st.session_state.step_idx]["prompt"])
            else:
                st.session_state.ready_to_submit = True
            st.rerun()

# ---------------- Final submission button ----------------
if st.session_state.get("ready_to_submit") and not st.session_state.submitted:
    st.success(" 모든 정보가 입력되었습니다. 아래 버튼을 눌러 민원을 최종 제출해 주세요.")
    if st.button("민원 제출하기"):
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
                # 기타=기타,
            )

            st.session_state["last_ticket_no"] = 민원번호
            st.session_state["submitted"] = True
            st.switch_page("pages/complaint_submitted.py")
            st.stop()

        except Exception as e:
            bot_say(f"죄송합니다. 접수 중 오류가 발생했습니다: {e}")
