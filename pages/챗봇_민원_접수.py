#!/usr/bin/env python
# coding: utf-8

# pages/챗봇_민원_접수.py (full drop-in)

import sys, os, re, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # project root

import streamlit as st
import utils.env  # ensure .env is loaded
from utils.voice import record_voice, transcribe_google
from utils.ui import hide_multipage_nav_css
from storage import init_db, 민원_등록
from citizen_complain_app.inference_wrapper import run_full_inference  # ✅ wrapper, not model_core

# ---------------- Page config ----------------
st.set_page_config(
    page_title="민원 접수",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)
hide_multipage_nav_css()



# ---------------- Styles: real chat look (bot left, user right) ----------------
st.markdown("""
<style>
/* Outer chat container border */
#conv-start + div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"]{
  border:2px solid #D8E3F6 !important;
  border-radius:16px !important;
  padding:16px !important;
  background:#fff !important;
  box-shadow:0 4px 14px rgba(11,47,89,.06);
  margin-top:8px;
}

/* Reset streamlit chat defaults */
[data-testid="stChatMessage"]{
  background:transparent !important; box-shadow:none !important;
  margin:8px 0 !important; gap:10px !important; align-items:flex-end;
  display:flex !important;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"]{
  background:transparent !important; border:none !important; box-shadow:none !important; padding:0 !important;
}

/* Horizontal writing, safe wrapping */
[data-testid="stChatMessage"] *{
  writing-mode: horizontal-tb !important;
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-wrap: anywhere !important;
}

/* Bubbles */
.bubble{
  display:inline-block; max-width:75%;
  border-radius:14px; padding:10px 14px; margin:2px 0; line-height:1.5;
  border:1px solid transparent;
}
.bubble.assistant{ background:#E9F2FF; color:#0B2F59; border-color:#CFE0FF; }
.bubble.user{ background:#0B2F59; color:#FFFFFF; border-color:#0A2540; }

/* Layout directions: assistant left, user right */
[data-testid="stChatMessage"]:has(.bubble.assistant){
  flex-direction:row !important; justify-content:flex-start !important;
}
[data-testid="stChatMessage"]:has(.bubble.user){
  flex-direction:row-reverse !important; justify-content:flex-end !important;
}
[data-testid="stChatMessage"]:has(.bubble.user) [data-testid="stChatMessageContent"]{
  display:flex !important; justify-content:flex-end !important; width:100% !important;
}
[data-testid="stChatMessage"]:has(.bubble.assistant) [data-testid="stChatMessageContent"]{
  display:flex !important; justify-content:flex-start !important; width:100% !important;
}

/* Avatar styling */
[data-testid="stChatMessageAvatar"] img{
  width:36px; height:36px; border-radius:50%; border:1px solid #E5EAF2;
}
.avatar-badge{
  font-size:10px; padding:2px 6px; border-radius:999px; margin-top:4px;
  display:inline-block; color:#fff; background:#8898AA;
}
.avatar-badge.bot{ background:#2F6FED; }
.avatar-badge.me{ background:#0B2F59; }

/* Input styling pinned under the window */
section[data-testid="stChatInput"]{
  border-top:1px solid #D8E3F6; padding:12px;
  border-radius:0 0 16px 16px; background:#fff; max-width:100%;
}
section[data-testid="stChatInput"] textarea:focus{
  outline:none !important; box-shadow:0 0 0 2px #0B2F59 !important; border-color:#0B2F59 !important;
}

/* Success alert before submit */
#ready-to-submit ~ div[data-testid="stAlert"] {
  background:#ffffff !important; border:2px solid #0B2F59 !important;
  border-radius:12px !important; box-shadow:0 6px 18px rgba(11,47,89,.06);
}
#ready-to-submit ~ div[data-testid="stAlert"] p { color:#0B2F59 !important; font-weight:600; }
#ready-to-submit ~ div[data-testid="stAlert"] svg { color:#0B2F59 !important; fill:#0B2F59 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Session state init ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role":"assistant"|"user","content":str}]
    st.session_state.answers = {"name": None, "phone": None, "address": None, "content": None}
    st.session_state.step_idx = 0
    st.session_state.submitted = False
    st.session_state.voice = None
    st.session_state.ready_to_submit = False

# ---------------- Home 버튼 ----------------
if st.button("홈으로"):
    st.session_state.clear()
    st.switch_page("streamlit_app.py")
    st.stop()

# ---------------- Validation helpers ----------------
def validate_name(x: str):
    x = x.strip()
    return (len(x) >= 1, x, "이름을 다시 입력해 주세요.")

def validate_phone(x: str):
    digits = re.sub(r"\\D", "", x)
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

# ---------------- Chat helpers ----------------
def bot_say(msg: str):
    st.session_state.chat_history.append({"role": "assistant", "content": msg})

def user_say(msg: str):
    st.session_state.chat_history.append({"role": "user", "content": msg})

# First prompt
if not st.session_state.chat_history:
    bot_say(STEPS[0]["prompt"])
# ---------------- Render chat thread ----------------
from html import escape
st.markdown('<div id="conv-start"></div>', unsafe_allow_html=True)
chat_box = st.container(border=True)

def _avatar_for(role: str):
    return "🤖" if role == "assistant" else "🙂"

with chat_box:
    for m in st.session_state.chat_history:
        role = "assistant" if m["role"] == "assistant" else "user"
        avatar = _avatar_for(role)
        with st.chat_message(role, avatar=avatar):
            text = escape(m["content"]).replace("\\n", "<br>")
            st.markdown(f'<div class="bubble {role}">{text}</div>', unsafe_allow_html=True)

# ---------------- Voice input (content step only) ----------------
VOICE_ON = True
CONTENT_STEP_IDX = 3
if VOICE_ON and st.session_state.step_idx == CONTENT_STEP_IDX:
    with st.expander("음성으로 내용 입력 (선택)"):
        use_voice = st.checkbox("음성 입력 사용하기", value=False)
        if use_voice:
            st.write("녹음 버튼을 누른 뒤 말씀해 주세요.")
            try:
                rec = record_voice(just_once=True)
            except Exception as e:
                st.error(f"마이크 오류: {e}")
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
                    st.session_state.voice = {"gs_uri": None, "duration_sec": None}  # fill if you store these
                    st.session_state.step_idx += 1
                    if st.session_state.step_idx < len(STEPS):
                        bot_say(STEPS[st.session_state.step_idx]["prompt"])
                    else:
                        st.session_state.ready_to_submit = True
                    st.rerun()

# ---------------- Chat input (guard against double input) ----------------
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
    st.markdown('<div id="ready-to-submit"></div>', unsafe_allow_html=True)
    st.success("모든 정보가 입력되었습니다. 아래 버튼을 눌러 민원을 최종 제출해 주세요.")

    if st.button("민원 제출하기"):
        try:
            content_text = st.session_state.answers["content"]
            result = run_full_inference(content_text)

            # Voice meta (optional)
            cap = st.session_state.get("voice")
            voice_meta = {}
            if isinstance(cap, dict):
                voice_meta = {k: cap.get(k) for k in ("gs_uri","duration_sec") if k in cap}
            elif hasattr(cap, "gs_uri"):
                voice_meta = {"gs_uri": getattr(cap, "gs_uri", None),
                              "duration_sec": getattr(cap, "duration_sec", None)}
            기타 = {"voice": voice_meta} if voice_meta else {}
            기타.update(result.get("extra", {}) or {})

            router = result.get("extra", {}).get("router", {}) or {}
            intents = result.get("intents", {}) or {}
            의도값 = intents.get("의도") if "의도" in intents else (next(iter(intents), "") if intents else "")

            # Save to DB
            민원번호 = 민원_등록(
                접수경로="웹",
                이름=st.session_state.answers["name"],
                연락처=st.session_state.answers["phone"],
                주소=st.session_state.answers["address"],
                내용=content_text,
                첨부경로목록=[],
                키워드=", ".join(result.get("keywords", [])),
                의도=의도값,
                상위부서=result.get("department", ""),
                부서=result.get("subdepartment", ""),
                감정=result.get("emotion", None),
                긴급도=result.get("urgency", None),
                상위부서_후보TopK=json.dumps(router.get("상위부서_후보TopK", []), ensure_ascii=False),
                부서_후보TopK=json.dumps(router.get("부서_후보TopK", []), ensure_ascii=False),
                상위부서Top2=json.dumps(router.get("상위부서Top2", []), ensure_ascii=False),
                기타=json.dumps(기타, ensure_ascii=False),
            )

            st.session_state["last_ticket_no"] = 민원번호
            st.session_state["submitted"] = True
            st.switch_page("pages/complaint_submitted.py")
            st.stop()

        except Exception as e:
            bot_say(f"죄송합니다. 접수 중 오류가 발생했습니다: {e}")
