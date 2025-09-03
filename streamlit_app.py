import sys
from pathlib import Path

# 👇 Add project root to sys.path (go one level up from citizen_complain_app/)
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now imports will work
import os
import streamlit as st
from utils.ui import hide_multipage_nav_css  # ✅ should now work




# ---- load env (for OFFICER_PASS) ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(
    page_title="민원 포털",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="collapsed",  # collapse by default
)

hide_multipage_nav_css()

BRAND = "#0B2F59"   # navy
ACCENT = "#103D73"
BG_SOFT = "#F6F8FB"

def _find_logo():
    for n in ("logo", "logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "logo.svg"):
        if Path(n).exists():
            return n
    return None

LOGO = _find_logo()
OFFICER_PASS = os.getenv("OFFICER_PASS", "demo1234")

def _goto(page_path: str):
    try:
        st.switch_page(page_path)  # Streamlit ≥1.25
    except Exception:
        st.page_link(page_path, label="이동하기 →")
        st.stop()

# ---- styles ----
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 0rem !important; }}

    .k-header {{
        background: {BRAND}; color: #fff; padding:.75rem 1.25rem;
        border-top-left-radius:8px; border-top-right-radius:8px;
        display:flex; align-items:center; justify-content:space-between;
    }}
    .k-card {{
        background:#fff; border:1px solid #E5EAF2; border-radius:12px;
        box-shadow:0 6px 18px rgba(10,47,89,.06); padding:1.25rem 1.25rem 1.5rem;
    }}
    .k-hero {{ text-align:center; padding:1.6rem 0 .75rem; }}
    .k-hero h1 {{ font-size:2.1rem; font-weight:800; color:{BRAND}; margin:0; }}

    /* Style the Streamlit bordered containers made by st.container(border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {BG_SOFT};
        border: 1px solid #E5EAF2;
        border-radius: 14px;
        padding: 1rem 1.1rem 1.1rem;
        box-shadow: 0 6px 18px rgba(10,47,89,.06);
    }}

    .k-tile-title {{ margin:0; font-size:1.15rem; font-weight:800; color:{BRAND}; }}
    .k-tile-sub   {{ margin:0; font-size:.95rem;  color:#374151; }}

    .k-btn-primary button {{
        background:{BRAND}!important; color:#fff!important; border-radius:999px!important;
        border:1px solid {ACCENT}!important; padding:.65rem 1.2rem!important;
    }}
    .k-btn-secondary button {{
        background:#fff!important; color:{BRAND}!important; border-radius:999px!important;
        border:1px solid {BRAND}!important; padding:.65rem 1.2rem!important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

import base64

# ---- header ----
hdr = st.container()
with hdr:
    cols = st.columns([2, 7, 3])  # ← [1,8,3] 에서 살짝 여유 주기

    with cols[0]:
        if LOGO:
            with open(LOGO, "rb") as f:
                b64_logo = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <div style="
                    display:inline-flex; align-items:center; gap:16px;
                    white-space:nowrap; flex-wrap:nowrap; overflow:visible;
                    margin-top:20px;
                ">
                    <img src="data:image/png;base64,{b64_logo}"
                         style="height:80px; width:auto; object-fit:contain;">
                    <span style="font-weight:900; font-size:2.2rem; color:#0B2F59;">민심청</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with cols[2]:
        st.markdown(
            f"""<div style="display:flex;align-items:center;justify-content:flex-end;
                     height:46px;color:{BRAND};font-weight:700;">민원 포털</div>""",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="k-header"><div style="font-weight:900;font-size:1.2rem;">불편사항 접수</div>'
    '<div>담당 부서가 확인 후 처리합니다</div></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="k-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="k-hero"><h1>원하시는 업무를 선택해주세요.</h1>'
    '<p>민원접수를 원하실 경우 ‘접수 화면으로 이동’ 버튼을 눌러주세요.</p></div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):  # ← this really wraps the content
        st.markdown('<h3 class="k-tile-title">민원접수</h3>', unsafe_allow_html=True)
        st.markdown('<p class="k-tile-sub">• 챗봇 • 음성 녹음 • 접수 확인 안내</p>', unsafe_allow_html=True)

        if st.button("접수 화면으로 이동", use_container_width=True):
            st.session_state["role"] = "citizen"
            _goto("pages/챗봇_민원_접수.py")


with c2:
    with st.container(border=True):  # ← one box for title + bullets + input + button
        st.markdown('<h3 class="k-tile-title">담당자</h3>', unsafe_allow_html=True)
        st.markdown('<p class="k-tile-sub">• 대시보드 • 분석</p>', unsafe_allow_html=True)

        pw = st.text_input("비밀번호", type="password", key="off_pw")

        if st.button("담당자 화면으로 이동", use_container_width=True):
            if pw == OFFICER_PASS:
                st.session_state["role"] = "officer"
                _goto("pages/담당자_대시보드.py")
            else:
                st.error("비밀번호가 올바르지 않습니다.")
