#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pages/complaint_submitted.py
import streamlit as st

st.set_page_config(
    page_title="민원 접수 완료",
    page_icon="✅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Hide the default sidebar and tighten top padding
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none !important; }
      .block-container { padding-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Centered, large, bold message
st.markdown(
    """
    <style>
      /* Let the main view fill vertically so we can center properly */
      html, body, [data-testid="stAppViewContainer"] { height: 100%; }

      .center-wrap{
        min-height: 80vh;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .msg{
        max-width: 820px;
        text-align: center;
        background: #EAF7EF;
        border: 1px solid #D7F2E3;
        border-radius: 16px;
        padding: 28px 32px;
        box-shadow: 0 6px 20px rgba(20,83,45,0.06);
        color: #14532D;
      }
      .msg h1{
        margin: 0 0 .6rem 0;
        font-size: 36px;     /* Big bold headline */
        font-weight: 800;
      }
      .msg p{
        margin: 0;
        font-size: 18px;     /* Comfortable paragraph size */
        line-height: 1.75;
      }
    </style>

    <div class="center-wrap">
      <div class="msg">
        <h1>민원이 정상적으로 접수되었습니다.</h1>
        <p>소중한 의견을 남겨주셔서 진심으로 감사드립니다.<br/>신속하고 성실하게 처리해드리겠습니다.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

import time
import streamlit as st

# ... your message UI above ...

import time, streamlit as st

# ... your success UI ...

time.sleep(3)
try:
    st.switch_page("pages/챗봇_민원_접수.py")  # ← use the exact filename you want
except Exception:
    st.warning("자동 이동에 실패했습니다. 아래 버튼을 눌러 이동하세요.")
    if st.button("🏠 홈으로 이동"):
        st.switch_page("pages/첫봇_민원_접수.py")
    st.stop()

