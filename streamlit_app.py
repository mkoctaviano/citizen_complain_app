#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# streamlit_app.py
import streamlit as st

# One global config for the whole app (run once, at the top)
st.set_page_config(
    page_title="Citizen Complaint System",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ Citizen Complaint System")
st.write("왼쪽 사이드바에서 페이지를 선택하세요.")

# Optional: small sidebar header
with st.sidebar:
    st.markdown("### streamlit app")

# Optional: a tiny help section on the landing page
st.markdown(
    """
    **Pages**
    - **1_chatbot_intake**: 챗봇이 이름/연락처/주소/민원 내용을 단계적으로 수집 (음성 입력 지원)
    - **2_officer_dashboard**: 담당자용 대시보드 (필터/다운로드)
    """
)

# Optional: hide Streamlit default footer/menu (uncomment if you want)
# st.markdown(
#     """
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

