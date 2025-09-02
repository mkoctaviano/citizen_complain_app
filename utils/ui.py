#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st

def hide_multipage_nav_css():
    """Hide Streamlit's default sidebar page navigation via CSS."""
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {display: none;}
        [data-testid="stSidebar"] {display: none;}
        div[data-testid="collapsedControl"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

