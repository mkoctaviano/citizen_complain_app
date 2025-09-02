#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pages/99_debug_speech.py
import streamlit as st
from utils.voice import _speech_client
st.write(type(_speech_client()).__name__)

