#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# inference_entry.py
# Thin adapter so the UI always calls one stable function:
#   run_text_pipeline(text) -> dict (your model's full output)

from typing import Dict, Any, Callable
import importlib

# Optional caching when running under Streamlit
try:
    import streamlit as st
except Exception:
    st = None

def _load_runner() -> Callable[[str], Dict[str, Any]]:
    """
    Import the project's run_full_inference from:
      citizen_complain_app/inference_wrapper.py
    """
    mod = importlib.import_module("citizen_complain_app.inference_wrapper")
    # This is the function you showed in the screenshot
    return getattr(mod, "run_full_inference")

if st:
    @st.cache_resource
    def _get_runner():
        return _load_runner()
else:
    _runner_cache = None
    def _get_runner():
        nonlocal _runner_cache  # type: ignore
        if _runner_cache is None:
            _runner_cache = _load_runner()
        return _runner_cache

def run_text_pipeline(text: str, k_sim: int = 5) -> Dict[str, Any]:
    """
    Public entrypoint for the UI. Feeds text into your core model.
    Returns the exact dict produced by run_full_inference.
    """
    runner = _get_runner()
    return runner(text=text, k_sim=k_sim)

