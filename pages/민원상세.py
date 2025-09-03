#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pages/3_민원상세.py
import json
from typing import Any, Dict, List, Optional, Tuple

from utils.gcs import signed_url

import pandas as pd
import streamlit as st
from storage import init_db, 대시보드_조회

#  hide Streamlit's nav so this feels like an independent page
try:
    from utils.ui import hide_multipage_nav_css  # your CSS helper
except Exception:
    def hide_multipage_nav_css():  # no-op fallback
        pass

# ---- MUST be first Streamlit call on this page ----
st.set_page_config(page_title="민원 상세", page_icon="", layout="wide")
hide_multipage_nav_css()

# ---------- CSS (폰트 크게 + 굵게 + 가운데 정렬) ----------
st.markdown("""
    <style>
    .info-line {
        font-size: 30px !important;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
        color: #0B2F59;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Officer guard (no set_page_config here!) ----------
def _require_officer():
    if st.session_state.get("role") == "officer":
        return
    st.warning("담당자 전용 화면입니다. 홈에서 비밀번호로 입장하세요.")
    try:
        st.page_link("streamlit_app.py", label="← 홈으로")
    except Exception:
        pass
    st.stop()

_require_officer()

# ---------- helpers ----------
def convert_timestamp(ts):
    if isinstance(ts, pd.Timestamp):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts

def _to_obj(maybe_json: Any) -> Dict:
    if maybe_json is None:
        return {}
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {}
    return {}

def _parse_기타(extra_raw: Any) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Returns (cause_dict, normalized_similar_list).
    Similar items -> list of dicts each with a 'text' only (meta-aware).
    """
    extra = _to_obj(extra_raw)
    cause = extra.get("cause") or None

    raw_sim = (
        extra.get("similarity")
        or extra.get("similar")
        or extra.get("neighbors")
        or []
    )

    norm: List[Dict[str, str]] = []
    for item in raw_sim:
        if isinstance(item, dict):
            t = (
                item.get("text")
                or _to_obj(item.get("meta")).get("text")
                or _to_obj(item.get("meta")).get("내용")
                or _to_obj(item.get("meta")).get("본문")
                or ""
            )
            norm.append({"text": str(t).strip()})
        elif isinstance(item, (list, tuple)) and item:
            norm.append({"text": str(item[0]).strip()})
        else:
            norm.append({"text": str(item).strip()})
    return cause, norm

def _best_span(cause: Optional[Dict]) -> Optional[Dict]:
    if not cause:
        return None
    spans = (cause.get("extra") or {}).get("spans") or []
    if not isinstance(spans, list) or not spans:
        return None
    try:
        return max(spans, key=lambda s: float((s or {}).get("score", 0.0)))
    except Exception:
        return spans[0]

def _clean_similar_text(raw_text: Any) -> str:
    """Return only the human text (handles meta JSON)."""
    if raw_text is None:
        return ""
    if isinstance(raw_text, dict):
        return (
            str(
                raw_text.get("text")
                or _to_obj(raw_text.get("meta")).get("text")
                or _to_obj(raw_text.get("meta")).get("내용")
                or _to_obj(raw_text.get("meta")).get("본문")
                or ""
            ).strip()
        )
    s = str(raw_text).strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return (
                    str(
                        obj.get("text")
                        or _to_obj(obj.get("meta")).get("text")
                        or _to_obj(obj.get("meta")).get("내용")
                        or _to_obj(obj.get("meta")).get("본문")
                        or ""
                    ).strip()
                )
        except Exception:
            pass
    return s

def render_cause_block(cause: Optional[Dict]):
    st.markdown("## 원인 추출")
    if not cause:
        st.info("원인 추출 결과가 없습니다.")
        return

    best = _best_span(cause)
    sentence   = (cause.get("sentence") or (best or {}).get("sentence") or "").strip()
    cause_span = (cause.get("cause_span") or (best or {}).get("text") or "").strip()

    if not sentence and not cause_span:
        st.info("원인 추출 결과가 없습니다.")
        return

    # Only show 원인 문장 in UI
    st.markdown(f"**원인 문장:** {sentence}" if sentence else "원인 문장 없음")

    # ⬇️ 핵심 구간 intentionally hidden in UI
    # (still exists in cause_span variable for CSV or backend use)
    # with st.expander(" 핵심 구간 (internal)"):
    #     st.code(cause_span, language="text")


def render_similar_block(similar: List[Dict], limit: int = 10):
    st.markdown("##  유사 민원 TOP 5")
    if not similar:
        st.info("유사 민원 검색 결과가 없습니다.")
        return

    seen = set()
    cleaned: List[str] = []
    for it in similar:
        txt = _clean_similar_text(it)
        if not txt or txt in seen:
            continue
        seen.add(txt)
        cleaned.append(txt)
        if len(cleaned) >= limit:
            break

    if not cleaned:
        st.info("유사 민원 검색 결과가 없습니다.")
        return

    for i, t in enumerate(cleaned, 1):
        st.markdown(f"{i}. {t}")

# ---- DB init (with helpful error) ----
try:
    init_db()
except RuntimeError as e:
    st.error(
        "데이터베이스 연결에 실패했습니다. `.env` 파일의 `DATABASE_URL` 설정을 확인하세요.\n\n"
        f"에러: {e}"
    )
    st.stop()

# Back link to dashboard
import streamlit as st

# Only works inside multipage apps
if st.button("⬅️ 담당자 대시보드"):
    st.switch_page("pages/담당자_대시보드.py")  # filename of the page script
    

# Get complaint id (session or URL)
qid = st.session_state.get("detail_id")
if qid is None:
    try:
        qp = st.query_params
        if "complaint_id" in qp:
            qid = int(qp.get("complaint_id"))
            st.session_state["detail_id"] = qid
    except Exception:
        pass

if qid is None:
    st.error("선택된 민원이 없습니다. 대시보드에서 항목을 더블클릭하세요.")
    st.stop()

# Same filters as dashboard
df = 대시보드_조회(접수경로_제외=["old_import"], 모델버전_제외=["bulk_v1"])
if df.empty or qid not in df["민원번호"].values:
    st.error("해당 민원을 찾을 수 없습니다.")
    st.stop()

row = df[df["민원번호"] == qid].iloc[0]

# Title
st.title(f"민원 #{row['민원번호']}")

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    st.markdown(f'<div class="info-line">이름 : {row.get("이름", "")}</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="info-line">연락처 : {row.get("연락처", "")}</div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="info-line">주소 : {row.get("주소", "")}</div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="info-line">접수일시 : {convert_timestamp(row.get("접수일시", ""))}</div>', unsafe_allow_html=True)

# ---------- Classification ----------
st.markdown("## 분류 결과")
r1, r2, r3, r4 = st.columns([1, 1, 1, 1])
with r1:
    st.markdown(f'<div class="info-line">부서 : {row.get("부서", "")}</div>', unsafe_allow_html=True)
with r2:
    st.markdown(f'<div class="info-line">세부분야 : {row.get("세부분야", "")}</div>', unsafe_allow_html=True)
with r3:
    st.markdown(f'<div class="info-line">긴급도 : {row.get("긴급도", "")}</div>', unsafe_allow_html=True)
with r4:
    st.markdown(f'<div class="info-line">감정 : {row.get("감정", "")}</div>', unsafe_allow_html=True)

# Extras (cause + similar)
extra = row.get("기타") or row.get("extra")
cause, similar = _parse_기타(extra)

st.divider()
render_cause_block(cause)
st.divider()
render_similar_block(similar)


voice = (extra or {}).get("voice") or {}
if voice.get("gs_uri"):
    st.markdown("### 음성 첨부")
    try:
        url = signed_url(voice["gs_uri"], expires_sec=600)
        st.audio(url, format="audio/wav")
    except Exception as e:
        st.caption(f"오디오 링크 생성 실패: {e}")

