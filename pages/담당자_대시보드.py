#!/usr/bin/env python
# coding: utf-8

# pages/2_담당자_대시보드.py
import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from utils.ui import hide_multipage_nav_css

# ---------- page config FIRST ----------
st.set_page_config(page_title="담당자 대시보드", page_icon="", layout="wide")

# ===== 모던(Alpine) 테마 스타일 =====
st.markdown("""
<style>
/* Alpine 기반 라이트 톤 + 라운드 + 그림자 */
.ag-theme-alpine {
  --ag-foreground-color: #111827;
  --ag-background-color: #ffffff;
  --ag-header-background-color: #f8fafc;
  --ag-border-color: #e5e7eb;
  --ag-row-hover-color: #f1f5f9;
  --ag-selected-row-background-color: #eff6ff;
  --ag-font-size: 13px;
  --ag-cell-horizontal-padding: 10px;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(17,24,39,.06);
}

/* 헤더 타이포 + 가운데 정렬 + sticky */
.ag-theme-alpine .ag-header-cell-label {
  font-weight: 700;
  justify-content: center;
}
.ag-theme-alpine .ag-header {
  position: sticky; top: 0; z-index: 2;
}

/* ✅ 헤더 텍스트 색/크기 (여기 추가) */
.ag-theme-alpine .ag-header-cell-text {
  color: #0B2F59 !important;   /* 파란색 */
  font-size: 16px !important;  /* 크게 */
  font-weight: 800 !important; /* 굵게 */
  text-align: center !important; /* 가운데 정렬 */
  
/* zebra, hover, selected */
.ag-theme-alpine .ag-row-odd { background: #fbfdff; }
.ag-theme-alpine .ag-row-hover .ag-cell { background:#f5f8fd !important; }
.ag-theme-alpine .ag-row.ag-row-selected .ag-cell { background:#eef6ff !important; }

/* 셀 경계선 은은하게 */
.ag-theme-alpine .ag-cell { border-right: 1px solid #eef1f5; }

/* 긴 텍스트 2줄까지 표시(필요시 1로 바꾸면 됨) */
.ag-theme-alpine .ag-cell.ag-cell-wrap-text{
  display:-webkit-box; -webkit-box-orient:vertical;
  -webkit-line-clamp:2; overflow:hidden;
}

/* 마우스 포인터 */
.ag-theme-alpine .ag-row { cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# Hide the default multipage sidebar navigation
hide_multipage_nav_css()
st.markdown("""
<style>
/* "민원 검색" 라벨 크게 */
label[data-testid="stWidgetLabel"] p {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #0B2F59 !important;
}

/* 검색 입력창 폭 절반으로 제한 */
div[data-testid="stTextInput"] {
    max-width: 50% !important;
}
</style>
""", unsafe_allow_html=True)
# Top nav: back to Home
if st.button("🏠 홈으로"):
    st.switch_page("streamlit_app.py")  # filename of the page script

# ---------- officer guard ----------
def _require_officer():
    if st.session_state.get("role") == "officer":
        return
    st.warning("담당자 전용 화면입니다. 홈에서 비밀번호로 입장하세요.")
    try:
        st.page_link("Home.py", label="← 홈으로")
    except Exception:
        pass
    st.stop()

_require_officer()

# ---------- load .env BEFORE storage ----------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

from storage import init_db, 대시보드_조회  # noqa: E402

# ---------- helpers ----------
def convert_timestamp(ts):
    if isinstance(ts, pd.Timestamp):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts

def _switch_or_link(page_path: str):
    try:
        st.switch_page(page_path)  # Streamlit >= 1.25
    except Exception:
        st.page_link(page_path, label="이동하기 →")
        st.stop()

def _to_obj(x):
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}

def _find_numeric_by_keys(obj: dict, keys_lower: set):
    """Depth-first search for first numeric value whose key matches keys_lower."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(k, str) and k.lower() in keys_lower and isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return None

def extract_urgency_score(extra):
    """
    Try to pull a numeric urgency score from '기타' JSON.
    Looks for keys like: urgency_norm, urgency_score, urgency, score.
    """
    obj = _to_obj(extra)
    KEYS = {"urgency_norm", "urgency_score", "urgency", "urgency_raw", "score"}
    return _find_numeric_by_keys(obj, keys_lower=KEYS)

# ----- export-specific helpers (human-readable CSV) -----
def _best_span_from_cause(cause: dict) -> dict | None:
    """Pick the highest-score span from cause.extra.spans if available."""
    if not cause:
        return None
    spans = (cause.get("extra") or {}).get("spans") or []
    if not isinstance(spans, list) or not spans:
        return None
    try:
        return max(spans, key=lambda s: float((s or {}).get("score", 0.0)))
    except Exception:
        return spans[0]

def _extract_cause_sentence_and_span(extra_raw) -> tuple[str, str]:
    """
    From the '기타' JSON, extract:
      - sentence: human-readable cause sentence if present
      - cause_span: a short span or phrase (best-scoring), or cause.cause_span field
    """
    extra = _to_obj(extra_raw)
    cause = extra.get("cause") or {}
    sentence = (cause.get("sentence") or "").strip()

    # prefer explicit cause_span; else fall back to best span's 'text'
    cause_span = (cause.get("cause_span") or "").strip()
    if not cause_span:
        best = _best_span_from_cause(cause)
        if isinstance(best, dict):
            cause_span = (best.get("text") or "").strip()
    return sentence, cause_span

def _excel_text(x) -> str:
    """
    Ensure Excel treats text-like values as text—especially phone numbers.
    Numeric types are prefixed with a quote to avoid scientific notation.
    """
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        return f"'{x}"
    return str(x)

def build_human_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a readable export DataFrame (like the detail page), with:
      민원번호, 이름, 연락처, 주소, 내용, 부서, 세부분야, 긴급도, 감정, 상태,
      접수일시, 처리일시, 접수경로, 원인 문장, 핵심 구간
    """
    out = pd.DataFrame()
    out["민원번호"] = df.get("민원번호", "")
    out["이름"]    = df.get("이름", "").apply(_excel_text)
    out["연락처"]  = df.get("연락처", "").apply(_excel_text)
    out["주소"]    = df.get("주소", "").apply(_excel_text)
    out["내용"]    = df.get("내용", "").apply(lambda s: "" if s is None else str(s))

    out["상위부서"]      = df.get("부서", "")
    out["하위부서"]  = df.get("세부분야", "")
    out["긴급도"]    = df.get("긴급도", "")
    out["감정"]      = df.get("감정", "")
    out["상태"]      = df.get("상태", "")

    out["접수일시"] = df.get("접수일시", "").apply(
        lambda ts: ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, pd.Timestamp) else ("" if ts is None else str(ts))
    )
    out["처리일시"] = df.get("처리일시", "").apply(
        lambda ts: ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, pd.Timestamp) else ("" if ts is None else str(ts))
    )
    out["접수경로"] = df.get("접수경로", "")

    cause_sentences, cause_spans = [], []
    for extra_raw in df.get("기타", []):
        sent, span = _extract_cause_sentence_and_span(extra_raw)
        cause_sentences.append(sent)
        cause_spans.append(span)
    out["원인 문장"] = cause_sentences
    out["핵심 구간"] = cause_spans

    cols = ["민원번호","이름","연락처","주소","내용","부서","세부분야","긴급도","감정","상태",
            "접수일시","처리일시","접수경로","원인 문장","핵심 구간"]
    out = out[[c for c in cols if c in out.columns]]
    return out

# ---------- header ----------
st.title("담당자 대시보드")
st.caption("아래 표에서 **행을 더블클릭**하면 상세 페이지로 이동합니다.")

# ---------- DB init ----------
try:
    init_db()
except RuntimeError as e:
    st.error(
        "데이터베이스 연결에 실패했습니다. `.env` 파일에 `DATABASE_URL`을 설정하고 "
        "UTF-8 인코딩으로 저장했는지 확인하세요.\n\n"
        f"에러: {e}"
    )
    st.stop()

# ---------- load data ----------
df = 대시보드_조회(접수경로_제외=["old_import"], 모델버전_제외=["bulk_v1"])
if df.empty:
    st.info("표시할 민원이 없습니다.")
    st.stop()

# view columns (ADD '부서')
main_cols = ["민원번호", "이름", "부서", "접수일시", "내용"]
main_df = (
    df[main_cols]
    .copy()
    .sort_values("민원번호", ascending=False)
    .reset_index(drop=True)
)
main_df["접수일시"] = main_df["접수일시"].apply(convert_timestamp)

# ---------- grid (modern options) ----------
gb = GridOptionsBuilder.from_dataframe(main_df)

# 기본 컬럼: 리사이즈/정렬/필터 + 플로팅 필터
gb.configure_default_column(resizable=True, sortable=True, filter=True, floatingFilter=True)

# 수동 너비 + 정렬/스타일 + 고정열
center_style = {"justifyContent":"center", "display":"flex"}
gb.configure_column("민원번호", width=100, pinned="left", cellStyle={"fontWeight":"700"})
gb.configure_column("이름", width=120, cellStyle=center_style)
gb.configure_column("부서", width=150, cellStyle=center_style)
gb.configure_column("접수일시", width=160, cellStyle=center_style)

# 내용: flex-fill + 줄바꿈 + 자동 높이 + 툴팁
gb.configure_column(
    "내용",
    flex=1, minWidth=360,
    wrapText=True,
    tooltipField="내용"
)

# 페이지네이션 자동 + 행 애니메이션 + 더블클릭 선택
#gb.configure_pagination(paginationAutoPageSize=True)
gb.configure_selection(selection_mode="single", use_checkbox=False)
# grid 옵션 설정 부분
gb.configure_grid_options(
    rowHeight=36,
    suppressRowClickSelection=True,
    suppressClickEdit=True,
    onRowDoubleClicked=JsCode("""
        function(e){
            e.api.deselectAll();
            e.node.setSelected(true);
        }
    """),
)
gb.configure_grid_options(domLayout='normal')   # ← 추가: 내부 스크롤 사용

# 상단 빠른 검색
q = st.text_input("민원 검색", "", placeholder="이름, 내용, 부서…")

grid_options = gb.build()
grid_options["quickFilterText"] = q

# AgGrid 호출 부분 (height=500 유지)
grid_resp = AgGrid(
    main_df,
    gridOptions=grid_options,
    height=500,                       # ← 그대로 500 유지
    theme="alpine",                   # alpine/balham 중 쓰시는 테마 유지
    allow_unsafe_jscode=True,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    fit_columns_on_grid_load=False,
    custom_css={                      # ← 내부 스크롤 강제
        ".ag-root-wrapper": {"height": "100%"},
        ".ag-body-viewport": {"overflow-y": "auto"}
    }
)




# ---------- selection -> detail ----------
selected_rows = grid_resp.get("selected_rows", None)
selected_id = None
if isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
    selected_id = int(selected_rows.iloc[0]["민원번호"])
elif isinstance(selected_rows, list) and selected_rows:
    selected_id = int(selected_rows[0]["민원번호"])

if selected_id is not None:
    st.session_state["detail_id"] = selected_id
    try:
        st.query_params = {"complaint_id": str(selected_id)}
    except Exception:
        pass
    _switch_or_link("pages/민원상세.py")

# ---------- analytics ----------
st.markdown("---")
a1, a2 = st.columns([1, 1])

with a1:
    st.subheader("우선민원 TOP5")
    tmp = df.copy()
    tmp["긴급도점수"] = tmp["기타"].apply(extract_urgency_score)
    top5 = (
        tmp.dropna(subset=["긴급도점수"])
           .sort_values("긴급도점수", ascending=False)
           .head(5)[["민원번호", "부서", "이름", "내용"]]
           .copy()
    )
    if not top5.empty:
        # 순위 1..5 + 내용 축약
        top5.insert(0, "순위", range(1, len(top5) + 1))
        top5["내용"] = (
            top5["내용"].astype(str).str.slice(0, 80) +
            top5["내용"].astype(str).str.slice(80).map(lambda x: "…" if x else "")
        )

        # ---- Clickable mini grid (double-click a row to open 민원상세) ----
        gb2 = GridOptionsBuilder.from_dataframe(top5)
        gb2.configure_default_column(resizable=True, sortable=False, filter=False)
        gb2.configure_selection(selection_mode="single", use_checkbox=False)
        gb2.configure_column("순위", width=70)
        gb2.configure_column("민원번호", width=100)
        gb2.configure_column("부서", width=140)
        gb2.configure_column("이름", width=120)
        gb2.configure_column("내용", flex=1, minWidth=300)
        gb2.configure_grid_options(
            rowHeight=36,
            suppressRowClickSelection=True,
            suppressClickEdit=True,
            onRowDoubleClicked=JsCode("""
                function(e){
                    e.api.deselectAll();
                    e.node.setSelected(true);
                }
            """),
        )
        grid_top = AgGrid(
            top5,
            gridOptions=gb2.build(),
            height=260,
            theme="alpine",          # 톤 맞추기
            allow_unsafe_jscode=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=False,
        )

        # Navigate when a Top 5 row is selected
        sel = grid_top.get("selected_rows", None)
        selected_id_top5 = None
        if isinstance(sel, pd.DataFrame) and not sel.empty:
            selected_id_top5 = int(sel.iloc[0]["민원번호"])
        elif isinstance(sel, list) and sel:
            selected_id_top5 = int(sel[0]["민원번호"])

        if selected_id_top5 is not None:
            st.session_state["detail_id"] = selected_id_top5
            try:
                st.query_params = {"complaint_id": str(selected_id_top5)}
            except Exception:
                pass
            _switch_or_link("pages/민원상세.py")

        st.caption("행을 **더블클릭**하면 상세 페이지로 이동합니다.")
    else:
        st.info("긴급도 점수(urgency) 정보를 찾을 수 없습니다.")

import altair as alt
import pandas as pd

with a2:
    st.subheader("부서별 민원 건수")

    # 집계
    dept_counts = (
        df["부서"].fillna("미지정").replace("", "미지정").value_counts().reset_index()
    )
    dept_counts.columns = ["부서", "건수"]
    total = float(dept_counts["건수"].sum())
    dept_counts["비율"] = dept_counts["건수"] / total

    # ✅ 범례용 라벨: "부서 (xx.x%)"
    dept_counts["부서라벨"] = dept_counts.apply(
        lambda r: f"{r['부서']} ({r['비율']:.1%})", axis=1
    )

    if not dept_counts.empty:
        # 두 차트가 공유할 색상 스케일 (도메인 고정)
        color_domain = dept_counts["부서라벨"].tolist()
        color_scale  = alt.Scale(domain=color_domain, scheme="category20")

        # ── 가로 막대 ── (부서 많아지면 자동으로 높이 증가)
        bar_height = max(220, 28 * len(dept_counts))
        bar_chart = (
            alt.Chart(dept_counts)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(
                x=alt.X("건수:Q", axis=alt.Axis(title="민원 건수", labelFontSize=12)),
                y=alt.Y("부서:N", sort="-x", axis=alt.Axis(title=None, labelFontSize=12)),
                color=alt.Color("부서라벨:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("부서:N", title="부서"),
                    alt.Tooltip("건수:Q", title="건수"),
                    alt.Tooltip("비율:Q", format=".1%", title="비율"),
                ],
            )
            .properties(width=420, height=bar_height, title="부서별 민원 건수")
        )

        bar_text = bar_chart.mark_text(
            align="left", baseline="middle", dx=3, fontSize=11
        ).encode(text="건수:Q")

        # ── 도넛 ── (범례에 % 표시)
        pie_chart = (
            alt.Chart(dept_counts)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta("건수:Q"),
                color=alt.Color(
                    "부서라벨:N",
                    scale=color_scale,
                    legend=alt.Legend(
                        title="부서 (비율)",
                        orient="bottom",
                        direction="vertical",
                        labelFontSize=12,
                        titleFontSize=12,
                        symbolSize=200,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("부서:N", title="부서"),
                    alt.Tooltip("건수:Q", title="건수"),
                    alt.Tooltip("비율:Q", format=".1%", title="비율"),
                ],
            )
            .properties(width=360, height=360, title="부서별 민원 비율")
        )

        combo = ((bar_chart + bar_text) | pie_chart).resolve_scale(color="independent")
        st.altair_chart(combo, use_container_width=False)
    else:
        st.info("부서 데이터가 없습니다.")

# ---------- csv download (human-friendly) ----------
st.markdown("---")
export_df = build_human_export_df(df)
csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")  # BOM for Excel compatibility
st.download_button(
    "CSV 다운로드",
    data=csv_bytes,
    file_name="complaints_readable.csv",
    mime="text/csv",
)
