# app.py
# Streamlit app that runs:
# - KEI booster + Parent/Child routing
# - Cause extraction (cause_bert_tagger via pipeline)
# - Retriever-BERT "similar" search
# - Priority score = 0.8 * urgency_norm + 0.2 * emotion_norm

import os, json, re, unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
# STT helper (builds Google client from Streamlit Secrets)
from citizen_complain_app.stt_google import transcribe_bytes


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
)

from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process as fuzz_process
import pickle

# =========================
# Paths
# =========================
BASE_DIR    = Path(__file__).resolve().parent
KEI_PKL     = BASE_DIR / "kei_booster.pkl"
PARENT_DIR  = BASE_DIR / "main_model"
CHILD_DIR   = BASE_DIR / "child"
CHILD_REG   = BASE_DIR / "child_registry.json"
CSV_PATH    = BASE_DIR / "_merged_unidept_88_6cols_with_parent_fix_clean.csv"

CAUSE_DIR   = BASE_DIR / "cause_tagger"
RETR_DIR    = BASE_DIR / "retriever_bert"

PRIORITY_DIR = BASE_DIR / "priority_model"
URGENCY_DIR  = PRIORITY_DIR / "urgency_model_roberta_reg"   # <- rename if yours is *_reg
EMOTION_DIR  = PRIORITY_DIR / "KoElectra_emotion"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256

# =========================
# KEIBooster
# =========================
def _normalize_ko(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").strip()
    return re.sub(r"\s+", " ", s)

@st.cache_resource(show_spinner=False)
def load_kei_booster(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        blob = pickle.load(f)
    model = SentenceTransformer(blob["cfg"]["sbert_model_name"])
    return {
        "keyword_list":    blob["keyword_list"],
        "intent_list":     blob["intent_list"],
        "intent2examples": blob["intent2examples"],
        "kw_emb":          blob["kw_emb"],
        "intent_proto":    blob["intent_proto"],
        "cfg":             blob["cfg"],
        "sbert":           model,
    }

def kei_extract_keywords(booster, text: str, top_k: int = 5):
    t = _normalize_ko(text)
    kws = booster["keyword_list"]; kw_emb = booster["kw_emb"]
    if not kws or kw_emb is None:
        return []
    v = booster["sbert"].encode([t], convert_to_numpy=True, normalize_embeddings=True)[0]
    cos = kw_emb @ v
    fuzzy = np.array([fuzz.partial_ratio(t, kw)/100.0 for kw in kws])
    substr = np.array([0.1 if kw in t else 0.0 for kw in kws])
    score = 0.7 * cos + 0.3 * fuzzy + substr
    top = np.argsort(-score)[:top_k]
    return [kws[i] for i in top]

def kei_extract_intent(booster, text: str):
    t = _normalize_ko(text)
    proto = booster["intent_proto"]; intents = booster["intent_list"]
    if proto is None or not intents:
        return "미정"
    v = booster["sbert"].encode([t], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = proto @ v
    i = int(np.argmax(sims)); sc = float(sims[i])
    return intents[i] if sc >= 0.35 else "미정"

def kei_compose_input(text: str, keywords=None, intent=None):
    keywords = keywords or []
    kw_str = ";".join(dict.fromkeys(keywords)) if keywords else ""
    it_str = intent or "미정"
    return f"[키워드:{kw_str}][의도:{it_str}] {text}"

# =========================
# Helpers
# =========================
def _is_valid_model_dir(path: Path) -> bool:
    if not path.is_dir(): return False
    has_cfg = (path / "config.json").exists()
    has_wts = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    return has_cfg and has_wts

def _norm_label(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"[^0-9a-z가-힣]", "", s)
    return s

# =========================
# Parent / Child Router
# =========================
@st.cache_resource(show_spinner=False)
def load_parent_model(parent_dir: Path):
    assert (parent_dir / "label_encoder.json").exists(), "parent/label_encoder.json 없음"
    with open(parent_dir / "label_encoder.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    tok = AutoTokenizer.from_pretrained(parent_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(parent_dir, local_files_only=True).to(DEVICE).eval()
    return tok, mdl, classes

def build_child_map(child_dir: Optional[Path], child_registry: Optional[Path], parent_classes: List[str]) -> Dict[str, Path]:
    child_dirs: Dict[str, Path] = {}
    if child_dir and child_dir.exists() and child_dir.is_dir():
        for sub in sorted(child_dir.iterdir()):
            if sub.is_dir() and _is_valid_model_dir(sub):
                child_dirs[sub.name] = sub
    if child_registry and child_registry.exists():
        try:
            reg = json.load(open(child_registry, "r", encoding="utf-8"))
            for pl, info in reg.items():
                if isinstance(info, str):
                    p = (child_registry.parent / info).resolve()
                elif isinstance(info, dict):
                    mp = info.get("path") or info.get("model_path") or info.get("dir") or info.get("folder")
                    if not mp: continue
                    p = (child_registry.parent / mp).resolve()
                else:
                    continue
                if _is_valid_model_dir(p):
                    child_dirs[pl] = p
        except Exception as e:
            st.warning(f"child_registry.json parse error: {e}")

    # Fuzzy map
    candidates = [(name, _norm_label(name), path) for name, path in child_dirs.items()]
    mapping: Dict[str, Path] = {}
    for pl in parent_classes:
        pln = _norm_label(pl)
        exact = [p for (orig, norm, p) in candidates if norm == pln]
        if exact:
            mapping[pl] = exact[0]; continue
        if candidates:
            cand_names = [orig for (orig, norm, p) in candidates]
            m = fuzz_process.extractOne(pl, cand_names, scorer=fuzz.token_sort_ratio)
            if m and m[1] >= 85:
                matched = m[0]
                for (orig, norm, p) in candidates:
                    if orig == matched:
                        mapping[pl] = p; break
    return mapping

@torch.no_grad()
def predict_parent(tok, mdl, classes, text: str) -> Tuple[str, float, float]:
    X = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
    logits = mdl(**X).logits[0]
    prob = torch.softmax(logits, dim=-1).cpu().numpy()
    order = np.argsort(-prob)
    i1, i2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    parent1 = classes[i1] if i1 < len(classes) else f"parent_{i1}"
    return parent1, float(prob[i1]), float(prob[i1] - prob[i2])

@st.cache_resource(show_spinner=False)
def load_child_model(path: Path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(DEVICE).eval()
    # label map
    le_path = path / "label_encoder.json"
    labels_map = None
    if le_path.exists():
        with open(le_path, "r", encoding="utf-8") as f:
            classes = json.load(f)["classes"]
        labels_map = {i: classes[i] for i in range(len(classes))}
    elif getattr(mdl.config, "id2label", None):
        labels_map = {int(k): v for k, v in mdl.config.id2label.items()}
    else:
        labels_map = {i: f"class_{i}" for i in range(int(mdl.config.num_labels))}
    return tok, mdl, labels_map

@torch.no_grad()
def predict_child(path: Path, text: str) -> Tuple[str, float, float, Optional[str]]:
    tok, mdl, labels_map = load_child_model(path)
    num_labels = int(mdl.config.num_labels)
    if num_labels == 1:
        return labels_map[0], 1.0, 1.0, "single_class"
    X = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
    logits = mdl(**X).logits[0]
    prob = torch.softmax(logits, dim=-1).cpu().numpy()
    order = np.argsort(-prob)
    i1, i2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    label1 = labels_map.get(i1, f"class_{i1}")
    return label1, float(prob[i1]), float(prob[i1] - prob[i2]), None

# =========================
# Cause (pipeline)
# =========================
def normalize_cause_label(label: str) -> str:
    if not label: return label
    label = re.sub(r"(과|부|팀)$", "", label)
    return label.strip()

def _choose_subject_particle(word: str) -> str:
    if not word: return "이"
    ch = word[-1]; code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        jong = (code - 0xAC00) % 28
        return "이" if jong != 0 else "가"
    return "이"

def format_cause_sentence(info: dict) -> str:
    span = (info.get("cause_span") or "").strip()
    if not span:
        return "해당 민원에서 명확한 원인을 추출하지 못했습니다."
    particle = _choose_subject_particle(span)
    return f"{span}{particle} 원인으로 확인됩니다."

@st.cache_resource(show_spinner=False)
def load_cause_pipeline(path: Path):
    if not _is_valid_model_dir(path):
        return None
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mdl = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)
    return pipeline(
        "token-classification",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy="simple",
        device=0 if DEVICE == "cuda" else -1
    )

def run_cause(cause_pl, text: str) -> dict:
    if cause_pl is None:
        return {"cause_label": "N/A", "cause_span": "", "cause_score": 0.0}
    spans = cause_pl(text)
    if not spans:
        return {"cause_label": "", "cause_span": "", "cause_score": 0.0}
    best = max(spans, key=lambda r: r.get("score", 0.0))
    span = (best.get("word") or "").strip()
    return {
        "cause_label": normalize_cause_label(span),
        "cause_span": span,
        "cause_score": float(best.get("score", 0.0)),
    }

# =========================
# Retriever-BERT (similar)
# =========================
@st.cache_resource(show_spinner=False)
def load_retriever(model_dir: Path):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_dir, local_files_only=True).to(DEVICE).eval()
    return tok, mdl

@st.cache_data(show_spinner=False)
def load_retriever_index(index_dir: Path):
    E = np.load(index_dir / "embeddings_retriever.npy").astype(np.float32)
    with open(index_dir / "meta.jsonl", "r", encoding="utf-8") as f:
        meta = [line.strip() for line in f]
    E /= np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-12)
    return E, meta

@torch.no_grad()
def retriever_embed(tok, mdl, text: str):
    x = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
    v = mdl(**x).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
    return v  # (1, 768)

def retriever_search(tok, mdl, E, meta, q: str, k=5):
    v = retriever_embed(tok, mdl, q)
    sims = (E @ v.T).ravel()
    idx = sims.argsort()[::-1][:k]
    return [(meta[i], float(sims[i])) for i in idx]

# =========================
# Priority (Urgency + Emotion)
# =========================
ymin, ymax = 0.013, 2.181  # scaling used in training

def unscale_y(y):
    return np.asarray(y, np.float32) * (ymax - ymin) + ymin

def normalize_from_range(y):
    y = np.asarray(y, np.float32)
    return (y - ymin) / (ymax - ymin + 1e-12)

DEFAULT_EMOTION_WEIGHTS = {
    "angry": 1.00, "fear": 0.90, "surprise": 0.80, "disgust": 0.70,
    "happy": 0.60, "sad": 0.50, "neutral": 0.30,
}

@st.cache_resource(show_spinner=False)
def load_priority_models(urg_dir: Path, emo_dir: Path):
    tok_urg = AutoTokenizer.from_pretrained(urg_dir, local_files_only=True)
    mdl_urg = AutoModelForSequenceClassification.from_pretrained(urg_dir, local_files_only=True).to(DEVICE).eval()
    tok_emo = AutoTokenizer.from_pretrained(emo_dir, local_files_only=True)
    mdl_emo = AutoModelForSequenceClassification.from_pretrained(emo_dir, local_files_only=True).to(DEVICE).eval()
    return tok_urg, mdl_urg, tok_emo, mdl_emo

def _get_emotion_weights(model) -> np.ndarray:
    id2label = getattr(model.config, "id2label", None)
    if id2label and isinstance(id2label, dict) and len(id2label) == getattr(model.config, "num_labels", 0):
        weights = []
        for i in range(model.config.num_labels):
            name = str(id2label.get(i, "")).lower()
            weights.append(DEFAULT_EMOTION_WEIGHTS.get(name, 0.50))
        return np.asarray(weights, np.float32)
    # fallback common order
    assumed = ["angry","disgust","fear","sad","surprise","happy","neutral"]
    return np.asarray([DEFAULT_EMOTION_WEIGHTS.get(x, 0.50) for x in assumed[:model.config.num_labels]], np.float32)

@torch.no_grad()
def predict_urgency(tok_urg, mdl_urg, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    enc = tok_urg(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    logits_scaled = mdl_urg(**enc).logits.squeeze(-1)  # regression scalar
    logits_scaled = logits_scaled.detach().cpu().numpy().astype(np.float32)
    raw = np.clip(unscale_y(logits_scaled), ymin, ymax)
    norm = np.clip(normalize_from_range(raw), 0.0, 1.0)
    return raw, norm

@torch.no_grad()
def predict_emotion_score(tok_emo, mdl_emo, texts: List[str]) -> np.ndarray:
    enc = tok_emo(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    out = mdl_emo(**enc).logits
    num_labels = int(getattr(mdl_emo.config, "num_labels", 1))
    if num_labels == 1:
        score = torch.sigmoid(out.squeeze(-1))
        return np.clip(score.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
    probs = F.softmax(out, dim=-1).detach().cpu().numpy().astype(np.float32)
    weights = _get_emotion_weights(mdl_emo)[:probs.shape[1]]
    weighted = (probs * weights).sum(axis=1) / (weights.max() + 1e-12)
    return np.clip(weighted, 0.0, 1.0)

def combine_priority(urg_norm: np.ndarray, emo_norm: np.ndarray, w_urg: float = 0.8, w_emo: float = 0.2) -> np.ndarray:
    combined = w_urg * np.asarray(urg_norm, np.float32) + w_emo * np.asarray(emo_norm, np.float32)
    return np.clip(combined, 0.0, 1.0)

def predict_priority_single(tok_urg, mdl_urg, tok_emo, mdl_emo, text: str, w_urg: float = 0.8, w_emo: float = 0.2) -> Dict[str, float]:
    urg_raw, urg_norm = predict_urgency(tok_urg, mdl_urg, [text])
    emo_norm = predict_emotion_score(tok_emo, mdl_emo, [text])
    priority = combine_priority(urg_norm, emo_norm, w_urg, w_emo)
    return {
        "urgency_raw": float(np.round(urg_raw[0], 3)),
        "urgency_norm": float(np.round(urg_norm[0], 3)),
        "emotion_norm": float(np.round(emo_norm[0], 3)),
        "priority": float(np.round(priority[0], 3)),
    }

# =========================
# Keyword vote table (for ambiguity)
# =========================
@st.cache_data(show_spinner=False)
def load_kw_votes(csv_path: Path):
    kw2parent = defaultdict(Counter)
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        for _, row in df.iterrows():
            for kw in re.split(r"[,\|;/]", str(row.get("키워드", ""))):
                kw = kw.strip()
                if kw:
                    kw2parent[kw][str(row.get("상위부서", "")).strip()] += 1
    return kw2parent

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Citizen Complaint Classifier", layout="centered")
st.title("🧠 Citizen Complaint Classifier")
st.caption("상위/하위부서 분류 + 원인 추출 + 유사 사례 + 우선순위(긴급/감정)")

# Load resources once
booster = load_kei_booster(KEI_PKL) if KEI_PKL.exists() else None
parent_tok, parent_mdl, parent_classes = load_parent_model(PARENT_DIR) if PARENT_DIR.exists() else (None, None, [])
child_map = build_child_map(CHILD_DIR if CHILD_DIR.exists() else None,
                            CHILD_REG if CHILD_REG.exists() else None,
                            parent_classes)
kw_votes = load_kw_votes(CSV_PATH)

cause_pl = load_cause_pipeline(CAUSE_DIR) if CAUSE_DIR.exists() else None

retr_tok, retr_mdl = load_retriever(RETR_DIR) if RETR_DIR.exists() else (None, None)
retr_E, retr_meta = load_retriever_index(RETR_DIR) if RETR_DIR.exists() else (None, None)

tok_urg, mdl_urg, tok_emo, mdl_emo = load_priority_models(URGENCY_DIR, EMOTION_DIR) if (URGENCY_DIR.exists() and EMOTION_DIR.exists()) else (None, None, None, None)

# -------------------------
# Voice input (optional)
# -------------------------
st.divider()
st.subheader("🎙️ 음성 입력 (선택)")

# Use built-in mic if available; otherwise fall back to uploader
audio_rec = getattr(st, "audio_input", None)
voice_file = audio_rec("Record (ko-KR)") if audio_rec else st.file_uploader(
    "오디오 업로드 (webm/mp3/m4a/wav/flac/ogg)",
    type=["webm","mp3","m4a","wav","flac","ogg"],
    key="voice_uploader",
)

if voice_file is not None:
    audio_bytes = voice_file.getvalue()
    st.audio(audio_bytes)

    c1, c2 = st.columns(2)
    do_fill   = c1.button("📝 음성 → 텍스트 (입력란에 채우기)", use_container_width=True)
    do_run    = c2.button("⚡ 음성 → 텍스트 → 즉시 분석", use_container_width=True)

    if do_fill or do_run:
        try:
            with st.spinner("음성 인식 중…"):
                transcript = transcribe_bytes(
                    audio_bytes,
                    language_code="ko-KR",
                    phrase_hints=STT_PHRASE_HINTS
                )
            if not transcript:
                st.warning("음성을 인식하지 못했습니다.")
            else:
                # Put transcript into the text box the app already uses
                st.session_state.input_text = transcript
                st.success("전사 완료: 아래 입력란에 텍스트를 채웠습니다.")
                if do_run:
                    # Trigger the same path as the text button
                    # by simulating a click: set a session flag and rerun.
                    st.session_state._run_clicked_from_voice = True
                    st.rerun()
        except Exception as e:
            st.error(f"STT 오류: {e}")

# Input
DEMO_TEXT = "지하철 역사에서 연기가 발생하여 승객 대피 필요합니다."
txt = st.text_area("민원 문장 입력", height=140, key="input_text", placeholder="한국어 민원 문장을 입력하세요…")
c1, c2 = st.columns([1,1])
run_clicked  = c1.button("🔍 분석 실행", use_container_width=True)
run_clicked = run_clicked or st.session_state.pop("_run_clicked_from_voice", False)
demo_clicked = c2.button("🧪 데모 예시 채우기", use_container_width=True)
if demo_clicked:
    st.session_state.input_text = DEMO_TEXT
    st.rerun()

if run_clicked and txt.strip():
    if booster is None or parent_tok is None:
        st.error("모델이 준비되지 않았습니다. 폴더/파일 경로를 확인하세요.")
        st.stop()

    # --- KEI compose
    intent = kei_extract_intent(booster, txt)
    if intent == "공통확인":  # never surface as intent
        intent = "미정"
    keywords = kei_extract_keywords(booster, txt, top_k=5)
    composed = kei_compose_input(txt, keywords[:3], intent)

    # --- Parent prediction + ambiguity checks
    parent1, p1, p_margin = predict_parent(parent_tok, parent_mdl, parent_classes, composed)
    reasons = []
    if p1 < 0.35: reasons.append("parent_prob<0.35")
    if p_margin < 0.10: reasons.append("parent_margin<0.10")

    # kw vote conflict check
    if keywords:
        votes = Counter()
        for kw in keywords:
            for dept, c in kw_votes.get(kw, {}).items():
                votes[dept] += c
        if votes:
            tot = sum(votes.values())
            top2 = votes.most_common(2)
            if len(top2) == 1 or (top2[0][1] - top2[1][1]) / max(tot, 1) < 0.15:
                reasons.append("kw vote conflict")

    # --- Child prediction
    if reasons:
        parent_label = "공통확인"; sub_label = "공통확인"
    else:
        parent_label = parent1
        path = child_map.get(parent1)
        if path is None:
            sub_label = parent1
        else:
            child_label, c1, c_margin, note = predict_child(path, composed)
            if note == "single_class":
                sub_label = child_label
            else:
                sub_label = child_label if (c1 >= 0.30 and c_margin >= 0.10) else "공통확인"

    # --- Cause
    cause_info = run_cause(cause_pl, txt)
    cause_sentence = format_cause_sentence(cause_info)

    # --- Priority (Urgency + Emotion)
    if tok_urg and mdl_urg and tok_emo and mdl_emo:
        pr = predict_priority_single(tok_urg, mdl_urg, tok_emo, mdl_emo, txt, w_urg=0.8, w_emo=0.2)
    else:
        pr = {"urgency_raw": None, "urgency_norm": None, "emotion_norm": None, "priority": None}

    # --- Retriever similar
    similar = []
    if retr_tok and retr_mdl and retr_E is not None and retr_meta is not None:
        similar = retriever_search(retr_tok, retr_mdl, retr_E, retr_meta, txt, k=5)

    # =========================
    # Render results
    # =========================
    st.subheader("결과")

    if parent_label == "공통확인":
        st.error("공통확인 fallback  | 사유: " + ", ".join(reasons))
    else:
        st.success(f"상위부서: {parent_label} ({p1:.2f})")
        st.info(f"하위부서: {sub_label}")

    st.warning(f"🧩 원인: {cause_sentence}")
    with st.expander("원인 추출 상세"):
        st.write(f"Span: `{cause_info.get('cause_span','')}`  | Score: {cause_info.get('cause_score',0.0):.2f}")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("우선순위 (0~1)", value="-" if pr['priority'] is None else f"{pr['priority']:.3f}")
    colB.metric("긴급도 정규화", value="-" if pr['urgency_norm'] is None else f"{pr['urgency_norm']:.3f}")
    colC.metric("감정 스코어", value="-" if pr['emotion_norm'] is None else f"{pr['emotion_norm']:.3f}")
    colD.metric("긴급도 원점수", value="-" if pr['urgency_raw'] is None else f"{pr['urgency_raw']:.3f}")

    with st.expander("🔎 KEI / Router 디버그"):
        st.write(f"키워드: {keywords}")
        st.write(f"의도: {intent}")
        st.write(f"Composed: {composed}")
        st.write(f"Parent prob: {p1:.2f} | Parent margin: {p_margin:.2f}")
        if reasons: st.warning("Ambiguity: " + ", ".join(reasons))

    if similar:
        st.subheader("🔁 유사 사례 (Retriever-BERT)")
        for i, (txt_i, sc) in enumerate(similar, 1):
            st.write(f"Top {i}  |  **{sc:.4f}**  |  {txt_i}")
    else:
        st.caption("유사 사례 인덱스가 없거나 모델이 로드되지 않았습니다.")
