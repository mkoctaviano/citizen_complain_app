#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# citizen_complain_app/model_core.py
import os, json, re, unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Cache shims: use Streamlit cache if available, else lru_cache (no warnings)
# -----------------------------------------------------------------------------
from functools import lru_cache
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

def cache_resource(show_spinner=False):
    def deco(fn):
        if _HAS_ST:
            return st.cache_resource(show_spinner=show_spinner)(fn)
        return lru_cache(maxsize=1)(fn)
    return deco

def cache_data(show_spinner=False):
    def deco(fn):
        if _HAS_ST:
            return st.cache_data(show_spinner=show_spinner)(fn)
        return lru_cache(maxsize=1)(fn)
    return deco

# -----------------------------------------------------------------------------
# HuggingFace / libs
# -----------------------------------------------------------------------------
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
)

from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process as fuzz_process
import pickle

# -----------------------------------------------------------------------------
# Paths / constants (same as your app.py)
# -----------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
KEI_PKL     = BASE_DIR / "kei_booster.pkl"
PARENT_DIR  = BASE_DIR / "main_model"
CHILD_DIR   = BASE_DIR / "child"
CHILD_REG   = BASE_DIR / "child_registry.json"
CSV_PATH    = BASE_DIR / "_merged_unidept_88_6cols_with_parent_fix_clean.csv"

CAUSE_DIR   = BASE_DIR / "cause_tagger"
RETR_DIR    = BASE_DIR / "retriever_bert"

PRIORITY_DIR = BASE_DIR / "priority_model"
URGENCY_DIR  = PRIORITY_DIR / "urgency_model_roberta_reg"
EMOTION_DIR  = PRIORITY_DIR / "KoElectra_emotion"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256

# -----------------------------------------------------------------------------
# KEIBooster
# -----------------------------------------------------------------------------
def _normalize_ko(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").strip()
    return re.sub(r"\s+", " ", s)

@cache_resource(show_spinner=False)
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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Parent / Child Router
# -----------------------------------------------------------------------------
@cache_resource(show_spinner=False)
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
            print(f"[child_registry.json parse] {e}")

    candidates = [(name, _norm_label(name), path) for name, path in child_dirs.items()]
    mapping: Dict[str, Path] = {}
    for pl in parent_classes:
        pln = _norm_label(pl)
        exact = [p for (orig, norm, p) in candidates if norm == pln]
        if exact:
            mapping[pl] = exact[0]; continue
        if candidates:
            cand_names = [orig for (orig, norm, p) in candidates]
            m = fuzz_process.extractOne(pl, cand_names, scorer=None)
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

@cache_resource(show_spinner=False)
def load_child_model(path: Path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(DEVICE).eval()
    le_path = path / "label_encoder.json"
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

# -----------------------------------------------------------------------------
# Cause (pipeline)
# -----------------------------------------------------------------------------
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

@cache_resource(show_spinner=False)
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

# -----------------------------------------------------------------------------
# Retriever-BERT (similar)
# -----------------------------------------------------------------------------
@cache_resource(show_spinner=False)
def load_retriever(model_dir: Path):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_dir, local_files_only=True).to(DEVICE).eval()
    return tok, mdl

@cache_data(show_spinner=False)
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
    return v

def retriever_search(tok, mdl, E, meta, q: str, k=5):
    v = retriever_embed(tok, mdl, q)
    sims = (E @ v.T).ravel()
    idx = sims.argsort()[::-1][:k]
    return [(meta[i], float(sims[i])) for i in idx]

# -----------------------------------------------------------------------------
# Priority (Urgency + Emotion)
# -----------------------------------------------------------------------------
ymin, ymax = 0.013, 2.181

def unscale_y(y):
    return np.asarray(y, np.float32) * (ymax - ymin) + ymin

def normalize_from_range(y):
    y = np.asarray(y, np.float32)
    return (y - ymin) / (ymax - ymin + 1e-12)

DEFAULT_EMOTION_WEIGHTS = {
    "angry": 1.00, "fear": 0.90, "surprise": 0.80, "disgust": 0.70,
    "happy": 0.60, "sad": 0.50, "neutral": 0.30,
}

@cache_resource(show_spinner=False)
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
    assumed = ["angry","disgust","fear","sad","surprise","happy","neutral"]
    return np.asarray([DEFAULT_EMOTION_WEIGHTS.get(x, 0.50) for x in assumed[:model.config.num_labels]], np.float32)

@torch.no_grad()
def predict_urgency(tok_urg, mdl_urg, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    enc = tok_urg(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    logits_scaled = mdl_urg(**enc).logits.squeeze(-1)
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

# -----------------------------------------------------------------------------
# Keyword vote table
# -----------------------------------------------------------------------------
@cache_data(show_spinner=False)
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

