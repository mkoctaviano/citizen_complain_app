#!/usr/bin/env python
# coding: utf-8

# In[1]:


# citizen_complain_app/model_core.py
import os, json, re, unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass

# -------------------------------------------------------------------
# Streamlit-safe caching (falls back to lru_cache if not in Streamlit)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# HuggingFace / libs
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent

# KEI booster (keywords/intents)
KEI_PKL     = BASE_DIR / "kei_booster.pkl"

# Classifier (parent + children + optional registry)
PARENT_DIR  = BASE_DIR / "main_model"
CHILD_DIR   = BASE_DIR / "child_models"
CHILD_REG   = BASE_DIR / "child_registry.json"

# Cause tagger
CAUSE_DIR   = BASE_DIR / "cause_tagger"

# Similarity (Retriever-BERT) — model + index folder
RETR_DIR    = BASE_DIR / "retriever_bert"         # model (config + weights)
RETR_INDEX  = BASE_DIR / "retriever_bert"         # index: embeddings_retriever.npy + meta.jsonl

# Priority models
PRIORITY_DIR = BASE_DIR / "priority_model"
URGENCY_DIR  = PRIORITY_DIR / "urgency_model_roberta_reg"
EMOTION_DIR  = PRIORITY_DIR / "KoElectra_emotion"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256

# -------------------------------------------------------------------
# Input normalizer
# -------------------------------------------------------------------
def _textify(x) -> str:
    """Convert any input to a safe unicode string for tokenizers."""
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return x.decode(errors="ignore")
    if isinstance(x, (list, tuple, set)):
        return " ".join(_textify(v) for v in x)
    try:
        import pandas as _pd
        import numpy as _np
        if isinstance(x, (_pd.Timestamp, _np.datetime64)):
            return str(_pd.Timestamp(x))
    except Exception:
        pass
    return str(x)

# -------------------------------------------------------------------
# KEIBooster
# -------------------------------------------------------------------
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
    from rapidfuzz import fuzz as _fuzz
    fuzzy_sc = np.array([_fuzz.partial_ratio(t, kw)/100.0 for kw in kws])
    substr = np.array([0.1 if kw in t else 0.0 for kw in kws])
    score = 0.7 * cos + 0.3 * fuzzy_sc + substr
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

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Parent / Child Router
# -------------------------------------------------------------------
@cache_resource(show_spinner=False)
def load_parent_model(parent_dir: Path):
    assert (parent_dir / "label_encoder.json").exists(), "parent/label_encoder.json 없음"
    with open(parent_dir / "label_encoder.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    tok = AutoTokenizer.from_pretrained(parent_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(parent_dir, local_files_only=True).to(DEVICE).eval()
    return tok, mdl, classes

def build_child_map(child_dir: Optional[Path], child_registry: Optional[Path],
                    parent_classes: List[str], fuzzy_threshold: int = 85) -> Dict[str, Path]:
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
            print(f"[child_registry parse] {e}")

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
            if m and m[1] >= fuzzy_threshold:
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
    i1 = int(order[0])
    i2 = int(order[1]) if len(order) > 1 else i1
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
    i1 = int(order[0])
    i2 = int(order[1]) if len(order) > 1 else i1
    label1 = labels_map.get(i1, f"class_{i1}")
    return label1, float(prob[i1]), float(prob[i1] - prob[i2]), None

@torch.no_grad()
def predict_parent_topk(tok, mdl, classes, text: str, k: int = 3):
    X = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
    logits = mdl(**X).logits[0]
    prob = torch.softmax(logits, dim=-1).cpu().numpy()
    k = min(k, len(prob))
    order = np.argsort(-prob)[:k]
    labels = [classes[i] if i < len(classes) else f"parent_{i}" for i in order]
    probs  = [float(prob[i]) for i in order]
    margin = probs[0] - (probs[1] if len(probs) > 1 else probs[0])
    return labels, probs, margin

@torch.no_grad()
def predict_child_topk(path: Path, text: str, k: int = 3):
    tok, mdl, labels_map = load_child_model(path)
    num_labels = int(mdl.config.num_labels)
    if num_labels == 1:
        return [labels_map[0]], [1.0], 1.0, "single_class"
    X = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
    logits = mdl(**X).logits[0]
    prob = torch.softmax(logits, dim=-1).cpu().numpy()
    k = min(k, len(prob))
    order = np.argsort(-prob)[:k]
    labels = [labels_map.get(int(i), f"class_{int(i)}") for i in order]
    probs  = [float(prob[int(i)]) for i in order]
    margin = probs[0] - (probs[1] if len(probs) > 1 else probs[0])
    return labels, probs, margin, None

@dataclass
class RouterConfig:
    parent_floor: float = 0.35
    parent_margin: float = 0.10
    child_floor:  float = 0.30
    child_margin: float = 0.10
    parent_topk:  int   = 3
    child_topk:   int   = 3

def classify_with_router(booster, parent_tok, parent_mdl, parent_classes, child_map, text: str,
                         cfg: RouterConfig = RouterConfig()) -> Dict[str, Any]:
    # Step 1: KEI feature extraction
    keywords = kei_extract_keywords(booster, text, top_k=5)
    intent   = kei_extract_intent(booster, text)
    if intent == "공통확인":
        intent = "미정"
    inp = kei_compose_input(text, keywords=keywords[:3], intent=intent)

    # Step 2: Parent prediction (Top-K)
    p_labs, p_probs, p_margin = predict_parent_topk(parent_tok, parent_mdl, parent_classes, inp, k=cfg.parent_topk)
    p_label = p_labs[0]
    p_prob = p_probs[0]

    # Step 3: Low-confidence parent → "공통확인"
    if (p_prob < cfg.parent_floor) or (p_margin < cfg.parent_margin):
        return {
            "텍스트": text,
            "키워드Top": keywords,
            "의도": intent,
            "상위부서": "공통확인",
            "부서": "공통확인",
            "상위부서_후보TopK": [f"{l} ({p:.2f})" for l, p in zip(p_labs, p_probs)],
            "부서_후보TopK": [],
            "상위부서Top2": [f"{l} ({p:.2f})" for l, p in zip(p_labs[:2], p_probs[:2])],
            "input_final": inp,
            "공통확인_사유": "parent_prob<{:.2f}".format(cfg.parent_floor) if p_prob < cfg.parent_floor else "parent_margin<{:.2f}".format(cfg.parent_margin),
        }

    # Step 4: Route to child model
    c_path = child_map.get(p_label)
    if c_path is None:
        return {
            "텍스트": text,
            "키워드Top": keywords,
            "의도": intent,
            "상위부서": f"{p_label} ({p_prob:.2f})",
            "부서": f"{p_label} ({p_prob:.2f})",
            "상위부서_후보TopK": [],
            "부서_후보TopK": [],
            "상위부서Top2": [f"{l} ({p:.2f})" for l, p in zip(p_labs[:2], p_probs[:2])],
            "input_final": inp,
            "공통확인_사유": "",
        }

    # Step 5: Predict child
    c_label, c_prob, c_margin, note = predict_child(c_path, inp)

    # Step 6: Child = single class
    if note == "single_class":
        return {
            "텍스트": text,
            "키워드Top": keywords,
            "의도": intent,
            "상위부서": f"{p_label} ({p_prob:.2f})",
            "부서": f"{c_label} ({c_prob:.2f})",
            "상위부서_후보TopK": [],
            "부서_후보TopK": [],
            "상위부서Top2": [f"{l} ({p:.2f})" for l, p in zip(p_labs[:2], p_probs[:2])],
            "input_final": inp,
            "공통확인_사유": "",
        }

    # Step 7: Low-confidence child → "공통확인"
    if (c_prob < cfg.child_floor) or (c_margin < cfg.child_margin):
        c_labs, c_probs, _, note2 = predict_child_topk(c_path, inp, k=cfg.child_topk)
        c_list = [f"{c_labs[0]} (1.00)"] if note2 == "single_class" else [f"{l} ({p:.2f})" for l, p in zip(c_labs, c_probs)]
        return {
            "텍스트": text,
            "키워드Top": keywords,
            "의도": intent,
            "상위부서": f"{p_label} ({p_prob:.2f})",
            "부서": "공통확인",
            "상위부서_후보TopK": [f"{l} ({p:.2f})" for l, p in zip(p_labs, p_probs)],
            "부서_후보TopK": c_list,
            "상위부서Top2": [f"{l} ({p:.2f})" for l, p in zip(p_labs[:2], p_probs[:2])],
            "input_final": inp,
            "공통확인_사유": "child_prob<{:.2f}".format(cfg.child_floor) if c_prob < cfg.child_floor else "child_margin<{:.2f}".format(cfg.child_margin),
        }

    # Step 8: All confident → return both
        c_labs, c_probs, _, _ = predict_child_topk(c_path, inp, k=cfg.child_topk)
        return {
            "텍스트": text,
            "키워드Top": keywords,
            "의도": intent,
            "상위부서": f"{p_label} ({p_prob:.2f})",
            "부서": f"{c_label} ({c_prob:.2f})",
            "상위부서_후보TopK": [f"{l} ({p:.2f})" for l, p in zip(p_labs, p_probs)],
            "부서_후보TopK": [f"{l} ({p:.2f})" for l, p in zip(c_labs, c_probs)],
            "상위부서Top2": [f"{l} ({p:.2f})" for l, p in zip(p_labs[:2], p_probs[:2])],
            "input_final": inp,
            "공통확인_사유": "",
        }

@cache_resource(show_spinner=False)
def _router_artifacts():
    booster = load_kei_booster(KEI_PKL)
    p_tok, p_mdl, p_classes = load_parent_model(PARENT_DIR)
    c_map = build_child_map(CHILD_DIR, CHILD_REG, p_classes)
    return booster, p_tok, p_mdl, p_classes, c_map

def classify(text: str, cfg: RouterConfig = RouterConfig()) -> Dict[str, Any]:
    text = _textify(text).strip()
    booster, p_tok, p_mdl, p_classes, c_map = _router_artifacts()
    return classify_with_router(booster, p_tok, p_mdl, p_classes, c_map, text, cfg)

# -------------------------------------------------------------------
# Cause Extraction
# -------------------------------------------------------------------
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

_KO_ENDINGS = [
    "요","다","죠","임","음","슴",
    "습니다","입니다","했습니다","했습니다요",
    "해요","하세요","해주세요","해 주세요","해주십시오",
    "해주시기 바랍니다","해주시길 바랍니다","바랍니다",
    "부탁드립니다","요청드립니다","필요합니다","원합니다",
    "됩니까","되나요","되었어요","됐어요","맞습니다","맞나요",
    "막혀요","막힙니다","불편해요","불편합니다","해주세요요"
]
_ENDINGS_PATTERN = "|".join(map(re.escape, _KO_ENDINGS))
SENT_SPLIT = re.compile(
    rf"""\s*([^\n…!?\.]+?(?:{_ENDINGS_PATTERN}))(?=\s+|$|[~…!?\.])""",
    re.VERBOSE | re.DOTALL
)
JOSA = r"(은|는|이|가|을|를|에|에서|으로|와|과|에게|부터|까지|마다)"
BOUNDARY_LEFT  = re.compile(rf"{JOSA}$")
BOUNDARY_RIGHT = re.compile(rf"^{JOSA}")

def split_sentences_with_offsets(text: str):
    sents = []
    for m in SENT_SPLIT.finditer(text):
        s = m.group(1).strip()
        if s:
            sents.append({"text": s, "start": m.start(1), "end": m.end(1)})
    if sents:
        return sents
    parts, start = [], 0
    for line in text.splitlines(True):
        t = line.strip()
        if t:
            s = start + line.find(t); e = s + len(t)
            parts.append({"text": t, "start": s, "end": e})
        start += len(line)
    if parts:
        return parts
    t = text.strip()
    if t:
        i = text.find(t)
        return [{"text": t, "start": i, "end": i + len(t)}]
    return []

def expand_to_phrase(text: str, start: int, end: int, max_chars: int = 48):
    L = start
    while L > 0 and (not text[L-1].isspace()) and (L > start - max_chars):
        if text[L-1] in ",.!?…\n": break
        L -= 1
    R = end
    while R < len(text) and (not text[R].isspace()) and (R < end + max_chars):
        if text[R] in ",.!?…\n": break
        R += 1
    if L > 0 and BOUNDARY_LEFT.search(text[max(0, L-3):L]): L -= 1
    if R < len(text) and BOUNDARY_RIGHT.match(text[R:R+2]): R += 1
    return max(0, L), min(len(text), R)

def merge_overlaps(spans):
    spans = sorted(spans, key=lambda x: (-x["score"], x["start"], -x["end"]))
    kept = []
    for s in spans:
        overlapped = False
        for k in kept:
            if not (s["end"] <= k["start"] or s["start"] >= k["end"]):
                overlapped = True
                if s["score"] > k["score"] or (s["score"] == k["score"] and (s["end"]-s["start"]) > (k["end"]-k["start"])):
                    k.update(s)
                break
        if not overlapped:
            kept.append(s)
    kept.sort(key=lambda x: -x["score"])
    return kept

@cache_resource(show_spinner=False)
def load_cause_model(path: Path):
    if not _is_valid_model_dir(path):
        return None, None, None
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mdl = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True).to(DEVICE).eval()
    id2label = mdl.config.id2label
    return tok, mdl, id2label

def _id_to_label(pid, id2label):
    if isinstance(id2label, (list, tuple)): return id2label[pid]
    if isinstance(id2label, dict): return id2label.get(pid) or id2label.get(str(pid)) or str(pid)
    try: return id2label[pid]
    except Exception: return str(pid)

@torch.no_grad()
def predict_cause_spans(text: str,
                        target_label: Optional[str] = "CAUSE",
                        threshold: float = 0.60,
                        min_chars: int = 2,
                        top_k: Optional[int] = 2,
                        max_length: int = MAX_LEN):
    text = _textify(text)
    if not text.strip():
        return []

    tok, mdl, id2label = load_cause_model(CAUSE_DIR)
    if tok is None:
        return []

    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    out = mdl(**enc)
    probs = F.softmax(out.logits, dim=-1)         # [1, L, C]
    pred_ids = probs.argmax(dim=-1).squeeze(0).tolist()
    pred_scores = probs.max(dim=-1).values.squeeze(0).tolist()

    input_ids = enc["input_ids"].squeeze(0).tolist()
    toks_all = tok.convert_ids_to_tokens(input_ids)

    tokens, labels, scores, offs = [], [], [], []
    for t, pid, sc, of in zip(toks_all, pred_ids, pred_scores, offsets):
        if t in tok.all_special_tokens or of == [0,0]:
            continue
        lab = _id_to_label(pid, id2label)
        tokens.append(t); labels.append(lab); scores.append(float(sc)); offs.append(tuple(of))

    # BIO/BIOS merging
    spans = []
    cur_lab = None; cur_scores = []; cur_start = None; cur_end = None
    for lab, sc, (a,b) in zip(labels, scores, offs):
        if lab.startswith("B-"):
            if cur_lab: spans.append((cur_lab, cur_start, cur_end, float(np.mean(cur_scores))))
            cur_lab = lab[2:]; cur_start = a; cur_end = b; cur_scores = [sc]
        elif lab.startswith("I-") and cur_lab == lab[2:]:
            cur_end = b; cur_scores.append(sc)
        elif lab.startswith("S-"):
            if cur_lab: spans.append((cur_lab, cur_start, cur_end, float(np.mean(cur_scores))))
            spans.append((lab[2:], a, b, float(sc)))
            cur_lab = None; cur_scores = []; cur_start = None; cur_end = None
        else:
            if cur_lab: spans.append((cur_lab, cur_start, cur_end, float(np.mean(cur_scores))))
            cur_lab = None; cur_scores = []; cur_start = None; cur_end = None
    if cur_lab: spans.append((cur_lab, cur_start, cur_end, float(np.mean(cur_scores))))

    refined = []
    for lab, s, e, sc in spans:
        if target_label and lab != target_label: continue
        if sc < threshold: continue
        L, R = expand_to_phrase(text, s, e, max_chars=48)
        frag = re.sub(r"\s+", " ", text[L:R].strip())
        if len(frag) < min_chars: continue
        refined.append({"label": lab, "text": frag, "start": L, "end": R, "score": float(sc)})

    refined = merge_overlaps(refined)
    if top_k and top_k > 0:
        refined = refined[:top_k]

    sents = split_sentences_with_offsets(text)
    for r in refined:
        for stx in sents:
            if stx["start"] <= r["start"] and r["end"] <= stx["end"]:
                r["sentence"] = stx["text"]; r["sent_start"] = stx["start"]; r["sent_end"] = stx["end"]
                break
        if "sentence" not in r:
            r["sentence"] = text; r["sent_start"] = 0; r["sent_end"] = len(text)
    return refined

def run_cause(text: str,
              target_label: Optional[str] = "CAUSE",
              threshold: float = 0.60,
              min_chars: int = 2,
              top_k: int = 2) -> dict:
    spans = predict_cause_spans(
        text,
        target_label=target_label,
        threshold=threshold,
        min_chars=min_chars,
        top_k=top_k,
        max_length=MAX_LEN,
    )
    if not spans:
        return {"cause_label": "", "cause_span": "", "cause_score": 0.0, "extra": {"spans": []}}
    best = spans[0]
    return {
        "cause_label": best.get("label", "CAUSE"),
        "cause_span": best.get("text", ""),
        "cause_score": float(best.get("score", 0.0)),
        "extra": {"spans": spans},
    }

# -------------------------------------------------------------------
# Similarity (Retriever-BERT)
# -------------------------------------------------------------------
@cache_resource(show_spinner=False)
def load_retriever(model_dir: Path):
    assert (model_dir / "config.json").exists(), "Retriever model: config.json missing"
    assert (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists(), \
        "Retriever model weights missing"
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_dir, local_files_only=True).to(DEVICE).eval()
    return tok, mdl

@cache_data(show_spinner=False)
def load_retriever_index(index_dir: Path):
    emb_path = index_dir / "embeddings_retriever.npy"
    meta_path = index_dir / "meta.jsonl"
    E = np.load(emb_path).astype(np.float32)
    with open(meta_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    meta: List[Dict[str, Any]] = []
    for ln in raw_lines:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                meta.append(obj)
            else:
                meta.append({"text": str(obj)})
        except Exception:
            meta.append({"text": ln})
    E /= np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-12)
    return E, meta

@torch.no_grad()
def retriever_embed(tok, mdl, text: str):
    x = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN).to(DEVICE)
    v = mdl(**x).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)  # CLS
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
    return v

def retriever_search(text: str, k: int = 5, score_floor: float = 0.0) -> List[Dict[str, Any]]:
    text = _textify(text).strip()
    if not text:
        return []
    try:
        tok, mdl = load_retriever(RETR_DIR)
        E, meta = load_retriever_index(RETR_INDEX)
    except Exception:
        return []
    v = retriever_embed(tok, mdl, text)
    sims = (E @ v.T).ravel()
    order = sims.argsort()[::-1][:k]
    out: List[Dict[str, Any]] = []
    for i in order:
        sc = float(sims[i])
        if sc < score_floor:
            continue
        m = meta[i] if i < len(meta) else {}
        rec = {
            "score": sc,
            "meta": m,
            "text": m.get("내용") or m.get("text") or m.get("본문") or "",
            "id":   m.get("번호") or m.get("id") or m.get("민원번호"),
            "date": m.get("접수일시") or m.get("date"),
            "source": m.get("접수경로") or m.get("source"),
        }
        out.append(rec)
    return out

# -------------------------------------------------------------------
# Priority (Urgency + Emotion)  — same math as your Colab
# -------------------------------------------------------------------
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
    if not urg_dir.exists() or not (urg_dir / "config.json").exists():
        return None, None, None, None
    if not emo_dir.exists() or not (emo_dir / "config.json").exists():
        return None, None, None, None
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

def predict_priority_single(text: str, w_urg: float = 0.8, w_emo: float = 0.2) -> Optional[Dict[str, float]]:
    tok_urg, mdl_urg, tok_emo, mdl_emo = load_priority_models(URGENCY_DIR, EMOTION_DIR)
    if tok_urg is None or mdl_urg is None or tok_emo is None or mdl_emo is None:
        return None
    urg_raw, urg_norm = predict_urgency(tok_urg, mdl_urg, [text])
    emo_norm = predict_emotion_score(tok_emo, mdl_emo, [text])
    priority = combine_priority(urg_norm, emo_norm, w_urg, w_emo)
    return {
        "urgency_raw": float(np.round(urg_raw[0], 3)),
        "urgency_norm": float(np.round(urg_norm[0], 3)),
        "emotion_norm": float(np.round(emo_norm[0], 3)),
        "priority": float(np.round(priority[0], 3)),
    }

# -------------------------------------------------------------------
# NEW: label cleaners & grade mappers for UI/DB
# -------------------------------------------------------------------
_SCORE_SUFFIX_RE = re.compile(r"\s*\(\s*[-+]?\d+(?:\.\d+)?\s*\)\s*$")

def strip_score_suffix(s: Any) -> str:
    """'교통 (0.96)' -> '교통'; numeric-only like '0.63' -> '-' (hidden)."""
    if isinstance(s, (int, float)):
        return "-"
    s = str(s or "").strip()
    if not s:
        return "-"
    s = _SCORE_SUFFIX_RE.sub("", s).strip()
    try:
        float(s)  # numeric string?
        return "-"
    except Exception:
        return s

def grade_emotion_kr(score: Optional[float]) -> str:
    """감정 스코어(0~1) → 한국어 등급."""
    if score is None:
        return "-"
    s = float(score)
    if s >= 0.9: return "격한 불만"
    if s >= 0.7: return "강한 불편감"
    if s >= 0.5: return "불쾌감 표출"
    if s >= 0.3: return "경미한 불만"
    return "불만 없음"

def grade_priority_kr(score: Optional[float]) -> str:
    """우선순위 스코어(0~1) → 한국어 등급."""
    if score is None:
        return "-"
    s = float(score)
    if s >= 0.9: return "즉시 대응"
    if s >= 0.7: return "신속 대응"
    if s >= 0.5: return "일반 처리"
    if s >= 0.3: return "관찰 대상"
    return "후순위"

# -------------------------------------------------------------------
# Keyword vote table (unchanged)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Convenience: one-call that returns both classification & cause
# -------------------------------------------------------------------
def classify_and_cause(text: str) -> Dict[str, Any]:
    text = _textify(text).strip()
    cls = classify(text)
    cause = run_cause(text)
    return {
    "keywords": cls.get("키워드Top", []),
    "intents": {"의도": cls.get("의도", "미정")},
    "department": strip_score_suffix(cls.get("상위부서", "")),
    "subdepartment": strip_score_suffix(cls.get("부서", "")),
    "urgency": urgency_label,
    "emotion": emotion_label,
    "model_version": "chatbot_v1",
    "extra": {
        "router": cls,
        "cause": cause,
        "similarity": sim,
        "priority": pr,
        "상위부서Top2": cls.get("상위부서Top2", []),
        "상위부서_후보TopK": cls.get("상위부서_후보TopK", []),
        "부서_후보TopK": cls.get("부서_후보TopK", []),
        "공통확인_사유": cls.get("공통확인_사유", ""),
    },
}
# -------------------------------------------------------------------
# 🔶 Single entrypoint for the app (classification + cause + sim + priority)
# -------------------------------------------------------------------
def run_full_inference(text: str, k_sim: int = 5) -> Dict[str, Any]:
    text = _textify(text).strip()
    if not text:
        return {
            "keywords": [],
            "intents": {"의도": "미정"},
            "department": "",
            "subdepartment": "",
            "urgency": None,
            "emotion": None,
            "model_version": "chatbot_v1",
            "extra": {
                "router": {},
                "cause": {"cause_span": "", "cause_score": 0.0},
                "similarity": [],
                "priority": None,
                "상위부서Top2": [],
                "상위부서_후보TopK": [],
                "부서_후보TopK": [],
                "공통확인_사유": "",
            },
        }

    cls = classify(text)
    cause = run_cause(text)
    sim = retriever_search(text, k=k_sim, score_floor=0.0)
    pr = predict_priority_single(text)

    urgency_label = grade_priority_kr(pr["priority"]) if pr else None
    emotion_label = grade_emotion_kr(pr["emotion_norm"]) if pr else None

    return {
        "keywords": cls.get("키워드Top", []),
        "intents": {"의도": cls.get("의도", "미정")},
        "department": strip_score_suffix(cls.get("상위부서", "")),
        "subdepartment": strip_score_suffix(cls.get("부서", "")),
        "urgency": urgency_label,
        "emotion": emotion_label,
        "model_version": "chatbot_v1",
        "extra": {
            "router": cls,
            "cause": cause,
            "similarity": sim,
            "priority": pr,
            # ✅ Promote these for UI + DB compatibility
            "상위부서Top2": cls.get("상위부서Top2", []),
            "상위부서_후보TopK": cls.get("상위부서_후보TopK", []),
            "부서_후보TopK": cls.get("부서_후보TopK", []),
            "공통확인_사유": cls.get("공통확인_사유", ""),
        },
    }

# ----------------------------
# Backward-compatibility shims
# ----------------------------
def load_cause_pipeline(*args, **kwargs):
    """v1 compatibility stub."""
    return None

def run_cause_pipeline(pl, text: str, *args, **kwargs):
    """v1 compatibility: use v2 run_cause()."""
    return run_cause(text)


# In[ ]:




