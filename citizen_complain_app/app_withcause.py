# === kds_streamlit_app.py ===
import os, json, re, unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,  # ⬅ cause pipeline
)

from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz, process as fuzz_process
import pickle

# -----------------------
# Paths
# -----------------------
BASE_DIR   = Path(__file__).resolve().parent
KEI_PKL    = BASE_DIR / "kei_booster.pkl"
PARENT_DIR = BASE_DIR / "main_model"
CHILD_DIR  = BASE_DIR / "child"                  # optional
CHILD_REG  = BASE_DIR / "child_registry.json"    # optional
CSV_PATH   = BASE_DIR / "_merged_unidept_88_6cols_with_parent_fix_clean.csv"
CAUSE_DIR  = BASE_DIR / "cause_tagger"           # token classification model

# -----------------------
# KEIBooster
# -----------------------
def _normalize_ko(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").strip()
    return re.sub(r"\s+", " ", s)

class KEIBooster:
    def __init__(self, pkl_path: Path):
        with open(pkl_path, "rb") as f:
            blob = pickle.load(f)
        self.keyword_list     = blob["keyword_list"]
        self.intent_list      = blob["intent_list"]
        self.intent2examples  = blob["intent2examples"]
        self.kw_emb           = blob["kw_emb"]
        self.intent_proto     = blob["intent_proto"]
        self.cfg              = blob["cfg"]
        self.model            = SentenceTransformer(self.cfg["sbert_model_name"])

    def extract_keywords(self, text: str, top_k: int = 5):
        t = _normalize_ko(text)
        if not self.keyword_list or self.kw_emb is None:
            return []
        t_vec  = self.model.encode([t], convert_to_numpy=True, normalize_embeddings=True)[0]
        cos    = self.kw_emb @ t_vec
        fuzzy  = np.array([fuzz.partial_ratio(t, kw)/100.0 for kw in self.keyword_list])
        substr = np.array([0.1 if kw in t else 0.0 for kw in self.keyword_list])
        score  = 0.7 * cos + 0.3 * fuzzy + substr
        idx    = np.argsort(-score)[:top_k]
        return [self.keyword_list[i] for i in idx]

    def extract_intents(self, text: str):
        t = _normalize_ko(text)
        if not self.intent_list or self.intent_proto is None:
            return "미정"
        t_vec   = self.model.encode([t], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims    = self.intent_proto @ t_vec
        best_id = int(np.argmax(sims))
        best_sc = float(sims[best_id])
        if best_sc < 0.35:
            return "미정"
        return self.intent_list[best_id]

    def compose_input(self, text: str, keywords=None, intent=None):
        keywords = keywords or self.extract_keywords(text, top_k=3)
        intent   = intent if intent is not None else self.extract_intents(text)
        kw_str   = ";".join(dict.fromkeys(keywords)) if keywords else ""
        it_str   = intent or "미정"
        return f"[키워드:{kw_str}][의도:{it_str}] {text}"

# -----------------------
# Utilities
# -----------------------
def _is_valid_model_dir(path: Path) -> bool:
    if not path.is_dir(): return False
    has_cfg = (path/"config.json").exists()
    has_wts = (path/"model.safetensors").exists() or (path/"pytorch_model.bin").exists()
    return has_cfg and has_wts

def _norm_label(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"[^0-9a-z가-힣]", "", s)
    return s

# -----------------------
# Cause helpers (Colab parity)
# -----------------------
def normalize_cause_label(label: str) -> str:
    """원인 라벨/부서명 정제: '…과','…부','…팀' 제거 + 공백 정리"""
    if not label:
        return label
    label = re.sub(r"(과|부|팀)$", "", label)
    return label.strip()

def _choose_subject_particle(word: str) -> str:
    if not word:
        return "이"
    ch = word[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        jong = (code - 0xAC00) % 28
        return "이" if jong != 0 else "가"
    return "이"

def format_cause_sentence(info: dict) -> str:
    """추출된 원인 구간(span)을 자연스러운 문장으로 변환"""
    span = (info.get("cause_span") or "").strip()
    if not span:
        return "해당 민원에서 명확한 원인을 추출하지 못했습니다."
    particle = _choose_subject_particle(span)
    return f"{span}{particle} 원인으로 확인됩니다."

# -----------------------
# Cached builders for heavy objects
# -----------------------
@st.cache_resource(show_spinner=False)
def _load_booster_cached(pkl_path: Path) -> KEIBooster:
    return KEIBooster(pkl_path)

@st.cache_resource(show_spinner=False)
def _load_parent_cached(parent_dir: Path):
    tok = AutoTokenizer.from_pretrained(parent_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(parent_dir, local_files_only=True).eval()
    with open(parent_dir/"label_encoder.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]
    le = LabelEncoder().fit(classes)
    return tok, mdl, classes, le

@st.cache_resource(show_spinner=False)
def _build_cause_pipeline(path: Path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mdl = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except Exception:
        device = -1
    return pipeline(
        "token-classification",
        model=mdl,
        tokenizer=tok,
        aggregation_strategy="simple",
        device=device,
    )

# -----------------------
# ComplaintRouter (+ Cause)
# -----------------------
class ComplaintRouter:
    def __init__(self, parent_dir: Path, child_dir: Optional[Path], child_registry: Optional[Path], csv_path: Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.booster = _load_booster_cached(KEI_PKL)

        # Parent
        assert parent_dir.exists(), f"parent 폴더가 없습니다: {parent_dir}"
        self.parent_tok, self.parent_model, self.parent_classes, self.parent_le = _load_parent_cached(parent_dir)
        self.parent_model = self.parent_model.to(self.device)

        # Child registry + dirs
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
                print("⚠️ child_registry.json parse error:", e)

        # Fuzzy map child dirs to parent labels
        self.child_candidates = [(name, _norm_label(name), path) for name, path in child_dirs.items()]
        self.child_map: Dict[str, Path] = {}
        for pl in self.parent_classes:
            pln = _norm_label(pl)
            exact = [p for (orig, norm, p) in self.child_candidates if norm == pln]
            if exact:
                self.child_map[pl] = exact[0]; continue
            if self.child_candidates:
                cand_names = [orig for (orig, norm, p) in self.child_candidates]
                m = fuzz_process.extractOne(pl, cand_names, scorer=fuzz.token_sort_ratio)
                if m and m[1] >= 85:
                    matched = m[0]
                    for (orig, norm, p) in self.child_candidates:
                        if orig == matched:
                            self.child_map[pl] = p; break

        # Keyword vote table
        self.kw2parent = defaultdict(Counter)
        if CSV_PATH.exists():
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            for _, row in df.iterrows():
                for kw in re.split(r"[,\|;/]", str(row.get("키워드", ""))):
                    kw = kw.strip()
                    if kw:
                        self.kw2parent[kw][str(row.get("상위부서", "")).strip()] += 1
        else:
            print(f"ℹ️ Keyword vote CSV not found at {csv_path}")

        # Cause model (pipeline)
        self.cause_pipe = None
        if _is_valid_model_dir(CAUSE_DIR):
            try:
                self.cause_pipe = _build_cause_pipeline(CAUSE_DIR)
            except Exception as e:
                print("❌ Cause pipeline load error:", e)
        else:
            print(f"ℹ️ Cause model unavailable at {CAUSE_DIR}")

    @torch.no_grad()
    def _predict_parent_top2(self, s: str) -> Tuple[str, float, float]:
        X = self.parent_tok(s, return_tensors="pt", truncation=True, padding=True, max_length=256).to(self.device)
        logits = self.parent_model(**X).logits[0]
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        order = np.argsort(-prob)
        i1, i2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        parent1 = self.parent_classes[i1] if i1 < len(self.parent_classes) else f"parent_{i1}"
        return parent1, float(prob[i1]), float(prob[i1] - prob[i2])

    @torch.no_grad()
    def _predict_child(self, label: str, s: str) -> Tuple[str, float, float, Optional[str]]:
        path = self.child_map.get(label)
        if path is None:
            return label, 1.0, 1.0, "no_child"
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(self.device).eval()
        le  = None
        le_json = path / "label_encoder.json"
        if le_json.exists():
            cls = json.load(open(le_json, "r", encoding="utf-8"))["classes"]
            le = LabelEncoder().fit(cls)
        X = tok(s, return_tensors="pt", truncation=True, padding=True, max_length=256).to(self.device)
        logits = mdl(**X).logits[0]
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
        order = np.argsort(-prob)
        i1, i2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        if le is not None:
            sub = le.inverse_transform([i1])[0]
        else:
            sub = f"class_{i1}"
        return sub, float(prob[i1]), float(prob[i1] - prob[i2]), None

    def _predict_cause(self, text: str) -> Dict[str, Any]:
        """
        Colab parity:
        - run token-classification pipeline (aggregation_strategy='simple')
        - pick max-score span
        - normalize label + build natural sentence
        """
        if self.cause_pipe is None:
            info = {"cause_label": "N/A", "cause_span": "", "cause_score": 0.0}
            return {**info, "cause_sentence": "원인 모델이 준비되지 않았습니다."}

        spans = self.cause_pipe(text)  # [{'entity_group','word','score',...}, ...]
        if not spans:
            info = {"cause_label": "", "cause_span": "", "cause_score": 0.0}
            return {**info, "cause_sentence": format_cause_sentence(info)}

        best = max(spans, key=lambda r: r.get("score", 0.0))
        span = (best.get("word") or "").strip()
        info = {
            "cause_label": normalize_cause_label(span),
            "cause_span": span,
            "cause_score": float(best.get("score", 0.0)),
        }
        return {**info, "cause_sentence": format_cause_sentence(info)}

    def predict(self, text: str) -> Dict[str, Any]:
        # KEI
        intent   = self.booster.extract_intents(text)
        if intent == "공통확인":  # never surface '공통확인' as an intent
            intent = "미정"
        keywords = self.booster.extract_keywords(text, top_k=5)
        inp      = self.booster.compose_input(text, keywords=keywords[:3], intent=intent)

        # Parent + ambiguity gates
        parent1, p1, p_margin = self._predict_parent_top2(inp)
        ambiguous = []
        if p1 < 0.35: ambiguous.append("parent_prob<0.35")
        if p_margin < 0.10: ambiguous.append("child_margin<0.1")  # keep message as earlier UI

        # Keyword votes
        if keywords:
            votes = Counter()
            for kw in keywords:
                for dept, c in self.kw2parent.get(kw, {}).items():
                    votes[dept] += c
            if votes:
                tot = sum(votes.values())
                top2 = votes.most_common(2)
                if len(top2) == 1 or (top2[0][1] - top2[1][1]) / max(tot, 1) < 0.15:
                    ambiguous.append("kw vote conflict")

        if ambiguous:
            parent_label = "공통확인"
            sub_label    = "공통확인"
        else:
            parent_label = parent1
            # child
            sub_label, c1, c_margin, note = self._predict_child(parent1, inp)
            if note == "no_child":
                sub_label = parent1  # no child model => mirror parent per your spec
            else:
                if c1 < 0.30 or c_margin < 0.10:
                    sub_label = "공통확인"

        # Cause (pipeline with natural sentence)
        cinfo = self._predict_cause(text)

        return {
            "텍스트": text,
            "의도": intent,
            "키워드": keywords,
            "input_final": inp,
            "상위부서": parent_label,
            "하위부서": sub_label,
            "cause_label": cinfo["cause_label"],
            "cause_prob": cinfo["cause_score"],
            "cause_span": cinfo["cause_span"],
            "cause_sentence": cinfo["cause_sentence"],
            "parent_prob": p1,
            "ambiguous": bool(ambiguous),
            "reasons": ambiguous,
        }

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Citizen Complaint Classifier", layout="centered")
st.title("🧠 Citizen Complaint Classifier")
st.caption("🔍 Enter a Korean complaint")

text = st.text_area("Enter a Korean complaint", label_visibility="visible")

# Router availability
router_ready = PARENT_DIR.exists()
if "router" not in st.session_state and router_ready:
    st.session_state.router = ComplaintRouter(
        parent_dir=PARENT_DIR,
        child_dir=CHILD_DIR if CHILD_DIR.exists() else None,
        child_registry=CHILD_REG if CHILD_REG.exists() else None,
        csv_path=CSV_PATH,
    )

if st.button("🔍 Analyze with Parent + Child Models") and text.strip():
    if not router_ready:
        st.error("라우터가 준비되지 않았습니다. parent/ 와 child(또는 child_registry.json)를 확인하세요.")
    else:
        result = st.session_state.router.predict(text)

        # Parent / Child
        if result["상위부서"] == "공통확인":
            st.error("공통확인 fallback (사유: " + ", ".join(result["reasons"]) + ")")
        else:
            st.success(f"✅ Top-Level Department (상위부서): {result['상위부서']} ({result['parent_prob']:.2f})")
            st.info(f"📁 Sub-Department (하위부서): {result['하위부서']}")

        # Cause (natural sentence + details)
        st.warning(f"🧩 Cause (사유): {result['cause_sentence']}")
        st.caption(
            f"원인추출 스팬: {result.get('cause_span','') or 'N/A'}  |  score={result.get('cause_prob',0.0):.2f}"
        )

        # Debug
        with st.expander("🧠 KEI Booster Debug Info"):
            st.write(f"🔑 Keywords: {result['키워드']}")
            st.write(f"💬 Intent: {result['의도']}")
            st.write(f"🧠 Composed Input: {result['input_final']}")
            st.write(f"📊 Parent prob: {result['parent_prob']:.2f}")
            if result["ambiguous"]:
                st.warning("⚠️ Ambiguity reasons: " + ", ".join(result["reasons"]))
