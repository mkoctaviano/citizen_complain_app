#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# scripts/worker.py
import sys, os

# ✅ Ensure project root (citizen_demo) is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# scripts/worker.py
import os
import time
import argparse
import traceback
from pathlib import Path

# --- .env first ---
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

# --- your code ---
from storage import init_db, get_conn, 결과_등록, 상태_변경  # type: ignore
from citizen_complain_app.inference_wrapper import run_full_inference  # type: ignore
from sqlalchemy import text as sql


def fetch_unprocessed_ids(limit: int = 10):
    """
    Grab complaints that have NO 처리결과 yet.
    We only process items with no result rows to avoid duplicate inference.
    """
    q = sql(
        """
        SELECT m."번호"
        FROM "민원" m
        LEFT JOIN "처리결과" r ON r."민원번호" = m."번호"
        WHERE r."결과번호" IS NULL
        ORDER BY m."번호" ASC
        LIMIT :k
        """
    )
    with get_conn() as con:
        rows = con.execute(q, {"k": limit}).scalars().all()
    return [int(x) for x in rows]


def fetch_minwon_payload(minwon_id: int):
    q = sql(
        """
        SELECT "번호","이름","연락처","주소","내용","접수경로","상태"
        FROM "민원"
        WHERE "번호" = :id
        """
    )
    with get_conn() as con:
        row = con.execute(q, {"id": minwon_id}).mappings().first()
    return dict(row) if row else None


from storage import 처리결과_기존, 결과_등록
from inference_wrapper import run_full_inference_legacy
import json

def process_one(row):
    """
    Process a single 민원 row and write 처리결과.
    """
    idx = row["번호"]
    text = row["내용"]

    # Skip already processed
    if 처리결과_기존(idx):
        return False

    # Run full inference (legacy for now)
    pred = run_full_inference_legacy(text)

    # Parse predicted fields
    dept = pred.get("department") or "공통확인"
    subdept = pred.get("subdepartment") or "공통확인"
    intent = list(pred.get("intents", {}).keys())[0] if pred.get("intents") else "미정"
    urgency = pred.get("urgency") or "보통"
    emotion = pred.get("emotion") or "중립"

    # Build full 기타 / extra field from model output
    extra = pred.get("extra", {}).copy()  # grab full 'extra' dict

    # Add useful router fields as fallback
    if "router" in extra:
        extra["공통확인_사유"] = extra["router"].get("공통확인_사유", "")
        extra["input_final"] = extra["router"].get("input_final", "")

    # Add optional tops if present
    if "상위부서Top2" in pred:
        extra["상위부서Top2"] = pred["상위부서Top2"]
    if "상위부서_후보TopK" in pred:
        extra["상위부서_후보TopK"] = pred["상위부서_후보TopK"]
    if "부서_후보TopK" in pred:
        extra["부서_후보TopK"] = pred["부서_후보TopK"]

    # Save result
    결과_등록(
        번호=idx,
        상위부서=dept,
        세부분야=subdept,
        의도=intent,
        긴급도=urgency,
        감정=emotion,
        기타=extra,
    )

    return True


def run_loop(interval: float = 5.0, batch: int = 10, model_version: str = "worker_v1", once: bool = False):
    """Main polling loop."""
    init_db()
    print(f"[worker] started (interval={interval}s, batch={batch}, once={once})")

    try:
        while True:
            ids = fetch_unprocessed_ids(limit=batch)
            if not ids:
                if once:
                    print("[worker] no work; exiting (--once).")
                    return
                time.sleep(interval)
                continue

            for mid in ids:
                try:
                    process_one(mid, model_version=model_version)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    print(f"[worker] ERROR handling #{mid}\n{traceback.format_exc()}")
                    try:
                        상태_변경(mid, "오류")
                    except Exception:
                        pass

            if once:  # finish current batch then exit
                print("[worker] processed a batch; exiting (--once).")
                return

    except KeyboardInterrupt:
        print("\n[worker] shutdown requested. Bye.")


def main():
    parser = argparse.ArgumentParser(description="Background worker for 민원 처리")
    parser.add_argument("--interval", type=float, default=float(os.getenv("WORKER_POLL_INTERVAL", "5")),
                        help="Polling interval in seconds (default 5)")
    parser.add_argument("--batch", type=int, default=int(os.getenv("WORKER_BATCH_SIZE", "10")),
                        help="Max complaints to process per poll (default 10)")
    parser.add_argument("--model-version", type=str, default=os.getenv("WORKER_MODEL_VERSION", "worker_v1"),
                        help="모델버전 value to store with 결과 (default worker_v1)")
    parser.add_argument("--once", action="store_true", help="Process available items once then exit")
    args = parser.parse_args()

    run_loop(interval=args.interval, batch=args.batch, model_version=args.model_version, once=args.once)


if __name__ == "__main__":
    main()

