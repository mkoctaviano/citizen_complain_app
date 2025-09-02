#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# scripts/run_model_test.py

import sys, os
# add project root (C:\citizen_demo) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from citizen_complain_app.inference_wrapper import run_full_inference
from storage import init_db, 민원_등록, 결과_등록, 상태_변경, 대시보드_조회

def main():
    init_db()
    text = "우리 동네 도로에 포트홀이 생겨 위험합니다. 긴급 조치 바랍니다."
    cid = 민원_등록("웹", "user@example.com", text, [])
    pred = run_full_inference(text)

    결과_등록(
        민원번호=cid,
        키워드=pred["keywords"],
        의도=pred["intents"],
        부서=pred["department"],
        세부분야=pred["subdepartment"],
        긴급도=pred["urgency"],
        감정=pred["emotion"],
        모델버전=pred["model_version"],
        기타=pred.get("extra", {})
    )
    상태_변경(cid, "처리완료")
    print(대시보드_조회().tail(1))

if __name__ == "__main__":
    main()

