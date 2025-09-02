#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# scripts/import_old_complaints.py
import sys, os, pathlib as _p
import pandas as pd

# make project root importable (so "import storage" works)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import storage  # uses DATABASE_URL from .env; run storage.init_db() once before importing

def load_table(path: _p.Path, limit: int | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")  # handles BOM safely
    else:
        # pip install openpyxl (only needed for .xlsx)
        df = pd.read_excel(path, engine="openpyxl")
    if limit:
        df = df.head(limit)
    return df

def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print("Usage: python -m scripts.import_old_complaints <file.csv|file.xlsx> [--limit N]")
        sys.exit(2)

    path = _p.Path(argv[1]).expanduser().resolve()
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)

    # optional --limit
    limit = None
    if len(argv) >= 4 and argv[2] == "--limit":
        limit = int(argv[3])

    print("📄 Loading:", path)
    df = load_table(path, limit=limit)
    print("✅ Loaded rows:", len(df))

    # ensure tables exist
    storage.init_db()

    # column names expected in your sheet
    COL_TEXT = "텍스트"
    COL_DEPT = "부서"
    COL_CAT  = "카테고리"
    COL_INTENT = "의도"
    COL_KW   = "키워드"

    inserted = 0
    for _, r in df.iterrows():
        text = str(r.get(COL_TEXT, "") or "").strip()
        if not text:
            continue

        # 1) insert into 민원
        mid = storage.민원_등록(
            접수경로="old_import",
            연락처=None,
            내용=text,
            첨부경로목록=[],
            이름=None,
            주소=None,
        )

        # 2) insert into 처리결과 (optional fields)
        kw = r.get(COL_KW)
        intent = r.get(COL_INTENT)
        storage.결과_등록(
            민원번호=mid,
            키워드=[kw] if isinstance(kw, str) and kw.strip() else [],
            의도={"의도": intent} if isinstance(intent, str) and intent.strip() else {},
            부서=(r.get(COL_DEPT) or None),
            세부분야=(r.get(COL_CAT) or None),
            긴급도=None,
            감정=None,
            모델버전="bulk_v1",
            기타={},
        )

        inserted += 1
        if inserted % 1000 == 0:
            print(f"… inserted {inserted:,} rows")

    print(f"🎉 Finished. Inserted {inserted:,} complaints.")

if __name__ == "__main__":
    main(sys.argv)

