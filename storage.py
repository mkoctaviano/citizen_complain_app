# storage.py
import os
import re
import json
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text as sql
from sqlalchemy.engine import Engine


# -------------------------------------------------------------------
# .env bootstrap (no page changes required)
# -------------------------------------------------------------------
_LOADED_ENV_FROM: Optional[str] = None  # for debug / introspection

def _manual_parse_env(env_path: Path) -> None:
    """
    Minimal .env reader (no python-dotenv required).
    - UTF-8 with BOM safe
    - Ignores comments/blank lines
    - Handles quoted values
    """
    global _LOADED_ENV_FROM
    try:
        text = env_path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        text = env_path.read_text(encoding="utf-8", errors="ignore")

    line_re = re.compile(r"""^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$""")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
            v = v[1:-1]
        os.environ.setdefault(k, v)

    _LOADED_ENV_FROM = str(env_path)


def _find_dotenv_near_storage() -> Optional[Path]:
    """
    Walk upwards from this file to find the nearest .env (up to 5 levels).
    In your repo this will typically be citizen_demo/.env.
    """
    here = Path(__file__).resolve().parent
    for d in [here, *list(here.parents)[:5]]:
        p = d / ".env"
        if p.exists():
            return p
    return None


def _bootstrap_env() -> None:
    """
    Load DATABASE_URL from:
      1) existing OS env (if present)
      2) .env in current working directory
      3) .env near storage.py (walking upwards)
      4) manual parse fallback (if python-dotenv is missing)
    """
    global _LOADED_ENV_FROM

    if os.getenv("DATABASE_URL"):
        return

    debug = os.getenv("DEBUG_ENV") == "1"

    # 1) try python-dotenv from CWD
    try:
        from dotenv import load_dotenv, find_dotenv
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)
            if debug:
                print(f"[storage] dotenv loaded from CWD: {found}")
            if os.getenv("DATABASE_URL"):
                _LOADED_ENV_FROM = found
                return
    except Exception as e:
        if debug:
            print(f"[storage] python-dotenv (CWD) failed: {e}")

    # 2) try python-dotenv from near storage.py
    try:
        from dotenv import load_dotenv  # noqa: F401
        near = _find_dotenv_near_storage()
        if near:
            load_dotenv(near.as_posix(), override=False)
            if debug:
                print(f"[storage] dotenv loaded near storage: {near}")
            if os.getenv("DATABASE_URL"):
                _LOADED_ENV_FROM = str(near)
                return
    except Exception as e:
        if debug:
            print(f"[storage] python-dotenv (near storage) failed: {e}")

    # 3) manual fallback (no dependency on python-dotenv)
    for candidate in [Path.cwd() / ".env", _find_dotenv_near_storage()]:
        if candidate and candidate.exists():
            _manual_parse_env(candidate)
            if debug:
                print(f"[storage] manual .env parse: {candidate}")
            if os.getenv("DATABASE_URL"):
                return

    if debug:
        print("[storage] DATABASE_URL still not present after bootstrap")
        print("[storage] CWD:", os.getcwd())
        print("[storage] __file__:", __file__)
        print("[storage] .env near storage exists?:", bool(_find_dotenv_near_storage()))


_bootstrap_env()


# -------------------------------------------------------------------
# Connection resolution
# -------------------------------------------------------------------
def _get_database_url() -> str:
    """
    Resolve database URL from env:
      DATABASE_URL=postgresql+psycopg://user:pass@host/db?sslmode=require

    For local development (optional):
      ALLOW_SQLITE=1
      APP_DB_PATH=./data/민원저장소.db
    """
    url = os.getenv("DATABASE_URL")
    if url:
        # Some hosts append an extra flag – harmless to remove for psycopg
        url = url.replace("&channel_binding=require", "")
        # Normalize scheme to psycopg v3 if plain postgresql:// was used
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    # Optional SQLite fallback for local dev
    if os.getenv("ALLOW_SQLITE") == "1":
        local_path = os.environ.get("APP_DB_PATH", os.path.join("data", "민원저장소.db"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        return f"sqlite:///{local_path}"

    where = _LOADED_ENV_FROM or "(no .env found)"
    raise RuntimeError(
        "DATABASE_URL not set. Add to a UTF-8 .env, e.g.\n"
        "DATABASE_URL=postgresql+psycopg://user:pass@host/db?sslmode=require\n"
        f"(checked: {where}, CWD={Path.cwd()})"
    )


DB_URL: str = _get_database_url()
_ENGINE: Engine = create_engine(DB_URL, future=True, pool_pre_ping=True)


def _is_sqlite() -> bool:
    return DB_URL.startswith("sqlite")


@contextmanager
def get_conn():
    """Transaction-scoped connection (auto-commit on exit)."""
    with _ENGINE.begin() as conn:
        yield conn


# -------------------------------------------------------------------
# Schema init / upgrade
# -------------------------------------------------------------------
def init_db() -> None:
    """Create/upgrade tables for Postgres or SQLite."""
    with get_conn() as con:
        if _is_sqlite():
            con.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS "민원" (
                "번호"      INTEGER PRIMARY KEY AUTOINCREMENT,
                "접수일시"  TEXT,
                "접수경로"  TEXT,
                "연락처"    TEXT,
                "내용"      TEXT,
                "첨부경로"  TEXT,   -- JSON (TEXT)
                "상태"      TEXT,
                "이름"      TEXT,
                "주소"      TEXT
            );""")
            con.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS "처리결과" (
                "결과번호"  INTEGER PRIMARY KEY AUTOINCREMENT,
                "민원번호"  INTEGER,
                "처리일시"  TEXT,
                "키워드"    TEXT,   -- JSON (TEXT)
                "의도"      TEXT,   -- JSON (TEXT)
                "부서"      TEXT,
                "세부분야"  TEXT,
                "긴급도"    TEXT,
                "감정"      TEXT,
                "모델버전"  TEXT,
                "기타"      TEXT,   -- JSON (TEXT)
                FOREIGN KEY("민원번호") REFERENCES "민원"("번호")
            );""")
            con.exec_driver_sql("""CREATE INDEX IF NOT EXISTS "ix_결과_민원번호" ON "처리결과"("민원번호");""")
            con.exec_driver_sql("PRAGMA journal_mode=WAL;")
            con.exec_driver_sql("PRAGMA foreign_keys=ON;")
        else:
            con.exec_driver_sql("""CREATE EXTENSION IF NOT EXISTS pg_trgm;""")
            con.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS "민원" (
                "번호"       BIGSERIAL PRIMARY KEY,
                "접수일시"   TIMESTAMPTZ,
                "접수경로"   TEXT,
                "연락처"     TEXT,
                "내용"       TEXT,
                "첨부경로"   JSONB,
                "상태"       TEXT,
                "이름"       TEXT,
                "주소"       TEXT
            );""")
            con.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS "처리결과" (
                "결과번호"  BIGSERIAL PRIMARY KEY,
                "민원번호"  BIGINT REFERENCES "민원"("번호") ON DELETE CASCADE,
                "처리일시"  TIMESTAMPTZ,
                "키워드"    JSONB,
                "의도"      JSONB,
                "부서"      TEXT,
                "세부분야"  TEXT,
                "긴급도"    TEXT,
                "감정"      TEXT,
                "모델버전"  TEXT,
                "기타"      JSONB
            );""")
            con.exec_driver_sql("""CREATE INDEX IF NOT EXISTS "ix_결과_민원번호" ON "처리결과"("민원번호");""")
            con.exec_driver_sql("""CREATE INDEX IF NOT EXISTS "idx_민원_내용_trgm" ON "민원" USING gin ("내용" gin_trgm_ops);""")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _json_out_for_db(value: Any) -> str:
    base = [] if isinstance(value, list) else {}
    return json.dumps(value if value not in (None, "") else base, ensure_ascii=False)

def _pretty_json(val: Any) -> str:
    if val in (None, "", {}):
        return ""
    try:
        obj = json.loads(val) if isinstance(val, str) else val
        if isinstance(obj, list):
            return ", ".join(str(x) for x in obj)
        if isinstance(obj, dict):
            return ", ".join(f"{k}: {v}" for k, v in obj.items())
        return str(obj)
    except Exception:
        return str(val)

def _mk_in_clause(prefix: str, values: List[str], column_sql: str, negate: bool = False):
    names = [f":{prefix}{i}" for i in range(len(values))]
    clause = f'{column_sql} {"NOT IN" if negate else "IN"} ({",".join(names)})'
    params = {f"{prefix}{i}": v for i, v in enumerate(values)}
    return clause, params


# -------------------------------------------------------------------
# Write / update APIs
# -------------------------------------------------------------------
def 민원_등록(
    접수경로: str,
    연락처: Optional[str],
    내용: str,
    첨부경로목록: Optional[List[str]],
    이름: Optional[str] = None,
    주소: Optional[str] = None,
) -> int:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    첨부값 = _json_out_for_db(첨부경로목록 or [])
    with get_conn() as con:
        if _is_sqlite():
            con.execute(sql("""
                INSERT INTO "민원" ("접수일시","접수경로","이름","연락처","주소","내용","첨부경로","상태")
                VALUES (:ts,:src,:nm,:phone,:addr,:txt,:files,'접수됨')
            """), dict(ts=ts, src=접수경로, nm=이름, phone=연락처, addr=주소, txt=내용, files=첨부값))
            new_id = con.exec_driver_sql("SELECT last_insert_rowid()").scalar_one()
            return int(new_id)
        else:
            res = con.execute(sql("""
                INSERT INTO "민원" ("접수일시","접수경로","이름","연락처","주소","내용","첨부경로","상태")
                VALUES (:ts,:src,:nm,:phone,:addr,:txt,(:files)::jsonb,'접수됨')
                RETURNING "번호"
            """), dict(ts=ts, src=접수경로, nm=이름, phone=연락처, addr=주소, txt=내용, files=첨부값))
            return int(res.scalar_one())

def 상태_변경(민원번호: int, 상태: str) -> None:
    with get_conn() as con:
        con.execute(sql('UPDATE "민원" SET "상태"=:s WHERE "번호"=:id'), dict(s=상태, id=민원번호))

def 결과_등록(
    민원번호: int,
    키워드: Optional[List[str]],
    의도: Optional[Dict[str, Any]],
    부서: Optional[str],
    세부분야: Optional[str],
    긴급도: Optional[str],
    감정: Optional[str],
    모델버전: Optional[str],
    기타: Optional[Dict[str, Any]] = None,
) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    kw_val  = _json_out_for_db(키워드 or [])
    in_val  = _json_out_for_db(의도 or {})
    # Safely enrich router fields into 기타
    기타 = 기타 or {}
    if "router" in 기타:
        기타["상위부서Top2"] = 기타["router"].get("상위부서Top2", [])
        기타["상위부서_후보TopK"] = 기타["router"].get("상위부서_후보TopK", [])
        기타["부서_후보TopK"] = 기타["router"].get("부서_후보TopK", [])
        기타["공통확인_사유"] = 기타["router"].get("공통확인_사유", "")
    
    etc_val = _json_out_for_db(기타)
    with get_conn() as con:
        if _is_sqlite():
            con.execute(sql("""
                INSERT INTO "처리결과"
                ("민원번호","처리일시","키워드","의도","부서","세부분야","긴급도","감정","모델버전","기타")
                VALUES (:id,:ts,:kw,:intent,:dept,:sub,:urg,:emo,:ver,:extra)
            """), dict(id=민원번호, ts=ts, kw=kw_val, intent=in_val, dept=부서, sub=세부분야,
                       urg=긴급도, emo=감정, ver=모델버전, extra=etc_val))
        else:
            con.execute(sql("""
                INSERT INTO "처리결과"
                ("민원번호","처리일시","키워드","의도","부서","세부분야","긴급도","감정","모델버전","기타")
                VALUES (:id,:ts,(:kw)::jsonb,(:intent)::jsonb,:dept,:sub,:urg,:emo,:ver,(:extra)::jsonb)
            """), dict(id=민원번호, ts=ts, kw=kw_val, intent=in_val, dept=부서, sub=세부분야,
                       urg=긴급도, emo=감정, ver=모델버전, extra=etc_val))


# -------------------------------------------------------------------
# Read APIs
# -------------------------------------------------------------------
def 대시보드_조회(
    접수경로_제외: Optional[List[str]] = None,
    모델버전_포함: Optional[List[str]] = None,
    모델버전_제외: Optional[List[str]] = None,
) -> pd.DataFrame:
    where_sql_parts: List[str] = []
    params: Dict[str, Any] = {}

    if 접수경로_제외:
        c, p = _mk_in_clause("src", 접수경로_제외, 'm."접수경로"', negate=True)
        where_sql_parts.append(c); params.update(p)

    if 모델버전_포함:
        c, p = _mk_in_clause("ver_in", 모델버전_포함, 'r."모델버전"', negate=False)
        where_sql_parts.append(c); params.update(p)

    if 모델버전_제외:
        c, p = _mk_in_clause("ver_out", 모델버전_제외, 'r."모델버전"', negate=True)
        where_sql_parts.append(c); params.update(p)

    where_sql = "WHERE " + " AND ".join(where_sql_parts) if where_sql_parts else ""

    q = sql(f"""
        SELECT
          m."번호"      AS "민원번호",
          m."접수일시"  AS "접수일시",
          m."접수경로"  AS "접수경로",
          m."이름"      AS "이름",
          m."연락처"    AS "연락처",
          m."주소"      AS "주소",
          m."내용"      AS "내용",
          m."상태"      AS "상태",
          r."처리일시"  AS "처리일시",
          r."키워드"    AS "키워드",
          r."의도"      AS "의도",
          r."부서"      AS "부서",
          r."세부분야"  AS "세부분야",
          r."긴급도"    AS "긴급도",
          r."감정"      AS "감정",
          r."모델버전"  AS "모델버전",
          r."기타"      AS "기타"
        FROM "민원" m
        LEFT JOIN "처리결과" r ON m."번호" = r."민원번호"
        {where_sql}
        ORDER BY m."번호" DESC
    """)

    with get_conn() as con:
        df = pd.read_sql_query(q, con, params=params)

    for col in ["키워드", "의도"]:
        if col in df.columns:
            df[col] = df[col].apply(_pretty_json)

    preferred = [
        "민원번호","이름","연락처","주소","내용",
        "부서","세부분야","긴급도","감정","상태",
        "접수일시","처리일시","키워드","의도","모델버전","접수경로",
        "기타",
    ]
    have = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in have]
    return df[have + rest]


def 처리결과_기타_조회(민원번호: int):
    with get_conn() as con:
        row = con.execute(sql("""
            SELECT "기타"
            FROM "처리결과"
            WHERE "민원번호" = :id
            ORDER BY "결과번호" DESC
            LIMIT 1
        """), {"id": 민원번호}).mappings().first()
    return row["기타"] if row else None


# -------------------------------------------------------------------
# Similar complaints (Postgres only)
# -------------------------------------------------------------------
def 유사_민원(내용: str, 부서: Optional[str] = None, topk: int = 10):
    if _is_sqlite():
        return []
    base = """
        SELECT m."번호", m."내용", m."접수일시", m."상태",
               similarity(m."내용", :q) AS sim
        FROM "민원" m
        LEFT JOIN "처리결과" r ON m."번호" = r."민원번호"
    """
    if 부서:
        base += ' WHERE r."부서" = :dept '
    base += ' ORDER BY m."내용" <-> :q LIMIT :k '
    with get_conn() as con:
        rows = con.execute(sql(base), {"q": 내용, "dept": 부서, "k": topk}).mappings().all()
    return rows
