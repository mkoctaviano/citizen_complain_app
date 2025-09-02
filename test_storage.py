from storage import init_db, 민원_등록, 결과_등록, 상태_변경, 대시보드_조회

def main():
    init_db()

    # 1) Insert a complaint
    cid = 민원_등록(
        접수경로="웹",
        연락처="demo@example.com",
        내용="우리 동네 도로에 포트홀이 생겨 위험합니다. 긴급 조치 바랍니다.",
        첨부경로목록=[]
    )
    print("Inserted complaint id:", cid)

    # 2) Insert results
    결과_등록(
        민원번호=cid,
        키워드=["도로","포트홀"],
        의도={"road_issue": 1.0},
        부서="도시관리",
        세부분야="도로관리과",
        긴급도="높음",
        감정="중립",
        모델버전="v0.1-demo",
        기타={"비고": "테스트"}
    )

    # 3) Update status
    상태_변경(cid, "처리완료")

    # 4) Fetch dashboard data
    df = 대시보드_조회()
    print(df.head())

if __name__ == "__main__":
    main()
