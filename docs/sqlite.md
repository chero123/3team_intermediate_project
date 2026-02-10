# SQLite 대화 로그

이 문서는 TextParsing 모드에서 사용하는 SQLite 대화 로그의 스키마와 CLI 실행/조회 방법을 정리한다.

## 스키마

저장 테이블: `chat_log`

```sql
CREATE TABLE IF NOT EXISTS chat_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  rating INTEGER,
  created_at TEXT NOT NULL
);
```

필드 설명:
- `id`: 테이블 내 고유 행 식별자 (자동 증가)
- `question`: 사용자 질문
- `answer`: 모델 답변
- `rating`: 피드백 (좋아요=1, 싫어요=-1, 기본 NULL)
- `created_at`: `YYYY-MM-DD HH:MM:SS`

## DB 파일 위치

- 기본 경로: `data/chat_log.sqlite`

## CLI 실행 방법

SQLite 셸 접속:

```bash
sqlite3 data/chat_log.sqlite
```

## 조회 예시

최근 로그 20건:

```sql
SELECT question, answer, rating, created_at
FROM chat_log
ORDER BY id DESC
LIMIT 20;
```

피드백만 보기:

```sql
SELECT rating, created_at
FROM chat_log
WHERE rating IS NOT NULL
ORDER BY id DESC;
```

## 종료

SQLite 셸 종료:

```
.exit
```
