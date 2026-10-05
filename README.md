# ai-agent-dock

공통 코어 위에 도메인 팩을 붙여 확장하는 AI 에이전트 시스템 (학습 프로젝트).  

## 현재 단계
C0 (코어 v0.0, 프로젝트 기반).

## 실행

```bash
uv sync
uv run python -c "import ai_agent_dock.core"   # 출력 없이 끝나면 정상
```

Python 3.12, 패키지 관리는 [uv](https://docs.astral.sh/uv/)를 쓴다.

## 구조

- `src/ai_agent_dock/core/`: 도메인 지식이 없는 공통 코어
- `src/ai_agent_dock/packs/<도메인>/`: 도메인 팩 (코어 검증 후 추가 예정)
- `tests/`: 테스트 (코드가 생기는 C1부터)

설계와 커리큘럼은 Notion의 "Claude 학습 로그"에 정리한다.
