# ai-agent-dock

범용 대화형 에이전트 플랫폼(AI Agent Dock)을 만드는 학습 프로젝트.
Tool 호출, 스킬, 세션 기반 대화를 갖춘 플랫폼을 만들고, 그 위에 스킬·플러그인을 설치하거나 코드 워크플로를 얹어 **애플리케이션**을 만든다.

## 현재 단계
C9 완료(컨텍스트와 메모리: 토큰 보기, /compact로 이력 줄이기, 세션 저장과 이어 하기, 장기 메모리, Anthropic 프롬프트 캐싱). 태그: `c0`, `c1`, `c1.1`, `c2`, `c3`, `c4`, `c5`, `c6`, `c7`, `c8`, `c9`(커밋 후 태그).

## 실행

```bash
uv sync
cp .env.example .env     # 값을 채운다(.env는 git에 올리지 않는다)
uv run pytest            # 테스트(네트워크 없이 실행)
uv run python examples/walkthrough.py --scenario error   # 루프 동작을 단계별로 본다(모델 없이)
uv run python examples/walkthrough.py                    # 실제 모델로 단계별 확인(.env 필요)
uv run python examples/chat.py                           # 대화형 CLI(.env 필요). 스트리밍 여부는 AGENT_STREAM 또는 --stream/--no-stream
uv run python examples/chat_walkthrough.py --scenario memory --stream   # 글자 조각이 도착하는 모습(모델 없이)
uv run python examples/chat_walkthrough.py --scenario mcp               # MCP 서버의 Tool을 부르는 모습(모델 없이, 서버는 실제로 띄움)
uv run python examples/chat_walkthrough.py --scenario skill             # 스킬 본문을 load_skill로 불러 따르는 모습(모델 없이)
uv run python examples/chat_walkthrough.py --scenario hitl              # 승인 후 실행과 거부 후 미실행을 보여 준다(모델 없이, 가짜 승인자)
```

### 채팅 명령

`examples/chat.py`에서 `/help`를 입력하면 명령 목록이 나온다. `/help`와 `/exit` 말고는 모두 `core/commands.py`에 있어서, 다른 응용(예: 브라우저)도 같은 명령을 쓴다.

| 명령 | 하는 일 |
|---|---|
| `/tokens` | 마지막 턴의 입력(캐시)·출력 토큰, 세션 누적, 지금 이력 크기(다음 호출에 보낼 토큰 수) |
| `/compact` | 최근 턴(`COMPACT_KEEP_TURNS`, 기본 4)을 뺀 오래된 턴의 큰 Tool 결과를 자르고 요약 하나로 바꾼다. 명령으로만 실행하며, 이력이 `COMPACT_WARN_TOKENS`(기본 100,000)를 넘으면 답 뒤에 안내만 한다 |
| `/sessions`, `/session 번호` | 저장된 세션 목록을 보고, 번호로 골라 이어 간다. 세션은 턴마다 `data/sessions/`에 자동 저장된다 |
| `/session-name 새이름`, `/session-new` | 지금 세션 이름 바꾸기(겹치면 거절), 새 세션 시작. 새 세션 이름은 "날짜 첫 질문 앞 20자"로 자동으로 붙는다 |
| `/memory`, `/memory-add 내용`, `/memory-delete 번호` | 장기 메모리(`data/memory.md`, 번호 목록) 보기, 더하기, 지우기(확인 후). 메모리는 시스템 프롬프트에 들어가 모든 세션이 쓴다 |
| `/reset`, `/history`, `/tools`, `/skills`, `/skill 이름` | 이력 비우기, 이력 보기, Tool 목록, 스킬 목록, 스킬 지침 넣기 |

- 컴팩션 요약 모델은 `COMPACT_LLM_PROVIDER`, `COMPACT_LLM_MODEL`로 따로 정한다. 비워 두면 대화 모델을 쓴다.
- Anthropic 요청은 시스템 프롬프트와 마지막 Tool 정의에 캐시 표시를 단다. OpenAI는 제공자가 자동으로 캐시한다. 캐시로 처리된 입력 토큰은 `/tokens`와 추적 리포트의 `(캐시)` 열에 나온다.

### 평가와 추적

```bash
uv run pytest                          # 가짜 모델 테스트만 실행한다. 실제 모델 평가는 건너뛴다(비용과 네트워크 없음)
uv run evals                           # 실제 모델로 평가 케이스를 실행한다(uv run pytest --live와 같다)
uv run evals --provider anthropic --repeat 3   # 제공자를 바꾸고, 케이스마다 3번씩 돌린다
uv run python scripts/trace_report.py          # 세션 목록(저장된 세션은 [이름]이 붙는다). show 세션이름 또는 세션번호 / total / compare 로 통계를 본다
```

- 평가 케이스는 `evals/cases.jsonl`에 있다(git에 올린다). 카테고리는 tool, docs, approval, clarify, skill, memory다. memory 케이스는 앞선 턴(`setup_turns`), 마지막 질문 전 컴팩션(`compact_before`), 평가용 장기 메모리(`memory_items`)를 쓴다.
- 판정은 규칙 기반이다. 기대한 Tool 호출, 금지 Tool, 답에 있어야 할 표현, 있으면 안 되는 표현, 출처 표기, 실행한 Tool의 오류를 본다. 목표는 100% 통과다.
- 결과는 `data/evals/실행번호.json`에 저장되고, 직전 실행에서 통과했는데 이번에 실패한 케이스를 회귀로 표시한다.
- 추적 기록은 `logs/sessions/세션번호.jsonl`에 세션마다 남는다. 질문, 답, Tool 인자와 결과, 입력·출력 토큰 수를 담는다. 제공자가 토큰 수를 주지 않으면 tiktoken으로 계산하고 출처 태그(`provider`, `tiktoken`)를 붙인다.
- 추적 기록은 `.env`의 `TRACE_ENABLED=false`로 끈다. 민감한 내용이 들어갈 수 있어 `logs/`는 git에서 제외한다.

### MCP 서버 연결(선택)

외부 MCP 서버의 Tool을 함께 쓰려면 설정 파일을 만든다. 설정하지 않으면 내장 Tool만 쓴다.

```bash
cp mcp.example.json mcp.json    # 연결할 서버를 고친다(git에 올리지 않는다)
mkdir -p data/sandbox           # 예시의 Filesystem 서버가 접근을 허용하는 유일한 폴더(data/는 git에서 제외)
# .env에 MCP_CONFIG=mcp.json 을 적거나, 실행할 때 --mcp-config mcp.json 을 준다
uv run python examples/chat.py  # /tools 로 연결된 Tool과 출처를 확인한다
```

`mcp.json`의 서버마다 `"trusted": true`를 적으면, 그 서버의 읽기 전용 툴(MCP의 `readOnlyHint`가 true인 툴)은 승인 없이 자동 실행합니다. `trusted`를 적지 않으면 기본값은 false이고, 그 서버의 모든 툴은 실행 전에 승인 메뉴를 거칩니다. 쓰기 툴은 신뢰하더라도 승인을 거칩니다.

서버를 지우지 않고 잠시 끄려면 `"enabled": false`를 적습니다. 끈 서버는 연결하지 않으며, 설정 항목은 파일에 그대로 남깁니다. 기본값은 true입니다.

승인 메뉴는 `1) 승인하고 실행`, `2) 거부`, `3) 다르게 요청(이유 입력)` 세 가지입니다. 거부된 툴은 같은 턴 안에서 다시 묻지 않고 자동으로 막히며, 다음 턴에는 다시 묻습니다. 모든 실행 시도와 결정은 `logs/audit.jsonl`에 한 줄씩 남습니다(인자는 20자까지).

예시 설정은 Time 서버(`uvx mcp-server-time`)와 Filesystem 서버(`npx @modelcontextprotocol/server-filesystem`)를 쓴다. uv와 Node(npx)가 필요하고, 처음 실행은 패키지를 내려받느라 오래 걸릴 수 있다.
stdio 서버는 내 계정 권한으로 실행되는 프로그램이다. 신뢰하는 서버만 적고, 파일을 쓰는 서버는 접근 폴더를 좁힌다.

### 문서 검색(RAG)

`knowledge/` 폴더에 `.md` 문서를 넣으면, 모델이 `search_docs` Tool로 문서를 찾아 답합니다.

```bash
uv run python scripts/index_knowledge.py   # 문서를 임베딩해서 data/index.sqlite에 저장(바뀐 문서만 다시 임베딩)
uv run python examples/chat.py             # 저장된 색인으로 검색하며 대화
```

- 임베딩은 OpenAI를 씁니다. 그래서 제공자를 Anthropic으로 골라도 `OPENAI_API_KEY`가 필요합니다.
- 답의 출처는 `[문서 제목, 위치]` 형식으로 붙습니다. 검색 결과에 없는 출처는 `※` 경고로 표시합니다.
- 관련 문서가 없으면 모른다고 답합니다. 유사도 시작값은 0.3(`DEFAULT_THRESHOLD`)입니다.
- `data/index.sqlite`는 문서에서 다시 만들 수 있으므로 git에서 제외합니다. 감사 로그는 `logs/audit.jsonl`에 쌓이며 `logs/`도 git에서 제외합니다.

Python 3.12, 패키지 관리는 [uv](https://docs.astral.sh/uv/)를 쓴다.

## 용어

- **턴(Turn, 문답)**: 사용자 입력 하나와 최종 답변 하나
- **스텝(Step)**: 한 턴 안에서 모델을 한 번 호출하는 루프 한 바퀴 (`max_steps`)
- **스트리밍(Streaming)**: 응답이 만들어지는 대로 글자 조각(델타)을 받는 방식. 반대는 일괄 응답(Batch)
- **스킬(Skill)**: 절차와 지침을 담은 `SKILL.md` 파일. 모델에게는 이름과 한 줄 설명만 보이고, 필요하면 `load_skill` Tool로 본문을 불러온다.
- **툴 루프 에이전트(Tool loop agent)**: 모델이 Tool을 요청하고 결과를 받아 다시 답하는 과정을 반복하는 에이전트(ReAct 방식). 지금 `run_agent`가 이것이다.
- 그 밖의 용어는 Notion의 Terminology 페이지에 정리한다.

## 구조

- `src/ai_agent_dock/core/llm/`: 제공자 독립 LLM 호출(`call_llm`, 스트리밍은 `on_text_delta`)과 OpenAI·Anthropic 어댑터
- `src/ai_agent_dock/core/runtime/`: Agent Loop(`run_agent`), 설정·결과 타입(`AgentSettings`, `AgentResult`), 시스템 프롬프트 조립(`prompt.py`의 `build_system_prompt`: 기본 지침, 장기 메모리, 스킬 목록, 출처 규칙 순서)
- `src/ai_agent_dock/core/hitl/`: 사람 개입. 승인 메뉴(`approval.py`), 되묻기 Tool(`clarify.py`), 감사 로그(`audit.py`)
- `src/ai_agent_dock/core/evals/`: 평가. 케이스 읽기와 규칙 판정(`score.py`), 실행과 결과 저장, 회귀 표시(`run.py`)
- `src/ai_agent_dock/core/observability/`: 관측. 세션별 추적 기록(`trace.py`)과 추적 리포트 계산(`report.py`)
- `src/ai_agent_dock/core/llm/tokens.py`: 토큰 수 채우기(제공자 값이 없으면 tiktoken으로 계산하고 출처 태그를 붙인다)
- `evals/`: 평가 케이스(`cases.jsonl`)
- `src/ai_agent_dock/core/session/`: 대화 세션(턴 사이에 이력을 이어 간다), 세션 저장(`store.py`, `data/sessions/`), 컴팩션(`compact.py`)
- `src/ai_agent_dock/core/memory/`: 장기 메모리(`data/memory.md`)
- `src/ai_agent_dock/core/commands.py`: 슬래시 명령(이름, 사용법, 처리). 응용은 입력을 넘기고 결과를 보여 주기만 한다
- `src/ai_agent_dock/core/skills/`: 스킬 로더. `skills/` 폴더의 `SKILL.md`를 읽고 머리말을 검사하며, 목록 문장과 `load_skill` Tool을 만든다
- `src/ai_agent_dock/core/tools/`: 기본 제공 Tool(계산기, 현재 시각)과 `ToolRegistry`(이름 중복 방지, 인자 검증, 결과 길이 제한)
- `src/ai_agent_dock/core/toolkit.py`: 빌트인 Tool 조립(`make_builtin_tools`). 기능 모듈(스킬, 문서 검색, 되묻기)의 Tool을 한 목록으로 모은다. 기능 모듈은 이 파일을 import하지 않는다(순환 방지, `tests/test_toolkit.py`가 확인)
- `src/ai_agent_dock/core/tools/mcp/`: MCP 클라이언트(설정 파일의 stdio 서버에 접속해 그 서버의 Tool을 가져온다)
- `src/ai_agent_dock/env.py`: 환경 변수 읽기 도우미
- `examples/`: CLI 데모와 단계별 출력 데모
- `mcp.example.json`: MCP 서버 설정 예시(복사해서 `mcp.json`으로 쓴다)
- `src/ai_agent_dock/core/retrieval/`: 문서 검색(청킹, 임베딩, 색인, `search_docs` Tool, 출처 검사)
- `knowledge/`: 검색할 문서(.md). `scripts/index_knowledge.py`로 색인합니다
- `scripts/`: 색인 갱신(`index_knowledge.py`), 추적 리포트(`trace_report.py`)
- `tests/`: 테스트(`mcp_test_server.py`는 MCP 연결을 시험하는 테스트용 서버이며 플랫폼 기능이 아니다)

설계와 커리큘럼은 Notion의 "AI Agent Dock" 페이지에 정리한다.
