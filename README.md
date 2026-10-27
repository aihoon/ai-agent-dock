# ai-agent-dock

범용 대화형 에이전트 플랫폼(AI Agent Dock)을 만드는 학습 프로젝트.
Tool 호출, 스킬, 세션 기반 대화를 갖춘 플랫폼을 만들고, 그 위에 스킬·플러그인을 설치하거나 코드 워크플로를 얹어 **애플리케이션**을 만든다.

## 현재 단계
C14 진행 중(Workflow - Routing: `routing` 워크플로에서 router 에이전트가 질문을 분류하고, 고른 경로의 에이전트가 자기 프롬프트, Tool, 모델로 답한다. 에이전트별 모델과 추적의 모델별 통계). 워크플로 패턴 트랙(C13~C19)의 두 번째 단계다. 태그: `c0`, `c1`, `c1.1`, `c2`, `c3`, `c4`, `c5`, `c6`, `c7`, `c8`, `c9`, `c10`, `c11`, `c12`, `c13`.

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
uv run python examples/chat_walkthrough.py --scenario workflow          # 고쳐 쓰기 → 승인 대기 → 새 연결로 재개(모델 없이)
uv run python examples/chat_walkthrough.py --scenario safety            # 지시문이 든 파일 읽기 → 경고 있는 승인 메뉴 → 거부(모델 없이)
uv run python examples/chat_walkthrough.py --scenario subagent          # main이 researcher에게 맡기고 답만 받는 모습, 스텝별 입력과 출력(모델 없이)
uv run python examples/chat_walkthrough.py --scenario chain             # prompt-chaining의 gate 통과와 gate 실패, 단계마다 받은 Tool과 입력·출력(모델 없이)
uv run python examples/chat_walkthrough.py --scenario routing           # routing의 분류, 경로마다 받은 프롬프트·Tool·모델, 분류 실패 → general(모델 없이)
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
| `/workflows`, `/workflow 이름 요청`, `/workflow-resume 번호` | 워크플로 목록과 승인 대기 실행 보기, 시작, 승인 대기 실행 이어 가기(아래 "워크플로") |
| `/agents` | 설치된 에이전트(이름, 설명, Tool 패턴)와 main이 맡길 수 있는 에이전트, 맡기기 깊이 상한(아래 "서브에이전트") |
| `/safety`, `/hooks` | 항상 켜진 Safety 검사와 안전 정책(`safety.json`) 내용, Hook 자리 3개와 설치된 사용자 Hook(아래 "Safety와 Hook") |
| `/reset`, `/history`, `/tools`, `/skills`, `/skill 이름` | 이력 비우기, 이력 보기, Tool 목록, 스킬 목록, 스킬 지침 넣기 |

- 컴팩션 요약 모델은 `COMPACT_LLM_PROVIDER`, `COMPACT_LLM_MODEL`로 따로 정한다. 비워 두면 대화 모델을 쓴다.
- 되묻기(`[질문]`)의 답 자리에 `/`로 시작하는 명령을 입력하면 답 대신 질문 취소로 처리한다. 명령은 답이 끝난 뒤 다시 입력한다.
- Anthropic 요청은 시스템 프롬프트와 마지막 Tool 정의에 캐시 표시를 단다. OpenAI는 제공자가 자동으로 캐시한다. 캐시로 처리된 입력 토큰은 `/tokens`와 추적 리포트의 `(캐시)` 열에 나온다.

### 워크플로(LangGraph)

에이전트는 다음 단계를 모델이 정하고, 워크플로는 코드(그래프)가 단계의 순서와 분기를 정한다. 에이전트는 워크플로의 한 노드로 쓰인다.

```
/workflow answer-check MCP 서버는 어떻게 연결해?
/workflow prompt-chaining 이 프로젝트 문서에서 에이전트 루프의 최대 스텝은 어떻게 정해진다고 설명해?
/workflow routing 3의 12제곱은 얼마야?
```

- `answer-check`: 답 작성(에이전트, 문서 검색) → 출처 검사(코드) → 실패하면 이유를 붙여 고쳐 쓰기(최대 `WORKFLOW_MAX_REVISIONS`, 기본 2번) → 사람 승인. 상한을 넘으면 "관련 문서에서 근거를 찾지 못했습니다"로 끝난다. 문서 색인이 없으면 시작하지 않는다. 이 워크플로 안에서는 되묻기(`ask_user`)를 쓰지 않는다.
- `prompt-chaining`(C13): 개요 작성(에이전트, `search_docs`) → gate(코드: 항목 2~5개, 항목마다 출처, 이번에 검색한 출처인지) → 본문 작성(에이전트, Tool 없음, 개요와 검색 결과만 받는다). gate가 실패하면 고쳐 쓰지 않고 `gate 실패`와 이유로 끝난다. 사람 승인은 없다. 두 단계 모두 `run_agent`를 거치고, 단계마다 다른 것은 사용자 프롬프트와 Tool이다. gate는 Tool이 아니라 그래프 노드라 모델이 건너뛸 수 없다.
- `routing`(C14): router 에이전트(`agents/router/AGENT.md`)가 질문을 분류하고(경로 목록은 코드가 사용자 프롬프트에 넣는다), 코드가 경로를 골라 그 경로의 에이전트가 답한다. 경로 표는 `workflows/routing.py`의 `ROUTES`다.

  | 경로 | 에이전트 | Tool | 모델 |
  |---|---|---|---|
  | chat | chat | 없음 | `openai:gpt-6-luna` |
  | docs | main | `search_docs` | 대화 모델 |
  | web | web | Tavily 웹 검색 | 대화 모델 |
  | utility | main | `calculate`, `get_current_time` | `openai:gpt-6-luna` |
  | files | main | `files__*`(쓰기는 승인) | 대화 모델 |
  | general | main 그대로 | main의 Tool 전부 | 대화 모델 |

  router의 답은 대소문자, 앞뒤 공백, 따옴표, 마침표만 정리한다. 그 밖의 답이나 호출 오류는 `▸ 분류 실패("…") → general`로 간다. 경로 에이전트에는 질문만 넘기고, 여러 일이 섞인 질문은 나누지 않고 general로 보낸다. 웹 서버가 꺼져 있어도 web 경로는 빼지 않는다(Tool 없이 "찾지 못했다"고 답한다).
- 진행 중에는 `▸ 답 작성`, `▸ 출처 검사: …`, `▸ 고쳐 쓰기 1/2`처럼 노드마다 한 줄씩 보인다.
- 승인 메뉴는 `1) 승인`, `2) 거부`, `3) 다르게 요청(이유 입력)`이다. 엔터만 치면 승인 대기로 남는다. 승인 대기 상태는 `data/checkpoints.sqlite`에 저장되어 프로그램을 껐다 켜도 `/workflows`에 보이고 `/workflow-resume 번호`로 이어 간다.
- 승인이나 근거 없음으로 끝나면 질문과 최종 답만 채팅 이력에 한 턴으로 붙는다. 초안과 고쳐 쓴 과정은 체크포인트와 추적 기록에만 남는다. 워크플로는 채팅의 앞선 대화를 보지 않는다.
- LangGraph는 `core/orchestration/`과 `src/ai_agent_dock/workflows/`에서만 import한다(`tests/test_orchestration.py`가 확인). 함께 설치되는 langsmith는 `LANGSMITH_TRACING=true`일 때만 외부로 보내므로 `.env.example`은 `false`로 둔다.

### 서브에이전트

에이전트는 `agents/이름/AGENT.md` 파일 하나로 정의한다. 머리말에 여섯 항목 `name`, `description`, `model`(`제공자:모델`, 예 `openai:gpt-6-luna`. 비우면 대화 모델), `tools`(Tool 이름 패턴, `!`는 제외, `[]`는 Tool 없음. 비울 수 없다), `delegate`(`true`여야 main이 맡길 수 있다. 비우면 `false`), `max_steps`(비우면 `AGENT_MAX_STEPS` 또는 `SUBAGENT_MAX_STEPS`)를 모두 적고, 본문은 그 에이전트의 기본 지침(시스템 프롬프트 맨 앞)이 된다. `agents/main/AGENT.md`가 사용자와 대화하는 main이고, 없으면 시작하지 않는다.

```
---
name: researcher
description: 프로젝트 문서와 파일을 찾아 출처와 함께 조사 결과를 돌려준다(웹 검색 Tool이 연결되면 웹도).
model:
tools: [search_docs, files__read_*, web__tavily_search, web__tavily_extract]
delegate: true
max_steps: 6
---
당신은 조사 담당 에이전트다. ...
```

- main은 시스템 프롬프트의 에이전트 목록을 보고, 검색을 여러 번 해야 하는 조사는 `delegate(agent, task)`로 맡긴다. 진행 중에는 `→ researcher에게 맡김`, 들여 쓴 진행 줄, `← researcher 완료`가 보인다.
- 하위 에이전트는 새 이력으로 task 한 줄과 자기 본문만 받는다. main의 이력, 장기 메모리, 스킬, 되묻기(`ask_user`)는 받지 않는다. 일이 끝나면 이력은 버리고 최종 답만 `<untrusted>`로 감싸 main에게 돌려준다(최대 `TOOL_RESULT_MAX_CHARS`).
- 하위 에이전트는 읽기 전용 Tool만 쓴다. 쓰기 Tool이 패턴에 걸리면 시작할 때 오류가 난다. 승인, Safety, Hook, 감사 로그는 main 것을 그대로 쓰고, 승인 메뉴에는 `[researcher]`가 붙는다.
- 맡길 수 있는 깊이는 `SUBAGENT_MAX_DEPTH`(기본 1, 최대 3)다. 1이면 main만 맡긴다. 자기 자신과 부모 줄의 에이전트는 부르지 못한다. 하위 에이전트의 스텝 상한은 `AGENT.md`의 `max_steps`, 없으면 `SUBAGENT_MAX_STEPS`(기본 6)다.
- 웹 검색(선택): `mcp.example.json`의 `web` 서버(Tavily)를 `mcp.json`에 옮기고 `"enabled": true`로 바꾼 뒤 `.env`에 `TAVILY_API_KEY`를 넣는다. main은 웹 Tool을 받지 않고(`!web__*`) researcher에게 맡겨야 쓴다.

### Safety와 Hook

**Safety**는 플랫폼이 항상 거는 고정 검사다. 끄는 설정은 없고, 안전 정책 파일로 더 엄격하게만 바꾼다. **Hook**은 사용자가 설치해 더하는 확장이다.

| 자리 | Safety(항상 켜짐) | Hook(`hooks.json`) |
|---|---|---|
| `before_tool`(실행 직전, 승인보다 먼저) | 안전 정책 검사: 막힌 Tool이나 허용 패턴 밖 경로는 `⛔`로 막는다 | 외부 명령이 `{"block": "이유"}`를 내면 막는다 |
| `after_tool`(실행 직후) | MCP와 `search_docs` 결과를 `<untrusted>` 경계로 감싸고, 지시문으로 보이는 문장이 있으면 경고를 붙인다 | `{"result": "새 결과"}`로 결과를 바꾼다 |
| `after_answer`(답을 이력에 넣기 전) | 답의 API 키를 `[KEY]`로 가린다 | `{"answer": "새 답"}`으로 답을 바꾼다 |

```bash
cp safety.example.json safety.json   # 안전 정책: Tool별 deny(차단) 또는 ask(승인)와 허용 경로 패턴. .env의 SAFETY_CONFIG
cp hooks.example.json hooks.json     # 사용자 Hook: 이벤트별 명령 목록. .env의 HOOKS_CONFIG, HOOK_TIMEOUT
```

- 순서: `before_tool`은 Safety → Hook, 나머지는 Hook → Safety. Hook이 결과나 답을 바꿔도 감싸기와 키 가리기가 빠지지 않는다.
- 지시문 의심 문장을 찾은 턴에는 승인 메뉴 위에 `⚠` 경고가 나온다. 차단하지는 않는다. 위험 작업은 승인과 안전 정책이 막는다.
- 안전 정책은 지금보다 느슨하게(`allow`) 바꿀 수 없고, 없는 Tool 이름을 적으면 시작할 때 오류가 난다.
- Hook 명령은 `hooks.json`이 있는 폴더에서 실행되고, 표준 입력으로 `{"event", "tool", "arguments", "result", "answer"}`를 받는다. 실패하거나 제한 시간을 넘기면 `before_tool`은 막고, 나머지는 원래 값을 둔 채 경고한다. 예시 `hooks/log_call.py`는 Tool 이름과 인자 이름을 `logs/hook_calls.jsonl`에 남긴다.
- 추적 기록과 감사 로그는 API 키, 이메일, 전화번호, 주민등록번호, 카드번호를 가린다. 세션 파일의 사용자 입력과 Tool 결과는 이어 하기를 위해 원문으로 둔다(`data/`는 git 제외).
- `safety.json`과 `hooks.json`은 git에 올리지 않는다. 파일이 없으면 경고를 보이고 그것 없이 시작한다.

### 평가와 추적

```bash
uv run pytest                          # 가짜 모델 테스트만 실행한다. 실제 모델 평가는 건너뛴다(비용과 네트워크 없음)
uv run evals                           # 실제 모델로 평가 케이스를 실행한다(uv run pytest --live와 같다)
uv run evals --provider anthropic --repeat 3   # 제공자를 바꾸고, 케이스마다 3번씩 돌린다
uv run python scripts/trace_report.py          # 세션 목록(저장된 세션은 [이름]이 붙는다). show 세션이름 또는 세션번호 / total / compare 로 통계를 본다
uv run python scripts/trace_report.py steps    # 가장 최근 세션을 스텝 순서대로: 프롬프트, 모델 출력, Tool 결과, 워크플로 노드. steps 세션번호로 세션을 고른다
```

- 평가 케이스는 `evals/cases.jsonl`에 있다(git에 올린다). 카테고리는 tool, docs, approval, clarify, skill, memory, workflow, safety, subagent, routing이다. routing 케이스는 고른 경로(`expect_route`)를 본다. subagent 케이스는 맡겨야 하는지(`expect_delegate`)를 보고, 하위 에이전트가 부른 Tool도 판정에 센다. 평가는 웹 서버(`web`)를 연결하지 않는다. safety 케이스는 실제로 실행되면 안 되는 Tool(`forbid_executed`)과 평가 전용 문서 색인(`"knowledge": "fixtures"`)을 쓴다. 평가 전용 자료는 `evals/fixtures/`에 있고, 평가할 때 `data/sandbox/eval/`로 복사하고 `data/evals/index.sqlite`로 색인한다. workflow 케이스는 승인 단계의 답(`workflow_decisions`), 마지막 상태(`expect_status`), 고쳐 쓴 횟수 상한(`expect_max_revisions`)을 쓴다. memory 케이스는 앞선 턴(`setup_turns`), 마지막 질문 전 컴팩션(`compact_before`), 평가용 장기 메모리(`memory_items`)를 쓴다.
- 판정은 규칙 기반이다. 기대한 Tool 호출, 금지 Tool, 답에 있어야 할 표현, 있으면 안 되는 표현, 출처 표기, 실행한 Tool의 오류를 본다. 목표는 100% 통과다.
- 결과는 `data/evals/실행번호.json`에 저장되고, 직전 실행에서 통과했는데 이번에 실패한 케이스를 회귀로 표시한다.
- 추적 기록은 `logs/sessions/세션번호.jsonl`에 세션마다 남는다. 질문, 답, Tool 인자와 결과, 입력·출력 토큰 수를 담는다. 제공자가 토큰 수를 주지 않으면 tiktoken으로 계산하고 출처 태그(`provider`, `tiktoken`)를 붙인다.
- 위임은 `delegate` 줄로 남고, 하위 에이전트의 줄에는 `agent`와 `parent`가 붙는다. `trace_report.py show`는 위임 횟수와 에이전트별 LLM 호출, 토큰을 보여 준다. `llm` 줄에는 `model`(`제공자:모델`)이 붙고, 모델이 둘 이상 쓰인 세션(예: routing)은 모델별 LLM 호출과 토큰도 보여 준다.
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

서버가 읽기 전용 힌트를 주지 않는 Tool은 서버 항목의 `"read_only": [Tool 이름]`으로 읽기 전용으로 지정한다. `env` 값에 `"${TAVILY_API_KEY}"`처럼 쓰면 `.env`의 값으로 바꾸고, 그 변수가 없으면 시작할 때 오류가 난다(끈 서버는 읽지 않는다). 키를 `mcp.json`에 직접 적지 않기 위해서다.

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
- 답의 출처는 `[문서 제목, 위치]` 형식으로 붙습니다. 첫 소제목 앞 내용은 위치가 `처음`입니다(C13). 검색 결과에 없는 출처는 `※` 경고로 표시합니다.
- 관련 문서가 없으면 모른다고 답합니다. 유사도 시작값은 0.3(`DEFAULT_THRESHOLD`)입니다.
- `data/index.sqlite`는 문서에서 다시 만들 수 있으므로 git에서 제외합니다. 감사 로그는 `logs/audit.jsonl`에 쌓이며 `logs/`도 git에서 제외합니다.

Python 3.12, 패키지 관리는 [uv](https://docs.astral.sh/uv/)를 쓴다.

## 용어

- **턴(Turn, 문답)**: 사용자 입력 하나와 최종 답변 하나
- **스텝(Step)**: 한 턴 안에서 모델을 한 번 호출하는 루프 한 바퀴 (`max_steps`)
- **스트리밍(Streaming)**: 응답이 만들어지는 대로 글자 조각(델타)을 받는 방식. 반대는 일괄 응답(Batch)
- **스킬(Skill)**: 절차와 지침을 담은 `SKILL.md` 파일. 모델에게는 이름과 한 줄 설명만 보이고, 필요하면 `load_skill` Tool로 본문을 불러온다.
- **서브에이전트(Subagent, 하위 에이전트)**: main이 `delegate` Tool로 일을 맡기는 에이전트. 새 이력과 줄인 Tool로 일하고 최종 답만 돌려준다.
- **Prompt chaining**: 일을 정해진 단계로 나눠 앞 단계의 출력을 다음 단계의 입력으로 넘기는 워크플로. 단계 사이에 코드 검사(**gate**)를 둔다.
- **툴 루프 에이전트(Tool loop agent)**: 모델이 Tool을 요청하고 결과를 받아 다시 답하는 과정을 반복하는 에이전트(ReAct 방식). 지금 `run_agent`가 이것이다.
- 그 밖의 용어는 Notion의 Terminology 페이지에 정리한다.

## 구조

- `src/ai_agent_dock/core/llm/`: 제공자 독립 LLM 호출(`call_llm`, 스트리밍은 `on_text_delta`)과 OpenAI·Anthropic 어댑터
- `src/ai_agent_dock/core/runtime/`: Agent Loop(`run_agent`), 설정·결과 타입(`AgentConfig`, `AgentResult`). 시스템 프롬프트 조립은 `core/agents/system_prompt.py`
- `src/ai_agent_dock/core/agents/`: 서브에이전트. `AGENT.md` 읽기와 Tool 패턴(`loader.py`), 시스템 프롬프트 조립(`system_prompt.py`), 에이전트 설정 만들기와 `delegate` Tool(`team.py`의 `AgentTeam.config_for`)
- `agents/`: 에이전트 정의(`main`, `researcher`, routing 경로용 `router`, `web`, `chat`의 `AGENT.md`)
- `src/ai_agent_dock/core/hitl/`: 사람 개입. 승인 메뉴(`approval.py`), 되묻기 Tool(`clarify.py`), 감사 로그(`audit.py`)
- `src/ai_agent_dock/core/evals/`: 평가. 케이스 읽기와 규칙 판정(`score.py`), 실행과 결과 저장, 회귀 표시(`run.py`)
- `src/ai_agent_dock/core/observability/`: 관측. 세션별 추적 기록(`trace.py`)과 추적 리포트 계산(`report.py`)
- `src/ai_agent_dock/core/llm/tokens.py`: 토큰 수 채우기(제공자 값이 없으면 tiktoken으로 계산하고 출처 태그를 붙인다)
- `src/ai_agent_dock/core/safety/`: Safety. 안전 정책(`policy.py`), 감싸기와 지시문 탐지, 키 가리기(`checks.py`), 가리기 패턴(`mask.py`). runtime은 이 폴더를 import하지 않고 Safety의 모양(`SafetyChecks`)만 안다
- `src/ai_agent_dock/core/hooks/`: Hook 엔진(`hooks.json` 읽기, 외부 명령 실행)
- `hooks/`, `hooks.example.json`, `safety.example.json`: 사용자 Hook 스크립트와 설정 예시(복사해서 `hooks.json`, `safety.json`으로 쓴다)
- `evals/`: 평가 케이스(`cases.jsonl`)와 평가 전용 자료(`fixtures/`)
- `src/ai_agent_dock/core/session/`: 대화 세션(턴 사이에 이력을 이어 간다), 세션 저장(`store.py`, `data/sessions/`), 컴팩션(`compact.py`)
- `src/ai_agent_dock/core/memory/`: 장기 메모리(`data/memory.md`)
- `src/ai_agent_dock/core/orchestration/`: 워크플로 실행 기반(`WorkflowRunner`: 시작, 재개, 승인 대기 찾기, 추적 기록). 체크포인트 저장 방식은 밖에서 받는다(채팅은 SQLite, 테스트와 평가는 메모리)
- `src/ai_agent_dock/workflows/`: 워크플로 정의(`answer_check.py`, `prompt_chaining.py`, `routing.py`). 애플리케이션이 자기 그래프를 정의하는 방식의 첫 예. core는 이 폴더를 import하지 않는다
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
