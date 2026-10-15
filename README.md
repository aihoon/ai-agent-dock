# ai-agent-dock

범용 대화형 에이전트 플랫폼(AI Agent Dock)을 만드는 학습 프로젝트.
Tool 호출, 스킬, 세션 기반 대화를 갖춘 플랫폼을 만들고, 그 위에 스킬·플러그인을 설치하거나 코드 워크플로를 얹어 **애플리케이션**을 만든다.

## 현재 단계
C4 완료(Tool 레이어와 MCP). 다음은 C5(스킬). 태그: `c0`, `c1`, `c1.1`, `c2`, `c3`, `c4`.

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
```

### MCP 서버 연결(선택)

외부 MCP 서버의 Tool을 함께 쓰려면 설정 파일을 만든다. 설정하지 않으면 내장 Tool만 쓴다.

```bash
cp mcp.example.json mcp.json    # 연결할 서버를 고친다(git에 올리지 않는다)
mkdir sandbox                   # 예시의 Filesystem 서버가 접근을 허용하는 유일한 폴더
# .env에 MCP_CONFIG=mcp.json 을 적거나, 실행할 때 --mcp-config mcp.json 을 준다
uv run python examples/chat.py  # /tools 로 연결된 Tool과 출처를 확인한다
```

예시 설정은 Time 서버(`uvx mcp-server-time`)와 Filesystem 서버(`npx @modelcontextprotocol/server-filesystem`)를 쓴다. uv와 Node(npx)가 필요하고, 처음 실행은 패키지를 내려받느라 오래 걸릴 수 있다.
stdio 서버는 내 계정 권한으로 실행되는 프로그램이다. 신뢰하는 서버만 적고, 파일을 쓰는 서버는 접근 폴더를 좁힌다.

Python 3.12, 패키지 관리는 [uv](https://docs.astral.sh/uv/)를 쓴다.

## 용어

- **턴(Turn, 문답)**: 사용자 입력 하나와 최종 답변 하나
- **스텝(Step)**: 한 턴 안에서 모델을 한 번 호출하는 루프 한 바퀴 (`max_steps`)
- **스트리밍(Streaming)**: 응답이 만들어지는 대로 글자 조각(델타)을 받는 방식. 반대는 일괄 응답(Batch)
- 그 밖의 용어는 Notion의 Terminology 페이지에 정리한다.

## 구조

- `src/ai_agent_dock/core/llm/`: 제공자 독립 LLM 호출(`call_llm`, 스트리밍은 `on_text_delta`)과 OpenAI·Anthropic 어댑터
- `src/ai_agent_dock/core/runtime/`: Agent Loop
- `src/ai_agent_dock/core/session/`: 대화 세션(턴 사이에 이력을 이어 간다)
- `src/ai_agent_dock/core/tools/`: 기본 제공 Tool(계산기, 현재 시각)과 `ToolRegistry`(이름 중복 방지, 인자 검증, 결과 길이 제한)
- `src/ai_agent_dock/core/tools/mcp/`: MCP 클라이언트(설정 파일의 stdio 서버에 접속해 그 서버의 Tool을 가져온다)
- `src/ai_agent_dock/env.py`: 환경 변수 읽기 도우미
- `examples/`: CLI 데모와 단계별 출력 데모
- `mcp.example.json`: MCP 서버 설정 예시(복사해서 `mcp.json`으로 쓴다)
- `tests/`: 테스트(`mcp_test_server.py`는 MCP 연결을 시험하는 테스트용 서버이며 플랫폼 기능이 아니다)

설계와 커리큘럼은 Notion의 "Claude 학습 로그"에 정리한다.
