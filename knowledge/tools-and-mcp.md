# Tool과 MCP

## Tool 정의

Tool은 이름, 설명, 인자 스키마, 실행 함수로 이루어진다. 모델은 이름과 설명을 보고 Tool을 고르고, 인자는 JSON Schema에 맞춰 보낸다.
인자가 스키마에 맞지 않으면 실행하지 않고 오류를 모델에 알려 준다.

## MCP 서버 연결

mcp.json에 서버를 적어 두면 프로그램을 시작할 때 서버에 연결하고, 그 서버가 제공하는 Tool을 가져온다.
연결은 표준 입출력(stdio)으로만 한다. 원격 서버는 아직 지원하지 않는다.

## Tool 이름 규칙

MCP 서버의 Tool 이름 앞에는 서버 이름과 두 개의 밑줄이 붙는다. 예를 들어 files 서버의 read_text_file은 files__read_text_file이 된다.
