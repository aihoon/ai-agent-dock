"""지식 색인 갱신: knowledge/의 .md 문서를 읽고 해시를 비교해서, 바뀐 문서만 임베딩해 index.sqlite에 저장한다.

사용법: uv run python scripts/index_knowledge.py
채팅(examples/chat.py)은 이 색인을 읽기만 한다. 문서를 넣거나 고쳤다면 이 스크립트를 먼저 다시 실행한다.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from ai_agent_dock.core.retrieval import KnowledgeIndex, OpenAIEmbedder

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"  # 색인할 문서 폴더(바로 아래의 .md 파일)
INDEX_PATH = ROOT / "index.sqlite"  # 색인 파일. 문서에서 다시 만들 수 있다.


def main() -> None:
    load_dotenv()  # OPENAI_API_KEY를 .env에서 읽는다.
    if not any(KNOWLEDGE_DIR.glob("*.md")):
        print(f"색인할 문서가 없습니다. {KNOWLEDGE_DIR}/ 에 .md 파일을 넣고 다시 실행하세요.")
        return

    store = KnowledgeIndex(INDEX_PATH, OpenAIEmbedder())
    try:
        report = store.refresh(KNOWLEDGE_DIR)
        files, chunks = store.stats()
    except Exception as e:
        raise SystemExit(f"색인 실패: {type(e).__name__}: {e}")
    finally:
        store.close()

    print(f"색인 완료: 추가 {report.added}, 갱신 {report.updated}, 삭제 {report.removed}, 그대로 {report.unchanged}")
    print(f"현재 색인: 문서 {files}개, 청크 {chunks}개 ({INDEX_PATH.name})")


if __name__ == "__main__":
    main()
