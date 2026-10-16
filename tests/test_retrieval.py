"""지식 검색 테스트: 청킹, 색인 갱신, 검색 임계값, 출처 검사, search_docs Tool. 네트워크는 쓰지 않는다."""

from ai_agent_dock.core.retrieval import (
    HashingEmbedder,
    KnowledgeIndex,
    check_citations,
    make_search_docs_tool,
    open_knowledge,
)
from ai_agent_dock.core.retrieval.chunking import MAX_CHARS, chunk_markdown
from ai_agent_dock.core.retrieval.index import DEFAULT_THRESHOLD


class CountingEmbedder(HashingEmbedder):
    """embed에 넘어간 글자 수를 세서, 다시 임베딩한 양을 확인한다."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return super().embed(texts)


def write_doc(root, name, text):
    (root / name).write_text(text, encoding="utf-8")


# --- 청킹 ---


def test_chunks_follow_headings_and_keep_section_names():
    chunks = chunk_markdown("앞말\n# 첫째\n가나다\n## 둘째\n라마바")

    assert [(c.section, c.text) for c in chunks] == [("", "앞말"), ("첫째", "가나다"), ("둘째", "라마바")]


def test_long_section_is_split_with_overlap():
    text = "가" * (MAX_CHARS * 2)

    chunks = chunk_markdown(f"# 긴 제목\n{text}")

    assert len(chunks) >= 2
    assert all(len(c.text) <= MAX_CHARS for c in chunks)
    assert chunks[0].text[-100:] == chunks[1].text[:100]  # 겹침


def test_empty_or_title_only_document_has_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("# 제목만 있다\n\n") == []


# --- 색인 갱신 ---


def test_refresh_adds_then_skips_unchanged_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_doc(docs, "a.md", "# 가\n내용 가")
    index = KnowledgeIndex(":memory:", HashingEmbedder())

    first = index.refresh(docs)
    second = index.refresh(docs)

    assert (first.added, first.unchanged) == (1, 0)
    assert (second.added, second.unchanged) == (0, 1)


def test_refresh_embeds_only_the_changed_file(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_doc(docs, "a.md", "# 가\n내용 가")
    write_doc(docs, "b.md", "# 나\n내용 나")
    embedder = CountingEmbedder()
    index = KnowledgeIndex(":memory:", embedder)
    index.refresh(docs)
    embedder.calls = 0

    write_doc(docs, "a.md", "# 가\n바뀐 내용")
    report = index.refresh(docs)

    assert (report.updated, report.unchanged) == (1, 1)
    assert embedder.calls == 1  # a.md의 청크 1개만 다시 임베딩했다


def test_refresh_removes_deleted_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_doc(docs, "a.md", "# 가\n내용 가")
    index = KnowledgeIndex(":memory:", HashingEmbedder())
    index.refresh(docs)

    (docs / "a.md").unlink()
    report = index.refresh(docs)

    assert report.removed == 1
    assert index.search("내용 가", threshold=0.0) == []


# --- 검색 ---


def test_search_orders_by_score_and_drops_low_scores(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_doc(docs, "tools.md", "# Tool 정의\n인자 스키마와 실행 함수로 이루어진다")
    write_doc(docs, "cooking.md", "# 요리\n김치찌개는 돼지고기로 만든다")
    index = KnowledgeIndex(":memory:", HashingEmbedder())
    index.refresh(docs)

    hits = index.search("Tool 인자 스키마", top_k=5)

    assert hits[0].cite == "Tool 정의"
    assert all(hit.score >= DEFAULT_THRESHOLD for hit in hits)
    assert all(hit.cite != "요리" for hit in hits)


def test_search_returns_nothing_when_no_chunk_is_close(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_doc(docs, "a.md", "# 가\n내용 가")
    index = KnowledgeIndex(":memory:", HashingEmbedder())
    index.refresh(docs)

    assert index.search("전혀 관계없는 질문 zzzz") == []


# --- 출처 검사 ---


def test_citation_check_flags_sources_not_in_this_turn():
    seen = {"Tool 정의, 인자"}

    check = check_citations("정답입니다 [Tool 정의, 인자]. 또 다른 근거 [스킬, 본문] [스킬, 본문]", seen)

    assert check.cited == ["Tool 정의, 인자", "스킬, 본문", "스킬, 본문"]
    assert check.unknown == ["스킬, 본문"]  # 중복은 한 번만 보고한다


def test_citation_check_passes_when_no_citation_is_written():
    check = check_citations("출처 없는 답", set())

    assert check.cited == [] and check.unknown == []


# --- search_docs Tool ---


def test_search_docs_tool_returns_hits_and_records_their_sources(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_doc(docs, "tools.md", "# Tool 정의\n인자 스키마와 실행 함수로 이루어진다")
    index = KnowledgeIndex(":memory:", HashingEmbedder())
    index.refresh(docs)
    seen = set()
    tool = make_search_docs_tool(index, seen)

    output = tool.func(query="Tool 인자 스키마")

    assert "[Tool 정의]" in output
    assert "Tool 정의" in seen


def test_search_docs_tool_says_nothing_found_and_records_nothing(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_doc(docs, "a.md", "# 가\n내용 가")
    index = KnowledgeIndex(":memory:", HashingEmbedder())
    index.refresh(docs)
    seen = set()
    tool = make_search_docs_tool(index, seen)

    output = tool.func(query="전혀 관계없는 질문 zzzz")

    assert output == "관련 문서를 찾지 못했다."
    assert seen == set()


# --- 설정 (채팅은 임베딩하지 않고 색인만 연다) ---


def test_open_is_none_without_index(tmp_path):
    assert open_knowledge(tmp_path / "없음.sqlite", HashingEmbedder()) is None


def test_open_is_none_when_index_is_empty(tmp_path):
    db = tmp_path / "i.sqlite"
    KnowledgeIndex(db, HashingEmbedder()).close()

    assert open_knowledge(db, HashingEmbedder()) is None


def test_open_does_not_embed_documents(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    write_doc(docs, "a.md", "# 가\n내용 가")
    db = tmp_path / "i.sqlite"
    indexer = KnowledgeIndex(db, HashingEmbedder())
    indexer.refresh(docs)
    indexer.close()
    embedder = CountingEmbedder()

    knowledge = open_knowledge(db, embedder)

    assert embedder.calls == 0  # 열 때는 문서를 임베딩하지 않는다
    assert (knowledge.files, knowledge.chunks) == (1, 1)
    assert knowledge.seen == set()
