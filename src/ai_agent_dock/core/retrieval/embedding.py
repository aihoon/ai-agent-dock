"""임베딩: 글을 벡터(숫자 목록)로 바꾼다. 뜻이 가까운 글은 벡터도 가깝다."""

from __future__ import annotations

import re
import zlib
from typing import Protocol

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"  # 구현 시점 기준. 실제 사용 전에 공식 문서로 다시 확인한다.


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI 임베딩 API를 쓴다. Anthropic은 임베딩 API가 없어서 검색은 항상 이쪽을 쓴다."""

    def __init__(self, model: str = EMBEDDING_MODEL):
        self._client = OpenAI()  # OPENAI_API_KEY는 SDK가 환경에서 읽는다.
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


class HashingEmbedder:
    """네트워크 없이 단어와 두 글자 조각으로 벡터를 만든다. 테스트와 워크스루 전용이다.

    뜻을 이해하지는 못하고, 글자가 많이 겹칠수록 가깝다. 실제 검색 품질은 OpenAIEmbedder로 확인한다.
    """

    def __init__(self, dims: int = 256):
        self._dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        for token in re.findall(r"[0-9A-Za-z가-힣]+", text.lower()):
            features = [token] + [token[i : i + 2] for i in range(len(token) - 1)]
            for feature in features:
                vec[zlib.crc32(feature.encode("utf-8")) % self._dims] += 1.0
        return vec
