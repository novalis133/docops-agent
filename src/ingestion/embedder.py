"""Embedding generation for document chunks."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector."""

    chunk: Chunk
    embedding: list[float]
    model_name: str

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return len(self.embedding)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Optional[object] = None
        self._dimension: Optional[int] = None

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading sentence-transformer model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            # Get dimension from a test embedding
            test_embedding = self._model.encode(["test"])[0]
            self._dimension = len(test_embedding)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        self._load_model()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        self._load_model()
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name


class GradientProvider(EmbeddingProvider):
    """Embedding provider using DigitalOcean Gradient AI."""

    def __init__(
        self,
        api_key: str,
        workspace_id: str,
        model_name: str = "bge-large",
        base_url: str = "https://api.gradient.ai/api",
    ) -> None:
        self._api_key = api_key
        self._workspace_id = workspace_id
        self._model_name = model_name
        self._base_url = base_url
        self._dimension = 1024  # BGE-large default

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Gradient API."""
        import httpx

        url = f"{self._base_url}/embeddings"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Gradient-Workspace-Id": self._workspace_id,
            "Content-Type": "application/json",
        }

        embeddings: list[list[float]] = []

        # Process in batches
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            payload = {
                "inputs": [{"input": text} for text in batch],
                "model": self._model_name,
            }

            response = httpx.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            batch_embeddings = [item["embedding"] for item in result["embeddings"]]
            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name


class EmbeddingGenerator:
    """Generate embeddings for document chunks."""

    def __init__(
        self,
        provider: str = "sentence-transformers",
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        gradient_api_key: Optional[str] = None,
        gradient_workspace_id: Optional[str] = None,
    ) -> None:
        self.batch_size = batch_size
        self._provider = self._create_provider(
            provider=provider,
            model_name=model_name,
            gradient_api_key=gradient_api_key,
            gradient_workspace_id=gradient_workspace_id,
        )

    def _create_provider(
        self,
        provider: str,
        model_name: str,
        gradient_api_key: Optional[str],
        gradient_workspace_id: Optional[str],
    ) -> EmbeddingProvider:
        """Create the appropriate embedding provider."""
        if provider == "sentence-transformers":
            return SentenceTransformerProvider(model_name=model_name)
        elif provider == "gradient":
            if not gradient_api_key or not gradient_workspace_id:
                raise ValueError(
                    "Gradient API key and workspace ID required for gradient provider"
                )
            return GradientProvider(
                api_key=gradient_api_key,
                workspace_id=gradient_workspace_id,
                model_name=model_name,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Generate embeddings for a list of chunks."""
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        embeddings = self._embed_batch(texts)

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
                model_name=self._provider.model_name,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self._provider.embed_single(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return self._embed_batch(texts)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings in batches."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug(f"Embedding batch {i // self.batch_size + 1}")
            batch_embeddings = self._provider.embed(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._provider.dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._provider.model_name
