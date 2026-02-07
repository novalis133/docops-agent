"""DocOps Agent - Main entry point."""

import logging
from pathlib import Path

from .config import settings
from .ingestion import DocumentParser, EmbeddingGenerator, ElasticsearchIndexer, SectionChunker

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_indexer() -> ElasticsearchIndexer:
    """Create an Elasticsearch indexer from settings."""
    es_settings = settings.elasticsearch

    return ElasticsearchIndexer(
        host=es_settings.host,
        port=es_settings.port,
        scheme=es_settings.scheme,
        username=es_settings.username,
        password=es_settings.password,
        api_key=es_settings.api_key,
        cloud_id=es_settings.cloud_id,
        documents_index=es_settings.documents_index,
        chunks_index=es_settings.chunks_index,
        alerts_index=es_settings.alerts_index,
    )


def create_embedder() -> EmbeddingGenerator:
    """Create an embedding generator from settings."""
    emb_settings = settings.embedding
    grad_settings = settings.gradient

    return EmbeddingGenerator(
        provider=emb_settings.provider,
        model_name=emb_settings.model_name,
        batch_size=emb_settings.batch_size,
        gradient_api_key=grad_settings.api_key,
        gradient_workspace_id=grad_settings.workspace_id,
    )


def create_chunker() -> SectionChunker:
    """Create a section chunker from settings."""
    chunk_settings = settings.chunking

    return SectionChunker(
        max_chunk_size=chunk_settings.max_chunk_size,
        min_chunk_size=chunk_settings.min_chunk_size,
        overlap=chunk_settings.overlap,
        respect_sections=chunk_settings.respect_sections,
    )


def ingest_document(file_path: str | Path) -> dict:
    """Ingest a single document into the system."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Ingesting document: {path.name}")

    # Parse document
    parser = DocumentParser()
    document = parser.parse(path)
    logger.info(f"Parsed document with {len(document.sections)} sections")

    # Chunk document
    chunker = create_chunker()
    chunks = chunker.chunk_document(document)
    logger.info(f"Created {len(chunks)} chunks")

    # Generate embeddings
    embedder = create_embedder()
    embedded_chunks = embedder.embed_chunks(chunks)
    logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

    # Index into Elasticsearch
    indexer = create_indexer()
    result = indexer.index_document(document, embedded_chunks, refresh=True)
    logger.info(f"Indexed document: {result['document_id']}")

    return result


def search(
    query: str,
    top_k: int = 10,
    use_hybrid: bool = True,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> list[dict]:
    """Search for documents matching the query."""
    indexer = create_indexer()

    if use_hybrid:
        embedder = create_embedder()
        query_embedding = embedder.embed_text(query)

        return indexer.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )
    else:
        return indexer.search_documents(query=query, top_k=top_k)


def get_corpus_health() -> dict:
    """Get health metrics for the document corpus."""
    indexer = create_indexer()
    stats = indexer.get_corpus_stats()

    # Calculate health score
    open_alerts = stats["alert_counts"]["by_status"].get("open", 0)
    critical_alerts = stats["alert_counts"]["by_severity"].get("critical", 0)
    high_alerts = stats["alert_counts"]["by_severity"].get("high", 0)

    if critical_alerts > 0:
        health_status = "critical"
        health_color = "red"
    elif high_alerts > 0 or open_alerts > 5:
        health_status = "warning"
        health_color = "yellow"
    else:
        health_status = "healthy"
        health_color = "green"

    return {
        "status": health_status,
        "color": health_color,
        "document_count": stats["document_count"],
        "chunk_count": stats["chunk_count"],
        "open_alerts": open_alerts,
        "critical_alerts": critical_alerts,
        "high_alerts": high_alerts,
    }
