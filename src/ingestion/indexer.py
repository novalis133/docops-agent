"""Elasticsearch indexer for documents and chunks."""

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from elasticsearch import Elasticsearch, helpers

from .chunker import Chunk
from .embedder import EmbeddedChunk
from .parser import ParsedDocument

logger = logging.getLogger(__name__)


class ElasticsearchIndexer:
    """Index documents and chunks into Elasticsearch."""

    def __init__(
        self,
        es_client: Optional[Elasticsearch] = None,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        cloud_id: Optional[str] = None,
        documents_index: str = "docops-documents",
        chunks_index: str = "docops-chunks",
        alerts_index: str = "docops-alerts",
    ) -> None:
        self.documents_index = documents_index
        self.chunks_index = chunks_index
        self.alerts_index = alerts_index

        if es_client:
            self.es = es_client
        else:
            self.es = self._create_client(
                host=host,
                port=port,
                scheme=scheme,
                username=username,
                password=password,
                api_key=api_key,
                cloud_id=cloud_id,
            )

    def _create_client(
        self,
        host: str,
        port: int,
        scheme: str,
        username: Optional[str],
        password: Optional[str],
        api_key: Optional[str],
        cloud_id: Optional[str],
    ) -> Elasticsearch:
        """Create an Elasticsearch client."""
        if cloud_id:
            if api_key:
                return Elasticsearch(cloud_id=cloud_id, api_key=api_key)
            elif username and password:
                return Elasticsearch(
                    cloud_id=cloud_id, basic_auth=(username, password)
                )
            else:
                return Elasticsearch(cloud_id=cloud_id)

        url = f"{scheme}://{host}:{port}"

        if api_key:
            return Elasticsearch(hosts=[url], api_key=api_key)
        elif username and password:
            return Elasticsearch(hosts=[url], basic_auth=(username, password))
        else:
            return Elasticsearch(hosts=[url])

    def index_document(
        self,
        document: ParsedDocument,
        embedded_chunks: list[EmbeddedChunk],
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Index a document and its chunks."""
        timestamp = datetime.now(timezone.utc).isoformat()

        # Generate document ID
        doc_id = self._generate_doc_id(document)

        # Index the document metadata
        doc_body = {
            "filename": document.filename,
            "file_type": document.file_type,
            "title": document.title,
            "raw_text": document.raw_text,
            "page_count": document.page_count,
            "section_count": len(document.sections),
            "chunk_count": len(embedded_chunks),
            "metadata": document.metadata,
            "indexed_at": timestamp,
            "updated_at": timestamp,
        }

        self.es.index(
            index=self.documents_index,
            id=doc_id,
            document=doc_body,
            refresh=refresh,
        )

        # Index chunks in bulk
        if embedded_chunks:
            self._bulk_index_chunks(doc_id, embedded_chunks, timestamp, refresh)

        logger.info(
            f"Indexed document '{document.title}' with {len(embedded_chunks)} chunks"
        )

        return {
            "document_id": doc_id,
            "chunk_count": len(embedded_chunks),
            "indexed_at": timestamp,
        }

    def _bulk_index_chunks(
        self,
        document_id: str,
        embedded_chunks: list[EmbeddedChunk],
        timestamp: str,
        refresh: bool,
    ) -> None:
        """Bulk index chunks with embeddings."""
        actions = []

        for ec in embedded_chunks:
            chunk = ec.chunk

            action = {
                "_index": self.chunks_index,
                "_id": chunk.id,
                "_source": {
                    "document_id": document_id,
                    "document_title": chunk.document_title,
                    "section_title": chunk.section_title,
                    "section_level": chunk.section_level,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks_in_section": chunk.total_chunks_in_section,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "page_number": chunk.page_number,
                    "char_count": chunk.char_count,
                    "embedding": ec.embedding,
                    "embedding_model": ec.model_name,
                    "metadata": chunk.metadata,
                    "indexed_at": timestamp,
                },
            }
            actions.append(action)

        success, errors = helpers.bulk(
            self.es,
            actions,
            refresh=refresh,
            raise_on_error=False,
        )

        if errors:
            logger.warning(f"Bulk indexing had {len(errors)} errors")
            for error in errors[:5]:
                logger.warning(f"  Error: {error}")

        logger.debug(f"Bulk indexed {success} chunks")

    def search_documents(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Search documents using BM25."""
        must_clauses: list[dict] = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "raw_text", "filename"],
                }
            }
        ]

        if filters:
            for field, value in filters.items():
                must_clauses.append({"term": {field: value}})

        response = self.es.search(
            index=self.documents_index,
            query={"bool": {"must": must_clauses}},
            size=top_k,
        )

        return [
            {
                "id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"],
            }
            for hit in response["hits"]["hits"]
        ]

    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Perform hybrid search combining BM25 and vector similarity."""
        # Build filter clauses
        filter_clauses = []
        if filters:
            for field, value in filters.items():
                filter_clauses.append({"term": {field: value}})

        # BM25 query
        bm25_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["content^2", "section_title", "document_title"],
                        }
                    }
                ],
                "filter": filter_clauses,
            }
        }

        # KNN query
        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k * 2,  # Get more candidates for re-ranking
            "num_candidates": top_k * 10,
        }

        if filter_clauses:
            knn_query["filter"] = {"bool": {"filter": filter_clauses}}

        # Execute hybrid search with RRF (Reciprocal Rank Fusion)
        response = self.es.search(
            index=self.chunks_index,
            query=bm25_query,
            knn=knn_query,
            size=top_k,
            # RRF is handled automatically when both query and knn are present
        )

        return [
            {
                "id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"],
            }
            for hit in response["hits"]["hits"]
        ]

    def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Perform pure vector similarity search."""
        knn_query = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": top_k * 10,
        }

        if filters:
            filter_clauses = [{"term": {k: v}} for k, v in filters.items()]
            knn_query["filter"] = {"bool": {"filter": filter_clauses}}

        response = self.es.search(
            index=self.chunks_index,
            knn=knn_query,
            size=top_k,
        )

        return [
            {
                "id": hit["_id"],
                "score": hit["_score"],
                **hit["_source"],
            }
            for hit in response["hits"]["hits"]
        ]

    def get_document(self, document_id: str) -> Optional[dict]:
        """Get a document by ID."""
        try:
            response = self.es.get(index=self.documents_index, id=document_id)
            return {"id": response["_id"], **response["_source"]}
        except Exception:
            return None

    def get_document_chunks(self, document_id: str) -> list[dict]:
        """Get all chunks for a document."""
        response = self.es.search(
            index=self.chunks_index,
            query={"term": {"document_id": document_id}},
            sort=[{"chunk_index": "asc"}],
            size=1000,
        )

        return [
            {"id": hit["_id"], **hit["_source"]} for hit in response["hits"]["hits"]
        ]

    def delete_document(self, document_id: str, refresh: bool = False) -> bool:
        """Delete a document and all its chunks."""
        try:
            # Delete chunks first
            self.es.delete_by_query(
                index=self.chunks_index,
                query={"term": {"document_id": document_id}},
                refresh=refresh,
            )

            # Delete document
            self.es.delete(
                index=self.documents_index,
                id=document_id,
                refresh=refresh,
            )

            logger.info(f"Deleted document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def create_alert(
        self,
        document_id: str,
        alert_type: str,
        severity: str,
        title: str,
        description: str,
        metadata: Optional[dict] = None,
        refresh: bool = False,
    ) -> str:
        """Create an alert for a document issue."""
        import uuid

        alert_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        alert_body = {
            "document_id": document_id,
            "alert_type": alert_type,
            "severity": severity,
            "title": title,
            "description": description,
            "status": "open",
            "metadata": metadata or {},
            "created_at": timestamp,
            "updated_at": timestamp,
        }

        self.es.index(
            index=self.alerts_index,
            id=alert_id,
            document=alert_body,
            refresh=refresh,
        )

        return alert_id

    def get_alerts(
        self,
        document_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: str = "open",
        top_k: int = 100,
    ) -> list[dict]:
        """Get alerts with optional filtering."""
        must_clauses = [{"term": {"status": status}}]

        if document_id:
            must_clauses.append({"term": {"document_id": document_id}})
        if severity:
            must_clauses.append({"term": {"severity": severity}})

        response = self.es.search(
            index=self.alerts_index,
            query={"bool": {"must": must_clauses}},
            sort=[{"created_at": "desc"}],
            size=top_k,
        )

        return [
            {"id": hit["_id"], **hit["_source"]} for hit in response["hits"]["hits"]
        ]

    def get_corpus_stats(self) -> dict:
        """Get statistics about the indexed corpus."""
        doc_count = self.es.count(index=self.documents_index)["count"]
        chunk_count = self.es.count(index=self.chunks_index)["count"]

        # Get alert counts by severity
        alert_aggs = self.es.search(
            index=self.alerts_index,
            size=0,
            aggs={
                "by_severity": {"terms": {"field": "severity"}},
                "by_status": {"terms": {"field": "status"}},
            },
        )

        severity_counts = {
            bucket["key"]: bucket["doc_count"]
            for bucket in alert_aggs["aggregations"]["by_severity"]["buckets"]
        }

        status_counts = {
            bucket["key"]: bucket["doc_count"]
            for bucket in alert_aggs["aggregations"]["by_status"]["buckets"]
        }

        return {
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "alert_counts": {
                "by_severity": severity_counts,
                "by_status": status_counts,
            },
        }

    def _generate_doc_id(self, document: ParsedDocument) -> str:
        """Generate a unique document ID."""
        import hashlib

        content = f"{document.filename}:{document.title}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
