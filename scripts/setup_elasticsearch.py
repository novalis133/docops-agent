#!/usr/bin/env python3
"""Setup Elasticsearch indices for DocOps Agent."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Index mappings
DOCUMENTS_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "content_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "filename": {"type": "keyword"},
            "file_type": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "content_analyzer",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "raw_text": {"type": "text", "analyzer": "content_analyzer"},
            "page_count": {"type": "integer"},
            "section_count": {"type": "integer"},
            "chunk_count": {"type": "integer"},
            "metadata": {"type": "object", "enabled": True},
            "indexed_at": {"type": "date"},
            "updated_at": {"type": "date"},
        }
    },
}


CHUNKS_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "content_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "document_id": {"type": "keyword"},
            "document_title": {
                "type": "text",
                "analyzer": "content_analyzer",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "section_title": {
                "type": "text",
                "analyzer": "content_analyzer",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "section_level": {"type": "integer"},
            "content": {"type": "text", "analyzer": "content_analyzer"},
            "chunk_index": {"type": "integer"},
            "total_chunks_in_section": {"type": "integer"},
            "start_char": {"type": "integer"},
            "end_char": {"type": "integer"},
            "page_number": {"type": "integer"},
            "char_count": {"type": "integer"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,  # Default for all-MiniLM-L6-v2
                "index": True,
                "similarity": "cosine",
            },
            "embedding_model": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": True},
            "indexed_at": {"type": "date"},
        }
    },
}


ALERTS_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "document_id": {"type": "keyword"},
            "alert_type": {"type": "keyword"},
            "severity": {"type": "keyword"},
            "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "description": {"type": "text"},
            "status": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": True},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
            "resolved_at": {"type": "date"},
            "resolved_by": {"type": "keyword"},
            # Resolution tracking fields
            "resolution_status": {"type": "keyword"},  # open, in_progress, resolved, wont_fix
            "resolution_notes": {"type": "text"},
            "verification_status": {"type": "keyword"},  # pending, verified, failed
            "verified_at": {"type": "date"},
            "verified_by": {"type": "keyword"},
            # Remediation suggestion
            "remediation": {
                "type": "object",
                "properties": {
                    "action": {"type": "keyword"},
                    "target_document": {"type": "keyword"},
                    "target_section": {"type": "text"},
                    "suggested_change": {"type": "text"},
                    "rationale": {"type": "text"},
                    "priority": {"type": "keyword"},
                    "estimated_effort": {"type": "keyword"},
                    "confidence": {"type": "float"},
                }
            },
        }
    },
}


def create_client(
    host: str = "localhost",
    port: int = 9200,
    username: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
    cloud_id: str | None = None,
) -> Elasticsearch:
    """Create Elasticsearch client."""
    if cloud_id:
        if api_key:
            return Elasticsearch(cloud_id=cloud_id, api_key=api_key)
        elif username and password:
            return Elasticsearch(cloud_id=cloud_id, basic_auth=(username, password))
        else:
            return Elasticsearch(cloud_id=cloud_id)

    url = f"http://{host}:{port}"

    if api_key:
        return Elasticsearch(hosts=[url], api_key=api_key)
    elif username and password:
        return Elasticsearch(hosts=[url], basic_auth=(username, password))
    else:
        return Elasticsearch(hosts=[url])


def create_index(
    es: Elasticsearch,
    index_name: str,
    mapping: dict,
    delete_existing: bool = False,
) -> bool:
    """Create an index with the given mapping."""
    if es.indices.exists(index=index_name):
        if delete_existing:
            logger.warning(f"Deleting existing index: {index_name}")
            es.indices.delete(index=index_name)
        else:
            logger.info(f"Index already exists: {index_name}")
            return False

    logger.info(f"Creating index: {index_name}")
    es.indices.create(index=index_name, body=mapping)
    logger.info(f"Created index: {index_name}")
    return True


def update_embedding_dimension(
    mapping: dict, dimension: int, similarity: str = "cosine"
) -> dict:
    """Update the embedding dimension in the mapping."""
    updated = mapping.copy()
    updated["mappings"]["properties"]["embedding"]["dims"] = dimension
    updated["mappings"]["properties"]["embedding"]["similarity"] = similarity
    return updated


def setup_indices(
    es: Elasticsearch,
    documents_index: str = "docops-documents",
    chunks_index: str = "docops-chunks",
    alerts_index: str = "docops-alerts",
    embedding_dimension: int = 384,
    delete_existing: bool = False,
) -> dict[str, bool]:
    """Set up all indices."""
    results = {}

    # Update chunks mapping with correct embedding dimension
    chunks_mapping = update_embedding_dimension(CHUNKS_INDEX_MAPPING, embedding_dimension)

    # Create indices
    results["documents"] = create_index(
        es, documents_index, DOCUMENTS_INDEX_MAPPING, delete_existing
    )
    results["chunks"] = create_index(es, chunks_index, chunks_mapping, delete_existing)
    results["alerts"] = create_index(
        es, alerts_index, ALERTS_INDEX_MAPPING, delete_existing
    )

    return results


def verify_connection(es: Elasticsearch) -> bool:
    """Verify connection to Elasticsearch."""
    try:
        info = es.info()
        logger.info(f"Connected to Elasticsearch {info['version']['number']}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Setup Elasticsearch indices for DocOps Agent")
    parser.add_argument("--host", default="localhost", help="Elasticsearch host")
    parser.add_argument("--port", type=int, default=9200, help="Elasticsearch port")
    parser.add_argument("--username", help="Elasticsearch username")
    parser.add_argument("--password", help="Elasticsearch password")
    parser.add_argument("--api-key", help="Elasticsearch API key")
    parser.add_argument("--cloud-id", help="Elastic Cloud ID")
    parser.add_argument(
        "--documents-index", default="docops-documents", help="Documents index name"
    )
    parser.add_argument(
        "--chunks-index", default="docops-chunks", help="Chunks index name"
    )
    parser.add_argument(
        "--alerts-index", default="docops-alerts", help="Alerts index name"
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=384,
        help="Embedding vector dimension",
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing indices before creating",
    )

    args = parser.parse_args()

    # Create client
    es = create_client(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        api_key=args.api_key,
        cloud_id=args.cloud_id,
    )

    # Verify connection
    if not verify_connection(es):
        sys.exit(1)

    # Setup indices
    results = setup_indices(
        es=es,
        documents_index=args.documents_index,
        chunks_index=args.chunks_index,
        alerts_index=args.alerts_index,
        embedding_dimension=args.embedding_dimension,
        delete_existing=args.delete_existing,
    )

    # Print summary
    logger.info("Setup complete:")
    for index_type, created in results.items():
        status = "created" if created else "already exists"
        logger.info(f"  {index_type}: {status}")


if __name__ == "__main__":
    main()
