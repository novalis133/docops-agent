#!/usr/bin/env python3
"""Setup advanced Elasticsearch features for DocOps Agent.

This script configures:
1. Ingest pipeline for auto-enrichment
2. Runtime fields for dynamic staleness calculation
3. Elasticsearch Watcher for automated scanning
4. Index templates with advanced settings
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Ingest Pipeline Configuration
# =============================================================================

DOCOPS_ENRICHMENT_PIPELINE = {
    "description": "Auto-enrich documents with metadata, dates, and processing info",
    "processors": [
        # Add ingestion timestamp
        {
            "set": {
                "field": "ingested_at",
                "value": "{{_ingest.timestamp}}"
            }
        },
        # Extract dates from content for staleness detection
        {
            "script": {
                "description": "Extract dates from content",
                "source": """
                    if (ctx.containsKey('content') && ctx.content != null) {
                        def content = ctx.content.toLowerCase();
                        def dates = [];

                        // Match ISO dates (2024-01-15)
                        def isoPattern = /\\b(20\\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\\d|3[01])\\b/;
                        def matcher = isoPattern.matcher(content);
                        while (matcher.find()) {
                            dates.add(matcher.group());
                        }

                        // Match year references
                        def yearPattern = /\\b(20[12]\\d)\\b/;
                        matcher = yearPattern.matcher(content);
                        while (matcher.find()) {
                            def year = matcher.group();
                            if (!dates.contains(year)) {
                                dates.add(year);
                            }
                        }

                        ctx.extracted_dates = dates;
                        ctx.date_count = dates.size();
                    } else {
                        ctx.extracted_dates = [];
                        ctx.date_count = 0;
                    }
                """
            }
        },
        # Extract numeric requirements (for conflict detection hints)
        {
            "script": {
                "description": "Extract numeric requirements",
                "source": """
                    if (ctx.containsKey('content') && ctx.content != null) {
                        def content = ctx.content.toLowerCase();
                        def requirements = [];

                        // Match patterns like "12 characters", "30 days", "$500"
                        def patterns = [
                            /\\b(\\d+)\\s*(character|char|digit|day|hour|minute|week|month|year)s?\\b/,
                            /\\$\\s*(\\d+(?:,\\d{3})*(?:\\.\\d{2})?)\\b/,
                            /\\b(\\d+)\\s*%\\b/
                        ];

                        for (pattern in patterns) {
                            def matcher = pattern.matcher(content);
                            while (matcher.find()) {
                                requirements.add(matcher.group());
                            }
                        }

                        ctx.numeric_requirements = requirements;
                        ctx.requirement_count = requirements.size();
                    } else {
                        ctx.numeric_requirements = [];
                        ctx.requirement_count = 0;
                    }
                """
            }
        },
        # Calculate content metrics
        {
            "script": {
                "description": "Calculate content metrics",
                "source": """
                    if (ctx.containsKey('content') && ctx.content != null) {
                        ctx.word_count = ctx.content.split('\\\\s+').length;
                        ctx.char_count = ctx.content.length();

                        // Estimate reading time (200 words per minute)
                        ctx.reading_time_minutes = (int) Math.ceil(ctx.word_count / 200.0);
                    } else {
                        ctx.word_count = 0;
                        ctx.char_count = 0;
                        ctx.reading_time_minutes = 0;
                    }
                """
            }
        },
        # Add processing metadata
        {
            "set": {
                "field": "processing_version",
                "value": "1.0"
            }
        }
    ],
    "on_failure": [
        {
            "set": {
                "field": "pipeline_error",
                "value": "{{_ingest.on_failure_message}}"
            }
        }
    ]
}

# Chunks-specific pipeline
CHUNKS_ENRICHMENT_PIPELINE = {
    "description": "Enrich document chunks with additional metadata",
    "processors": [
        {
            "set": {
                "field": "indexed_at",
                "value": "{{_ingest.timestamp}}"
            }
        },
        # Calculate chunk complexity
        {
            "script": {
                "description": "Calculate chunk complexity metrics",
                "source": """
                    if (ctx.containsKey('content') && ctx.content != null) {
                        def content = ctx.content;
                        def words = content.split('\\\\s+');
                        ctx.word_count = words.length;

                        // Average word length as complexity indicator
                        def totalLen = 0;
                        for (word in words) {
                            totalLen += word.length();
                        }
                        ctx.avg_word_length = words.length > 0 ? (double) totalLen / words.length : 0;

                        // Count sentences (rough estimate)
                        ctx.sentence_count = content.split('[.!?]+').length;
                    }
                """
            }
        }
    ]
}


# =============================================================================
# Runtime Fields Configuration
# =============================================================================

DOCUMENTS_RUNTIME_FIELDS = {
    "days_since_indexed": {
        "type": "long",
        "script": {
            "source": """
                if (doc.containsKey('indexed_at') && doc['indexed_at'].size() > 0) {
                    long indexedMillis = doc['indexed_at'].value.toInstant().toEpochMilli();
                    long nowMillis = System.currentTimeMillis();
                    emit((nowMillis - indexedMillis) / (1000 * 60 * 60 * 24));
                } else {
                    emit(-1);
                }
            """
        }
    },
    "staleness_risk": {
        "type": "keyword",
        "script": {
            "source": """
                long days = -1;
                if (doc.containsKey('indexed_at') && doc['indexed_at'].size() > 0) {
                    long indexedMillis = doc['indexed_at'].value.toInstant().toEpochMilli();
                    long nowMillis = System.currentTimeMillis();
                    days = (nowMillis - indexedMillis) / (1000 * 60 * 60 * 24);
                }

                if (days < 0) emit('unknown');
                else if (days > 365) emit('critical');
                else if (days > 180) emit('high');
                else if (days > 90) emit('medium');
                else emit('low');
            """
        }
    },
    "content_complexity": {
        "type": "keyword",
        "script": {
            "source": """
                int wordCount = 0;
                if (doc.containsKey('word_count') && doc['word_count'].size() > 0) {
                    wordCount = (int) doc['word_count'].value;
                }

                if (wordCount > 5000) emit('very_high');
                else if (wordCount > 2000) emit('high');
                else if (wordCount > 500) emit('medium');
                else emit('low');
            """
        }
    },
    "has_numeric_requirements": {
        "type": "boolean",
        "script": {
            "source": """
                if (doc.containsKey('requirement_count') && doc['requirement_count'].size() > 0) {
                    emit(doc['requirement_count'].value > 0);
                } else {
                    emit(false);
                }
            """
        }
    }
}

CHUNKS_RUNTIME_FIELDS = {
    "chunk_size_category": {
        "type": "keyword",
        "script": {
            "source": """
                int charCount = 0;
                if (doc.containsKey('char_count') && doc['char_count'].size() > 0) {
                    charCount = (int) doc['char_count'].value;
                }

                if (charCount > 2000) emit('large');
                else if (charCount > 1000) emit('medium');
                else emit('small');
            """
        }
    }
}


# =============================================================================
# Watcher Configuration
# =============================================================================

NEW_DOCUMENT_SCAN_WATCHER = {
    "trigger": {
        "schedule": {
            "interval": "5m"
        }
    },
    "input": {
        "search": {
            "request": {
                "indices": ["docops-documents"],
                "body": {
                    "query": {
                        "range": {
                            "indexed_at": {
                                "gte": "now-5m"
                            }
                        }
                    }
                }
            }
        }
    },
    "condition": {
        "compare": {
            "ctx.payload.hits.total.value": {
                "gt": 0
            }
        }
    },
    "actions": {
        "log_new_documents": {
            "logging": {
                "text": "Detected {{ctx.payload.hits.total.value}} new documents. Triggering conflict scan."
            }
        },
        "trigger_scan": {
            "webhook": {
                "method": "POST",
                "host": "localhost",
                "port": 8000,
                "path": "/agent/workflow",
                "body": "{\"workflow\": \"conflict_scan\", \"parameters\": {\"scope\": \"recent\"}}",
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        }
    }
}

# Staleness check watcher - runs daily
STALENESS_CHECK_WATCHER = {
    "trigger": {
        "schedule": {
            "daily": {
                "at": "06:00"
            }
        }
    },
    "input": {
        "search": {
            "request": {
                "indices": ["docops-documents"],
                "body": {
                    "size": 0,
                    "runtime_mappings": DOCUMENTS_RUNTIME_FIELDS,
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"staleness_risk": "critical"}},
                                {"term": {"staleness_risk": "high"}}
                            ],
                            "minimum_should_match": 1
                        }
                    }
                }
            }
        }
    },
    "condition": {
        "compare": {
            "ctx.payload.hits.total.value": {
                "gt": 0
            }
        }
    },
    "actions": {
        "log_stale_documents": {
            "logging": {
                "text": "Found {{ctx.payload.hits.total.value}} stale documents requiring review."
            }
        },
        "trigger_staleness_audit": {
            "webhook": {
                "method": "POST",
                "host": "localhost",
                "port": 8000,
                "path": "/agent/workflow",
                "body": "{\"workflow\": \"staleness_audit\"}",
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        }
    }
}


# =============================================================================
# Setup Functions
# =============================================================================

def create_client(
    host: str = "localhost",
    port: int = 9200,
    username: str = None,
    password: str = None,
    api_key: str = None,
    cloud_id: str = None,
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


def setup_ingest_pipelines(es: Elasticsearch) -> dict:
    """Set up ingest pipelines for document enrichment."""
    results = {}

    # Documents enrichment pipeline
    try:
        es.ingest.put_pipeline(
            id="docops-documents-enrichment",
            body=DOCOPS_ENRICHMENT_PIPELINE
        )
        logger.info("Created ingest pipeline: docops-documents-enrichment")
        results["documents_pipeline"] = True
    except Exception as e:
        logger.error(f"Failed to create documents pipeline: {e}")
        results["documents_pipeline"] = False

    # Chunks enrichment pipeline
    try:
        es.ingest.put_pipeline(
            id="docops-chunks-enrichment",
            body=CHUNKS_ENRICHMENT_PIPELINE
        )
        logger.info("Created ingest pipeline: docops-chunks-enrichment")
        results["chunks_pipeline"] = True
    except Exception as e:
        logger.error(f"Failed to create chunks pipeline: {e}")
        results["chunks_pipeline"] = False

    return results


def setup_runtime_fields(es: Elasticsearch, documents_index: str, chunks_index: str) -> dict:
    """Add runtime fields to indices for dynamic calculations."""
    results = {}

    # Add runtime fields to documents index
    try:
        es.indices.put_mapping(
            index=documents_index,
            body={"runtime": DOCUMENTS_RUNTIME_FIELDS}
        )
        logger.info(f"Added runtime fields to {documents_index}")
        results["documents_runtime"] = True
    except Exception as e:
        logger.warning(f"Failed to add documents runtime fields (index may not exist): {e}")
        results["documents_runtime"] = False

    # Add runtime fields to chunks index
    try:
        es.indices.put_mapping(
            index=chunks_index,
            body={"runtime": CHUNKS_RUNTIME_FIELDS}
        )
        logger.info(f"Added runtime fields to {chunks_index}")
        results["chunks_runtime"] = True
    except Exception as e:
        logger.warning(f"Failed to add chunks runtime fields (index may not exist): {e}")
        results["chunks_runtime"] = False

    return results


def setup_watchers(es: Elasticsearch, api_host: str = "localhost", api_port: int = 8000) -> dict:
    """Set up Elasticsearch Watchers for automated monitoring."""
    results = {}

    # Update watcher configs with correct API host/port
    new_doc_watcher = NEW_DOCUMENT_SCAN_WATCHER.copy()
    new_doc_watcher["actions"]["trigger_scan"]["webhook"]["host"] = api_host
    new_doc_watcher["actions"]["trigger_scan"]["webhook"]["port"] = api_port

    staleness_watcher = STALENESS_CHECK_WATCHER.copy()
    staleness_watcher["actions"]["trigger_staleness_audit"]["webhook"]["host"] = api_host
    staleness_watcher["actions"]["trigger_staleness_audit"]["webhook"]["port"] = api_port

    # Create new document scan watcher
    try:
        es.watcher.put_watch(
            id="docops-new-document-scan",
            body=new_doc_watcher
        )
        logger.info("Created watcher: docops-new-document-scan")
        results["new_document_watcher"] = True
    except Exception as e:
        logger.warning(f"Failed to create new document watcher (Watcher may not be available): {e}")
        results["new_document_watcher"] = False

    # Create staleness check watcher
    try:
        es.watcher.put_watch(
            id="docops-staleness-check",
            body=staleness_watcher
        )
        logger.info("Created watcher: docops-staleness-check")
        results["staleness_watcher"] = True
    except Exception as e:
        logger.warning(f"Failed to create staleness watcher: {e}")
        results["staleness_watcher"] = False

    return results


def verify_setup(es: Elasticsearch) -> dict:
    """Verify that advanced features are properly configured."""
    results = {}

    # Check pipelines
    try:
        pipelines = es.ingest.get_pipeline()
        results["pipelines"] = [p for p in pipelines.keys() if p.startswith("docops")]
    except Exception:
        results["pipelines"] = []

    # Check watchers (if available)
    try:
        # Get watcher stats
        stats = es.watcher.stats()
        results["watcher_enabled"] = stats.get("stats", [{}])[0].get("watcher_state", "unknown")
    except Exception:
        results["watcher_enabled"] = "unavailable"

    # Get cluster info
    try:
        info = es.info()
        results["es_version"] = info["version"]["number"]
        results["cluster_name"] = info["cluster_name"]
    except Exception as e:
        results["error"] = str(e)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup advanced Elasticsearch features for DocOps Agent"
    )
    parser.add_argument("--host", default="localhost", help="Elasticsearch host")
    parser.add_argument("--port", type=int, default=9200, help="Elasticsearch port")
    parser.add_argument("--username", help="Elasticsearch username")
    parser.add_argument("--password", help="Elasticsearch password")
    parser.add_argument("--api-key", help="Elasticsearch API key")
    parser.add_argument("--cloud-id", help="Elastic Cloud ID")
    parser.add_argument("--api-host", default="localhost", help="DocOps API host for watchers")
    parser.add_argument("--api-port", type=int, default=8000, help="DocOps API port for watchers")
    parser.add_argument(
        "--documents-index", default="docops-documents", help="Documents index name"
    )
    parser.add_argument(
        "--chunks-index", default="docops-chunks", help="Chunks index name"
    )
    parser.add_argument(
        "--skip-watchers", action="store_true", help="Skip watcher setup"
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
    try:
        info = es.info()
        logger.info(f"Connected to Elasticsearch {info['version']['number']}")
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        sys.exit(1)

    # Setup ingest pipelines
    logger.info("\n=== Setting up Ingest Pipelines ===")
    pipeline_results = setup_ingest_pipelines(es)

    # Setup runtime fields
    logger.info("\n=== Setting up Runtime Fields ===")
    runtime_results = setup_runtime_fields(es, args.documents_index, args.chunks_index)

    # Setup watchers (optional)
    watcher_results = {}
    if not args.skip_watchers:
        logger.info("\n=== Setting up Watchers ===")
        watcher_results = setup_watchers(es, args.api_host, args.api_port)

    # Verify setup
    logger.info("\n=== Verifying Setup ===")
    verification = verify_setup(es)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 50)

    logger.info("\nIngest Pipelines:")
    for name, success in pipeline_results.items():
        status = "OK" if success else "FAILED"
        logger.info(f"  {name}: {status}")

    logger.info("\nRuntime Fields:")
    for name, success in runtime_results.items():
        status = "OK" if success else "FAILED/SKIPPED"
        logger.info(f"  {name}: {status}")

    if watcher_results:
        logger.info("\nWatchers:")
        for name, success in watcher_results.items():
            status = "OK" if success else "FAILED/UNAVAILABLE"
            logger.info(f"  {name}: {status}")

    logger.info(f"\nElasticsearch Version: {verification.get('es_version', 'unknown')}")
    logger.info(f"Cluster: {verification.get('cluster_name', 'unknown')}")
    logger.info(f"Watcher Status: {verification.get('watcher_enabled', 'unknown')}")

    if verification.get("pipelines"):
        logger.info(f"Active Pipelines: {', '.join(verification['pipelines'])}")


if __name__ == "__main__":
    main()
