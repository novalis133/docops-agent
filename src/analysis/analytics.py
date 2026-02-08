"""Advanced analytics using Elasticsearch aggregations.

Provides deep corpus insights through:
- Time-series analysis of conflicts
- Staleness distribution
- Document type health metrics
- Trend detection
- Significant terms analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesDataPoint:
    """A single data point in a time series."""
    timestamp: datetime
    value: float
    label: str = ""


@dataclass
class CorpusAnalytics:
    """Comprehensive corpus analytics results."""
    generated_at: datetime
    document_count: int
    chunk_count: int
    alert_count: int

    # Staleness metrics
    staleness_distribution: dict[str, int] = field(default_factory=dict)
    avg_document_age_days: float = 0.0

    # Conflict metrics
    conflicts_by_type: dict[str, int] = field(default_factory=dict)
    conflicts_over_time: list[TimeSeriesDataPoint] = field(default_factory=list)

    # Document health by type
    doc_type_health: dict[str, dict] = field(default_factory=dict)

    # Significant terms
    significant_conflict_terms: list[dict] = field(default_factory=list)

    # Complexity metrics
    complexity_distribution: dict[str, int] = field(default_factory=dict)
    avg_reading_time_minutes: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "alert_count": self.alert_count,
            "staleness": {
                "distribution": self.staleness_distribution,
                "avg_age_days": self.avg_document_age_days,
            },
            "conflicts": {
                "by_type": self.conflicts_by_type,
                "over_time": [
                    {"timestamp": dp.timestamp.isoformat(), "value": dp.value, "label": dp.label}
                    for dp in self.conflicts_over_time
                ],
            },
            "doc_type_health": self.doc_type_health,
            "significant_terms": self.significant_conflict_terms,
            "complexity": {
                "distribution": self.complexity_distribution,
                "avg_reading_time_minutes": self.avg_reading_time_minutes,
            }
        }


class CorpusAnalyticsEngine:
    """Generate advanced analytics using Elasticsearch aggregations."""

    # Runtime field definitions for dynamic calculations
    RUNTIME_FIELDS = {
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
                    } else if (doc.containsKey('raw_text') && doc['raw_text'].size() > 0) {
                        // Estimate from raw_text length
                        wordCount = (int) (doc['raw_text'].value.length() / 6);
                    }

                    if (wordCount > 5000) emit('very_high');
                    else if (wordCount > 2000) emit('high');
                    else if (wordCount > 500) emit('medium');
                    else emit('low');
                """
            }
        }
    }

    def __init__(
        self,
        es_client: Optional[Elasticsearch] = None,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        documents_index: str = "docops-documents",
        chunks_index: str = "docops-chunks",
        alerts_index: str = "docops-alerts",
    ):
        """Initialize the analytics engine."""
        if es_client:
            self.es = es_client
        else:
            self.es = Elasticsearch(f"{scheme}://{host}:{port}")

        self.documents_index = documents_index
        self.chunks_index = chunks_index
        self.alerts_index = alerts_index

    def get_corpus_analytics(self) -> CorpusAnalytics:
        """Generate comprehensive corpus analytics.

        Returns:
            CorpusAnalytics object with all metrics.
        """
        analytics = CorpusAnalytics(
            generated_at=datetime.utcnow(),
            document_count=0,
            chunk_count=0,
            alert_count=0,
        )

        # Get basic counts
        try:
            analytics.document_count = self._get_index_count(self.documents_index)
            analytics.chunk_count = self._get_index_count(self.chunks_index)
            analytics.alert_count = self._get_index_count(self.alerts_index)
        except Exception as e:
            logger.warning(f"Failed to get basic counts: {e}")

        # Get staleness distribution
        try:
            staleness_data = self._get_staleness_distribution()
            analytics.staleness_distribution = staleness_data.get("distribution", {})
            analytics.avg_document_age_days = staleness_data.get("avg_age", 0.0)
        except Exception as e:
            logger.warning(f"Failed to get staleness distribution: {e}")

        # Get conflicts over time
        try:
            analytics.conflicts_over_time = self._get_conflicts_over_time()
        except Exception as e:
            logger.warning(f"Failed to get conflicts over time: {e}")

        # Get significant terms
        try:
            analytics.significant_conflict_terms = self._get_significant_terms()
        except Exception as e:
            logger.warning(f"Failed to get significant terms: {e}")

        # Get complexity distribution
        try:
            complexity_data = self._get_complexity_distribution()
            analytics.complexity_distribution = complexity_data.get("distribution", {})
            analytics.avg_reading_time_minutes = complexity_data.get("avg_reading_time", 0.0)
        except Exception as e:
            logger.warning(f"Failed to get complexity distribution: {e}")

        # Get document type health
        try:
            analytics.doc_type_health = self._get_doc_type_health()
        except Exception as e:
            logger.warning(f"Failed to get doc type health: {e}")

        # Get conflicts by type
        try:
            analytics.conflicts_by_type = self._get_conflicts_by_type()
        except Exception as e:
            logger.warning(f"Failed to get conflicts by type: {e}")

        return analytics

    def _get_index_count(self, index: str) -> int:
        """Get document count for an index."""
        try:
            result = self.es.count(index=index)
            return result.get("count", 0)
        except Exception:
            return 0

    def _get_staleness_distribution(self) -> dict:
        """Get distribution of documents by staleness risk level."""
        query = {
            "size": 0,
            "runtime_mappings": self.RUNTIME_FIELDS,
            "aggs": {
                "staleness_distribution": {
                    "terms": {
                        "field": "staleness_risk",
                        "size": 10
                    }
                },
                "avg_age": {
                    "avg": {
                        "field": "days_since_indexed"
                    }
                }
            }
        }

        try:
            result = self.es.search(index=self.documents_index, body=query)
            aggs = result.get("aggregations", {})

            distribution = {}
            for bucket in aggs.get("staleness_distribution", {}).get("buckets", []):
                distribution[bucket["key"]] = bucket["doc_count"]

            avg_age = aggs.get("avg_age", {}).get("value", 0.0) or 0.0

            return {"distribution": distribution, "avg_age": avg_age}

        except Exception as e:
            logger.error(f"Staleness distribution query failed: {e}")
            return {"distribution": {}, "avg_age": 0.0}

    def _get_conflicts_over_time(self, weeks: int = 12) -> list[TimeSeriesDataPoint]:
        """Get conflict count trend over time using date histogram."""
        query = {
            "size": 0,
            "aggs": {
                "conflicts_over_time": {
                    "date_histogram": {
                        "field": "created_at",
                        "calendar_interval": "week",
                        "format": "yyyy-MM-dd",
                        "min_doc_count": 0,
                        "extended_bounds": {
                            "min": f"now-{weeks}w/w",
                            "max": "now/w"
                        }
                    }
                }
            }
        }

        try:
            result = self.es.search(index=self.alerts_index, body=query)
            buckets = result.get("aggregations", {}).get("conflicts_over_time", {}).get("buckets", [])

            data_points = []
            for bucket in buckets:
                data_points.append(TimeSeriesDataPoint(
                    timestamp=datetime.fromisoformat(bucket["key_as_string"]),
                    value=bucket["doc_count"],
                    label=bucket["key_as_string"]
                ))

            return data_points

        except Exception as e:
            logger.error(f"Conflicts over time query failed: {e}")
            return []

    def _get_significant_terms(self, min_doc_count: int = 2, size: int = 20) -> list[dict]:
        """Find significant terms that appear in documents with conflicts."""
        # First, get documents that have alerts
        query = {
            "size": 0,
            "aggs": {
                "documents_with_alerts": {
                    "terms": {
                        "field": "document_id",
                        "size": 100
                    }
                }
            }
        }

        try:
            alert_result = self.es.search(index=self.alerts_index, body=query)
            doc_ids = [
                bucket["key"]
                for bucket in alert_result.get("aggregations", {}).get("documents_with_alerts", {}).get("buckets", [])
            ]

            if not doc_ids:
                return []

            # Get significant terms from chunks of those documents
            sig_terms_query = {
                "size": 0,
                "query": {
                    "terms": {
                        "document_id": doc_ids
                    }
                },
                "aggs": {
                    "significant_terms": {
                        "significant_text": {
                            "field": "content",
                            "size": size,
                            "min_doc_count": min_doc_count
                        }
                    }
                }
            }

            result = self.es.search(index=self.chunks_index, body=sig_terms_query)
            buckets = result.get("aggregations", {}).get("significant_terms", {}).get("buckets", [])

            return [
                {
                    "term": bucket["key"],
                    "doc_count": bucket["doc_count"],
                    "score": bucket.get("score", 0),
                    "bg_count": bucket.get("bg_count", 0)
                }
                for bucket in buckets
            ]

        except Exception as e:
            logger.error(f"Significant terms query failed: {e}")
            return []

    def _get_complexity_distribution(self) -> dict:
        """Get distribution of document complexity."""
        query = {
            "size": 0,
            "runtime_mappings": self.RUNTIME_FIELDS,
            "aggs": {
                "complexity_distribution": {
                    "terms": {
                        "field": "content_complexity",
                        "size": 10
                    }
                },
                "avg_reading_time": {
                    "avg": {
                        "field": "reading_time_minutes"
                    }
                }
            }
        }

        try:
            result = self.es.search(index=self.documents_index, body=query)
            aggs = result.get("aggregations", {})

            distribution = {}
            for bucket in aggs.get("complexity_distribution", {}).get("buckets", []):
                distribution[bucket["key"]] = bucket["doc_count"]

            avg_reading = aggs.get("avg_reading_time", {}).get("value", 0.0) or 0.0

            return {"distribution": distribution, "avg_reading_time": avg_reading}

        except Exception as e:
            logger.error(f"Complexity distribution query failed: {e}")
            return {"distribution": {}, "avg_reading_time": 0.0}

    def _get_doc_type_health(self) -> dict[str, dict]:
        """Get health metrics broken down by document type."""
        query = {
            "size": 0,
            "runtime_mappings": self.RUNTIME_FIELDS,
            "aggs": {
                "by_file_type": {
                    "terms": {
                        "field": "file_type",
                        "size": 20
                    },
                    "aggs": {
                        "avg_age": {
                            "avg": {
                                "field": "days_since_indexed"
                            }
                        },
                        "staleness": {
                            "terms": {
                                "field": "staleness_risk"
                            }
                        },
                        "total_chunks": {
                            "sum": {
                                "field": "chunk_count"
                            }
                        }
                    }
                }
            }
        }

        try:
            result = self.es.search(index=self.documents_index, body=query)
            buckets = result.get("aggregations", {}).get("by_file_type", {}).get("buckets", [])

            health = {}
            for bucket in buckets:
                file_type = bucket["key"]

                staleness_counts = {}
                for sb in bucket.get("staleness", {}).get("buckets", []):
                    staleness_counts[sb["key"]] = sb["doc_count"]

                health[file_type] = {
                    "document_count": bucket["doc_count"],
                    "avg_age_days": bucket.get("avg_age", {}).get("value", 0) or 0,
                    "total_chunks": bucket.get("total_chunks", {}).get("value", 0) or 0,
                    "staleness_distribution": staleness_counts
                }

            return health

        except Exception as e:
            logger.error(f"Doc type health query failed: {e}")
            return {}

    def _get_conflicts_by_type(self) -> dict[str, int]:
        """Get alert counts grouped by type."""
        query = {
            "size": 0,
            "aggs": {
                "by_type": {
                    "terms": {
                        "field": "alert_type",
                        "size": 20
                    }
                }
            }
        }

        try:
            result = self.es.search(index=self.alerts_index, body=query)
            buckets = result.get("aggregations", {}).get("by_type", {}).get("buckets", [])

            return {bucket["key"]: bucket["doc_count"] for bucket in buckets}

        except Exception as e:
            logger.error(f"Conflicts by type query failed: {e}")
            return {}

    def get_trend_analysis(self, metric: str = "alerts", period_days: int = 30) -> dict:
        """Analyze trends in a specific metric.

        Args:
            metric: The metric to analyze (alerts, documents, chunks).
            period_days: Number of days to analyze.

        Returns:
            Trend analysis including direction and percentage change.
        """
        index_map = {
            "alerts": self.alerts_index,
            "documents": self.documents_index,
            "chunks": self.chunks_index
        }

        index = index_map.get(metric, self.alerts_index)
        date_field = "created_at" if metric == "alerts" else "indexed_at"

        # Get counts for current and previous period
        half_period = period_days // 2

        query = {
            "size": 0,
            "aggs": {
                "current_period": {
                    "filter": {
                        "range": {
                            date_field: {
                                "gte": f"now-{half_period}d"
                            }
                        }
                    }
                },
                "previous_period": {
                    "filter": {
                        "range": {
                            date_field: {
                                "gte": f"now-{period_days}d",
                                "lt": f"now-{half_period}d"
                            }
                        }
                    }
                }
            }
        }

        try:
            result = self.es.search(index=index, body=query)
            aggs = result.get("aggregations", {})

            current = aggs.get("current_period", {}).get("doc_count", 0)
            previous = aggs.get("previous_period", {}).get("doc_count", 0)

            if previous > 0:
                change_pct = ((current - previous) / previous) * 100
            else:
                change_pct = 100.0 if current > 0 else 0.0

            if change_pct > 10:
                trend = "increasing"
            elif change_pct < -10:
                trend = "decreasing"
            else:
                trend = "stable"

            return {
                "metric": metric,
                "period_days": period_days,
                "current_count": current,
                "previous_count": previous,
                "change_percent": round(change_pct, 1),
                "trend": trend
            }

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {
                "metric": metric,
                "error": str(e)
            }

    def get_hotspot_analysis(self, top_k: int = 10) -> list[dict]:
        """Identify document sections that are conflict hotspots.

        Args:
            top_k: Number of top hotspots to return.

        Returns:
            List of sections with high conflict density.
        """
        # Get sections that appear most frequently in alerts
        query = {
            "size": 0,
            "aggs": {
                "hotspots": {
                    "terms": {
                        "field": "document_id",
                        "size": top_k
                    },
                    "aggs": {
                        "severity_breakdown": {
                            "terms": {
                                "field": "severity"
                            }
                        },
                        "latest_alert": {
                            "max": {
                                "field": "created_at"
                            }
                        }
                    }
                }
            }
        }

        try:
            result = self.es.search(index=self.alerts_index, body=query)
            buckets = result.get("aggregations", {}).get("hotspots", {}).get("buckets", [])

            hotspots = []
            for bucket in buckets:
                severity = {}
                for sb in bucket.get("severity_breakdown", {}).get("buckets", []):
                    severity[sb["key"]] = sb["doc_count"]

                hotspots.append({
                    "document_id": bucket["key"],
                    "alert_count": bucket["doc_count"],
                    "severity_breakdown": severity,
                    "latest_alert": bucket.get("latest_alert", {}).get("value_as_string"),
                    "risk_score": self._calculate_risk_score(severity)
                })

            # Sort by risk score
            hotspots.sort(key=lambda x: x["risk_score"], reverse=True)
            return hotspots

        except Exception as e:
            logger.error(f"Hotspot analysis failed: {e}")
            return []

    def _calculate_risk_score(self, severity_counts: dict) -> float:
        """Calculate a risk score based on severity distribution."""
        weights = {
            "critical": 10,
            "high": 5,
            "medium": 2,
            "low": 1
        }

        score = 0
        for severity, count in severity_counts.items():
            score += weights.get(severity, 1) * count

        return score
