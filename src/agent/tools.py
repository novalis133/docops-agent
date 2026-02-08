"""Agent tools for document operations.

Each tool is a callable function that the agent can invoke during
multi-step reasoning. Tools follow the Elasticsearch Agent Builder pattern
with clear input/output schemas.
"""

from dataclasses import dataclass, asdict
from typing import Any, Optional
import json

from elasticsearch import Elasticsearch

from ..analysis import (
    ConflictDetector,
    StalenessChecker,
    GapAnalyzer,
    EntityExtractor,
    RemediationSuggester,
    SemanticConflictDetector,
)
from ..actions.alert_manager import AlertManager
from ..ingestion.indexer import ElasticsearchIndexer


@dataclass
class ToolResult:
    """Standard result format for all tools."""
    success: bool
    data: Any
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class AgentTools:
    """Collection of tools available to the DocOps agent.

    Each tool follows the pattern:
    - Clear input parameters
    - Structured JSON output
    - Error handling with meaningful messages
    """

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
        """Initialize agent tools with ES connection.

        Args:
            es_client: Existing ES client (optional).
            host: ES host.
            port: ES port.
            scheme: http or https.
            documents_index: Name of documents index.
            chunks_index: Name of chunks index.
            alerts_index: Name of alerts index.
        """
        if es_client:
            self.es = es_client
        else:
            self.es = Elasticsearch(f"{scheme}://{host}:{port}")

        self.host = host
        self.port = port
        self.scheme = scheme
        self.documents_index = documents_index
        self.chunks_index = chunks_index
        self.alerts_index = alerts_index

        # Initialize analysis components
        self.indexer = ElasticsearchIndexer(
            es_client=self.es,
            documents_index=documents_index,
            chunks_index=chunks_index,
            alerts_index=alerts_index,
        )
        self.conflict_detector = ConflictDetector(
            es_client=self.es,
            chunks_index=chunks_index,
            documents_index=documents_index,
        )
        self.staleness_checker = StalenessChecker(
            es_client=self.es,
            chunks_index=chunks_index,
            documents_index=documents_index,
        )
        self.gap_analyzer = GapAnalyzer(
            es_client=self.es,
            chunks_index=chunks_index,
            documents_index=documents_index,
        )
        self.entity_extractor = EntityExtractor()
        self.alert_manager = AlertManager(
            es_client=self.es,
            alerts_index=alerts_index,
        )
        self.remediation_suggester = RemediationSuggester()
        self.semantic_detector = SemanticConflictDetector(
            es_client=self.es,
            chunks_index=chunks_index,
        )

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions in Agent Builder format.

        Returns:
            List of tool definitions with name, description, and parameters.
        """
        return [
            {
                "name": "search_documents",
                "description": "Search across all documents using hybrid search (text + semantic). Returns relevant sections with citations. Use this to find information about a specific topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query - can be a question or keywords"
                        },
                        "doc_type": {
                            "type": "string",
                            "description": "Optional filter by document type (e.g., 'policy', 'handbook')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "compare_sections",
                "description": "Compare two document sections to identify conflicts, overlaps, or differences. Use this after finding relevant sections with search_documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id_a": {
                            "type": "string",
                            "description": "ID of the first chunk/section"
                        },
                        "chunk_id_b": {
                            "type": "string",
                            "description": "ID of the second chunk/section"
                        },
                        "comparison_type": {
                            "type": "string",
                            "enum": ["conflict", "overlap", "difference"],
                            "description": "Type of comparison to perform",
                            "default": "conflict"
                        }
                    },
                    "required": ["chunk_id_a", "chunk_id_b"]
                }
            },
            {
                "name": "run_consistency_check",
                "description": "Scan documents for internal inconsistencies and conflicts. Can check specific documents or the entire corpus.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document IDs to check (empty for full corpus)"
                        },
                        "focus_topic": {
                            "type": "string",
                            "description": "Optional topic to focus on (e.g., 'password', 'remote work')"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "generate_report",
                "description": "Generate a structured report from analysis findings. Use after running consistency checks or comparisons.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report_type": {
                            "type": "string",
                            "enum": ["compliance", "conflict", "summary", "gap", "staleness"],
                            "description": "Type of report to generate"
                        },
                        "scope": {
                            "type": "string",
                            "description": "Description of what to include in the report"
                        },
                        "include_recommendations": {
                            "type": "boolean",
                            "description": "Whether to include remediation recommendations",
                            "default": True
                        }
                    },
                    "required": ["report_type"]
                }
            },
            {
                "name": "create_alert",
                "description": "Create an alert for a discovered issue that needs attention. Use when you find conflicts, staleness, or compliance gaps.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low"],
                            "description": "Severity level of the alert"
                        },
                        "title": {
                            "type": "string",
                            "description": "Brief alert title"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the issue"
                        },
                        "affected_doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of documents affected by this issue"
                        }
                    },
                    "required": ["severity", "title", "description"]
                }
            },
            {
                "name": "get_document_health",
                "description": "Get health metrics for the document corpus: staleness, conflict count, coverage stats. Use this to understand the overall state of the documentation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_staleness": {
                            "type": "boolean",
                            "description": "Include staleness analysis",
                            "default": True
                        },
                        "include_gaps": {
                            "type": "boolean",
                            "description": "Include coverage gap analysis",
                            "default": True
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "verify_resolution",
                "description": "Verify that a resolved alert has been properly fixed. Re-runs the original detection logic to confirm the conflict/staleness/gap no longer exists.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "alert_id": {
                            "type": "string",
                            "description": "ID of the alert to verify"
                        }
                    },
                    "required": ["alert_id"]
                }
            },
            {
                "name": "get_remediation_suggestion",
                "description": "Get AI-powered remediation suggestion for an alert. Suggests which document to update and what changes to make.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "alert_id": {
                            "type": "string",
                            "description": "ID of the alert to get suggestion for"
                        }
                    },
                    "required": ["alert_id"]
                }
            },
            {
                "name": "detect_semantic_conflicts",
                "description": "Find SEMANTIC conflicts that require understanding document meaning, not just pattern matching. This is the most powerful conflict detection tool - it finds contradictions that simple keyword matching misses. Use this for complex policy analysis where implications matter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of specific document IDs to analyze. If empty, analyzes all documents."
                        },
                        "topic": {
                            "type": "string",
                            "description": "Optional focus area (e.g., 'remote work', 'data security', 'expenses')"
                        },
                        "deep_scan": {
                            "type": "boolean",
                            "description": "If true, analyze more section pairs (slower but more thorough). Default false.",
                            "default": False
                        }
                    },
                    "required": []
                }
            }
        ]

    def search_documents(
        self,
        query: str,
        doc_type: Optional[str] = None,
        top_k: int = 5
    ) -> ToolResult:
        """Search across all documents using hybrid search.

        Args:
            query: The search query.
            doc_type: Optional document type filter.
            top_k: Number of results to return.

        Returns:
            ToolResult with search results.
        """
        try:
            # Build the query
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"content": query}}
                        ]
                    }
                },
                "size": top_k,
                "_source": ["document_id", "document_title", "section_title", "content"]
            }

            if doc_type:
                search_body["query"]["bool"]["filter"] = [
                    {"match": {"document_title": doc_type}}
                ]

            result = self.es.search(index=self.chunks_index, body=search_body)

            hits = []
            for hit in result["hits"]["hits"]:
                source = hit["_source"]
                hits.append({
                    "chunk_id": hit["_id"],
                    "document_id": source.get("document_id", ""),
                    "document_title": source.get("document_title", ""),
                    "section_title": source.get("section_title", ""),
                    "content": source.get("content", "")[:500],  # Truncate for readability
                    "score": hit["_score"]
                })

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "total_hits": result["hits"]["total"]["value"],
                    "results": hits
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Search failed: {str(e)}"
            )

    def compare_sections(
        self,
        chunk_id_a: str,
        chunk_id_b: str,
        comparison_type: str = "conflict"
    ) -> ToolResult:
        """Compare two document sections.

        Args:
            chunk_id_a: ID of first chunk.
            chunk_id_b: ID of second chunk.
            comparison_type: Type of comparison (conflict, overlap, difference).

        Returns:
            ToolResult with comparison analysis.
        """
        try:
            # Fetch both chunks
            chunk_a = self.es.get(index=self.chunks_index, id=chunk_id_a)
            chunk_b = self.es.get(index=self.chunks_index, id=chunk_id_b)

            source_a = chunk_a["_source"]
            source_b = chunk_b["_source"]

            content_a = source_a.get("content", "")
            content_b = source_b.get("content", "")

            # Extract entities for comparison
            entities_a = self.entity_extractor.extract(content_a)
            entities_b = self.entity_extractor.extract(content_b)

            # Find differences
            comparison = {
                "chunk_a": {
                    "id": chunk_id_a,
                    "document": source_a.get("document_title", ""),
                    "section": source_a.get("section_title", ""),
                    "content": content_a[:300]
                },
                "chunk_b": {
                    "id": chunk_id_b,
                    "document": source_b.get("document_title", ""),
                    "section": source_b.get("section_title", ""),
                    "content": content_b[:300]
                },
                "comparison_type": comparison_type,
                "findings": []
            }

            # Compare numeric values
            numeric_a = {e.text: e.normalized_value for e in entities_a if e.normalized_value}
            numeric_b = {e.text: e.normalized_value for e in entities_b if e.normalized_value}

            for text_a, val_a in numeric_a.items():
                for text_b, val_b in numeric_b.items():
                    if val_a != val_b:
                        comparison["findings"].append({
                            "type": "numeric_difference",
                            "value_a": text_a,
                            "value_b": text_b,
                            "description": f"Different values: '{text_a}' vs '{text_b}'"
                        })

            if not comparison["findings"]:
                comparison["findings"].append({
                    "type": "no_conflicts",
                    "description": "No obvious conflicts detected between these sections"
                })

            return ToolResult(success=True, data=comparison)

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Comparison failed: {str(e)}"
            )

    def run_consistency_check(
        self,
        doc_ids: Optional[list[str]] = None,
        focus_topic: Optional[str] = None
    ) -> ToolResult:
        """Scan for inconsistencies and conflicts.

        Args:
            doc_ids: List of document IDs to check (empty for full corpus).
            focus_topic: Optional topic to focus on.

        Returns:
            ToolResult with inconsistencies found.
        """
        try:
            # Run conflict detection
            conflicts = self.conflict_detector.detect_conflicts(topic=focus_topic)

            # Filter by doc_ids if provided
            if doc_ids:
                conflicts = [
                    c for c in conflicts
                    if c.location_a.document_id in doc_ids or c.location_b.document_id in doc_ids
                ]

            inconsistencies = []
            for conflict in conflicts:
                inconsistencies.append({
                    "id": conflict.id,
                    "type": conflict.conflict_type.value,
                    "severity": conflict.severity.value,
                    "description": conflict.description,
                    "document_a": conflict.location_a.document_title,
                    "document_b": conflict.location_b.document_title,
                    "value_a": conflict.value_a,
                    "value_b": conflict.value_b,
                    "topic": conflict.topic
                })

            return ToolResult(
                success=True,
                data={
                    "total_inconsistencies": len(inconsistencies),
                    "focus_topic": focus_topic,
                    "inconsistencies": inconsistencies
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Consistency check failed: {str(e)}"
            )

    def generate_report(
        self,
        report_type: str,
        scope: Optional[str] = None,
        include_recommendations: bool = True
    ) -> ToolResult:
        """Generate a structured report.

        Args:
            report_type: Type of report (compliance, conflict, summary, gap, staleness).
            scope: Description of what to include.
            include_recommendations: Whether to include recommendations.

        Returns:
            ToolResult with generated report.
        """
        try:
            report = {
                "type": report_type,
                "scope": scope or "Full corpus analysis",
                "generated_at": str(__import__('datetime').datetime.utcnow()),
                "sections": []
            }

            if report_type == "conflict":
                conflicts = self.conflict_detector.detect_all_conflicts()
                report["sections"].append({
                    "title": "Conflict Summary",
                    "content": f"Found {len(conflicts)} conflicts across the document corpus."
                })

                # Group by severity
                by_severity = {}
                for c in conflicts:
                    sev = c.severity.value
                    if sev not in by_severity:
                        by_severity[sev] = []
                    by_severity[sev].append({
                        "description": c.description,
                        "documents": [c.location_a.document_title, c.location_b.document_title],
                        "values": [c.value_a, c.value_b]
                    })

                for severity in ["critical", "high", "medium", "low"]:
                    if severity in by_severity:
                        report["sections"].append({
                            "title": f"{severity.upper()} Severity Conflicts",
                            "items": by_severity[severity]
                        })

                if include_recommendations:
                    report["recommendations"] = [
                        "Review and resolve critical conflicts immediately",
                        "Establish document ownership to prevent future conflicts",
                        "Implement regular cross-document consistency reviews"
                    ]

            elif report_type == "staleness":
                issues = self.staleness_checker.check_all_documents()
                report["sections"].append({
                    "title": "Staleness Summary",
                    "content": f"Found {len(issues)} staleness issues."
                })

                for issue in issues:
                    report["sections"].append({
                        "title": issue.document_title,
                        "content": issue.description,
                        "severity": issue.severity.value,
                        "recommended_action": issue.recommended_action
                    })

                if include_recommendations:
                    report["recommendations"] = [
                        "Update or retire expired documents immediately",
                        "Establish document review cadence (quarterly recommended)",
                        "Add expiration dates to all policy documents"
                    ]

            elif report_type == "gap":
                gaps = self.gap_analyzer.analyze_all()
                report["sections"].append({
                    "title": "Coverage Gap Summary",
                    "content": f"Found {len(gaps)} coverage gaps."
                })

                for gap in gaps[:10]:  # Limit for readability
                    report["sections"].append({
                        "title": f"Missing: {gap.topic}",
                        "content": gap.description,
                        "covered_in": gap.covered_in,
                        "missing_from": gap.missing_from
                    })

                if include_recommendations:
                    report["recommendations"] = [
                        "Prioritize security-related coverage gaps",
                        "Ensure consistent topic coverage across related documents",
                        "Create a documentation coverage matrix"
                    ]

            elif report_type == "summary" or report_type == "compliance":
                # Comprehensive summary
                conflicts = self.conflict_detector.detect_all_conflicts()
                staleness = self.staleness_checker.check_all_documents()
                gaps = self.gap_analyzer.analyze_all()
                stats = self.indexer.get_corpus_stats()

                report["sections"] = [
                    {
                        "title": "Corpus Overview",
                        "content": f"Total documents: {stats.get('document_count', 0)}, Total chunks: {stats.get('chunk_count', 0)}"
                    },
                    {
                        "title": "Conflict Analysis",
                        "content": f"Found {len(conflicts)} conflicts. Critical: {len([c for c in conflicts if c.severity.value == 'critical'])}"
                    },
                    {
                        "title": "Staleness Analysis",
                        "content": f"Found {len(staleness)} staleness issues. Expired: {len([s for s in staleness if s.staleness_type.value == 'expired'])}"
                    },
                    {
                        "title": "Coverage Analysis",
                        "content": f"Found {len(gaps)} coverage gaps across documents."
                    }
                ]

                if include_recommendations:
                    report["recommendations"] = [
                        "Address critical conflicts immediately",
                        "Review and update stale documents",
                        "Close priority coverage gaps",
                        "Schedule quarterly documentation review"
                    ]

            return ToolResult(success=True, data=report)

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Report generation failed: {str(e)}"
            )

    def create_alert(
        self,
        severity: str,
        title: str,
        description: str,
        affected_doc_ids: Optional[list[str]] = None
    ) -> ToolResult:
        """Create an alert for an issue.

        Args:
            severity: Alert severity (critical, high, medium, low).
            title: Brief alert title.
            description: Detailed description.
            affected_doc_ids: IDs of affected documents.

        Returns:
            ToolResult with alert ID.
        """
        try:
            # Use the first affected doc ID or create a general alert
            doc_id = affected_doc_ids[0] if affected_doc_ids else "general"

            alert_id = self.indexer.create_alert(
                document_id=doc_id,
                alert_type="agent_generated",
                severity=severity,
                title=title,
                description=description,
                metadata={"affected_docs": affected_doc_ids or []}
            )

            return ToolResult(
                success=True,
                data={
                    "alert_id": alert_id,
                    "severity": severity,
                    "title": title,
                    "affected_documents": affected_doc_ids or []
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Alert creation failed: {str(e)}"
            )

    def get_document_health(
        self,
        include_staleness: bool = True,
        include_gaps: bool = True
    ) -> ToolResult:
        """Get health metrics for the document corpus.

        Args:
            include_staleness: Include staleness analysis.
            include_gaps: Include gap analysis.

        Returns:
            ToolResult with health metrics.
        """
        try:
            # Get basic stats
            stats = self.indexer.get_corpus_stats()

            health = {
                "document_count": stats.get("document_count", 0),
                "chunk_count": stats.get("chunk_count", 0),
                "open_alerts": stats.get("open_alerts", 0),
                "alerts_by_severity": stats.get("alerts_by_severity", {}),
            }

            # Run conflict detection
            conflicts = self.conflict_detector.detect_all_conflicts()
            health["total_conflicts"] = len(conflicts)
            health["critical_conflicts"] = len([c for c in conflicts if c.severity.value == "critical"])
            health["high_conflicts"] = len([c for c in conflicts if c.severity.value == "high"])

            if include_staleness:
                staleness = self.staleness_checker.check_all_documents()
                health["staleness_issues"] = len(staleness)
                health["expired_documents"] = len([s for s in staleness if s.staleness_type.value == "expired"])
                health["outdated_documents"] = len([s for s in staleness if s.staleness_type.value == "outdated_year"])

            if include_gaps:
                gaps = self.gap_analyzer.analyze_all()
                health["coverage_gaps"] = len(gaps)
                health["critical_gaps"] = len([g for g in gaps if g.severity.value in ["critical", "high"]])

            # Calculate overall health score (0-100)
            issues = (
                health.get("critical_conflicts", 0) * 10 +
                health.get("high_conflicts", 0) * 5 +
                health.get("expired_documents", 0) * 10 +
                health.get("critical_gaps", 0) * 3
            )
            health["health_score"] = max(0, 100 - issues)

            if health["health_score"] >= 80:
                health["status"] = "healthy"
            elif health["health_score"] >= 50:
                health["status"] = "warning"
            else:
                health["status"] = "critical"

            return ToolResult(success=True, data=health)

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Health check failed: {str(e)}"
            )

    def verify_resolution(self, alert_id: str) -> ToolResult:
        """Verify that a resolved alert has been properly fixed.

        Args:
            alert_id: ID of the alert to verify.

        Returns:
            ToolResult with verification outcome.
        """
        try:
            # Get the alert
            alert = self.alert_manager.get_alert(alert_id)
            if not alert:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Alert not found: {alert_id}"
                )

            if alert.resolution_status not in ("resolved", "in_progress"):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Alert is not resolved. Current status: {alert.resolution_status}"
                )

            # Re-run detection based on alert type
            still_exists = False
            verification_details = {}

            if alert.alert_type in ("conflict", "agent_generated"):
                # Re-run conflict detection
                topic = alert.metadata.get("topic") if alert.metadata else None
                conflicts = self.conflict_detector.detect_conflicts(topic=topic)

                # Check if the same conflict still exists
                for conflict in conflicts:
                    # Match by document pair and similar description
                    if (alert.title and conflict.description and
                        any(doc in alert.title for doc in [conflict.location_a.document_title, conflict.location_b.document_title])):
                        still_exists = True
                        verification_details = {
                            "conflict_found": True,
                            "conflict_type": conflict.conflict_type.value,
                            "value_a": conflict.value_a,
                            "value_b": conflict.value_b,
                        }
                        break

            elif alert.alert_type == "staleness":
                # Re-run staleness check
                issues = self.staleness_checker.check_all_documents()
                for issue in issues:
                    if alert.document_id == issue.document_id:
                        still_exists = True
                        verification_details = {
                            "staleness_found": True,
                            "staleness_type": issue.staleness_type.value,
                        }
                        break

            elif alert.alert_type == "gap":
                # Re-run gap analysis
                gaps = self.gap_analyzer.analyze_all()
                for gap in gaps:
                    if alert.title and gap.topic in alert.title:
                        still_exists = True
                        verification_details = {
                            "gap_found": True,
                            "topic": gap.topic,
                        }
                        break

            # Update verification status
            if still_exists:
                self.alert_manager.update_verification_status(
                    alert_id, "failed", verified_by="agent"
                )
                return ToolResult(
                    success=True,
                    data={
                        "alert_id": alert_id,
                        "verification_result": "failed",
                        "issue_still_exists": True,
                        "details": verification_details,
                        "message": "Issue still exists. Alert reopened for further action."
                    }
                )
            else:
                self.alert_manager.update_verification_status(
                    alert_id, "verified", verified_by="agent"
                )
                return ToolResult(
                    success=True,
                    data={
                        "alert_id": alert_id,
                        "verification_result": "verified",
                        "issue_still_exists": False,
                        "message": "Issue has been successfully resolved and verified."
                    }
                )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Verification failed: {str(e)}"
            )

    def get_remediation_suggestion(self, alert_id: str) -> ToolResult:
        """Get AI-powered remediation suggestion for an alert.

        Args:
            alert_id: ID of the alert.

        Returns:
            ToolResult with remediation suggestion.
        """
        try:
            # Get the alert
            alert = self.alert_manager.get_alert(alert_id)
            if not alert:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Alert not found: {alert_id}"
                )

            # Check if remediation already exists
            if alert.remediation:
                return ToolResult(
                    success=True,
                    data={
                        "alert_id": alert_id,
                        "remediation": alert.remediation,
                        "source": "cached"
                    }
                )

            # Generate remediation based on alert type
            if alert.alert_type in ("conflict", "agent_generated"):
                # Try to find the original conflict
                topic = alert.metadata.get("topic") if alert.metadata else None
                conflicts = self.conflict_detector.detect_conflicts(topic=topic)

                for conflict in conflicts:
                    if (alert.title and
                        any(doc in alert.title for doc in [conflict.location_a.document_title, conflict.location_b.document_title])):
                        suggestion = self.remediation_suggester.suggest_remediation(conflict, alert_id)
                        remediation_dict = suggestion.recommendation.to_dict()

                        # Save to alert
                        self.alert_manager.set_remediation_suggestion(alert_id, remediation_dict)

                        return ToolResult(
                            success=True,
                            data={
                                "alert_id": alert_id,
                                "remediation": remediation_dict,
                                "source": "generated"
                            }
                        )

            elif alert.alert_type == "staleness":
                suggestion = self.remediation_suggester.suggest_staleness_remediation(
                    document_title=alert.title,
                    staleness_type=alert.metadata.get("staleness_type", "unknown") if alert.metadata else "unknown",
                    expired_date=alert.metadata.get("expired_date") if alert.metadata else None,
                    alert_id=alert_id
                )
                remediation_dict = suggestion.recommendation.to_dict()
                self.alert_manager.set_remediation_suggestion(alert_id, remediation_dict)

                return ToolResult(
                    success=True,
                    data={
                        "alert_id": alert_id,
                        "remediation": remediation_dict,
                        "source": "generated"
                    }
                )

            elif alert.alert_type == "gap":
                covered_in = alert.metadata.get("covered_in", []) if alert.metadata else []
                missing_from = alert.metadata.get("missing_from", []) if alert.metadata else []
                suggestion = self.remediation_suggester.suggest_gap_remediation(
                    topic=alert.title or "unknown",
                    covered_in=covered_in,
                    missing_from=missing_from,
                    severity=alert.severity,
                    alert_id=alert_id
                )
                remediation_dict = suggestion.recommendation.to_dict()
                self.alert_manager.set_remediation_suggestion(alert_id, remediation_dict)

                return ToolResult(
                    success=True,
                    data={
                        "alert_id": alert_id,
                        "remediation": remediation_dict,
                        "source": "generated"
                    }
                )

            # Default fallback
            return ToolResult(
                success=True,
                data={
                    "alert_id": alert_id,
                    "remediation": {
                        "action": "escalate_to_owner",
                        "target_document": alert.document_id,
                        "suggested_change": "Review and resolve manually",
                        "rationale": "Unable to generate automated suggestion for this alert type.",
                        "priority": "high",
                        "estimated_effort": "30-60 minutes"
                    },
                    "source": "fallback"
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Remediation suggestion failed: {str(e)}"
            )

    def detect_semantic_conflicts(
        self,
        doc_ids: Optional[list[str]] = None,
        topic: Optional[str] = None,
        deep_scan: bool = False
    ) -> ToolResult:
        """Detect semantic conflicts using LLM reasoning.

        This is the key differentiator - finds conflicts that require UNDERSTANDING,
        not just pattern matching.

        Args:
            doc_ids: Optional list of document IDs to analyze.
            topic: Optional topic to focus on.
            deep_scan: If True, analyze more pairs (slower but thorough).

        Returns:
            ToolResult with semantic conflicts.
        """
        try:
            conflicts = self.semantic_detector.detect_semantic_conflicts(
                doc_ids=doc_ids,
                topic=topic,
                deep_scan=deep_scan,
                max_pairs=100 if deep_scan else 30,
            )

            # Convert to dictionaries
            conflict_dicts = [c.to_dict() for c in conflicts]

            # Group by type
            by_type = {}
            for c in conflict_dicts:
                ctype = c["conflict_type"]
                if ctype not in by_type:
                    by_type[ctype] = []
                by_type[ctype].append(c)

            # Calculate severity summary
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for c in conflict_dicts:
                severity_counts[c["severity"]] = severity_counts.get(c["severity"], 0) + 1

            return ToolResult(
                success=True,
                data={
                    "total_conflicts": len(conflicts),
                    "conflicts": conflict_dicts,
                    "by_type": by_type,
                    "severity_summary": severity_counts,
                    "analysis_type": "semantic",
                    "deep_scan": deep_scan,
                    "topic_focus": topic,
                    "documents_analyzed": doc_ids if doc_ids else "all",
                    "differentiator": "Uses LLM reasoning to find conflicts that pattern-matching misses"
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Semantic conflict detection failed: {str(e)}"
            )

    def execute_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dictionary of arguments.

        Returns:
            ToolResult from the tool execution.
        """
        tool_map = {
            "search_documents": self.search_documents,
            "compare_sections": self.compare_sections,
            "run_consistency_check": self.run_consistency_check,
            "generate_report": self.generate_report,
            "create_alert": self.create_alert,
            "get_document_health": self.get_document_health,
            "verify_resolution": self.verify_resolution,
            "get_remediation_suggestion": self.get_remediation_suggestion,
            "detect_semantic_conflicts": self.detect_semantic_conflicts,
        }

        if tool_name not in tool_map:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}"
            )

        try:
            return tool_map[tool_name](**arguments)
        except TypeError as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid arguments for {tool_name}: {str(e)}"
            )
