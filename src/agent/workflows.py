"""Pre-defined workflows for common document operations.

These workflows orchestrate multiple tool calls to accomplish
complex analysis tasks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .tools import AgentTools, ToolResult


@dataclass
class WorkflowResult:
    """Result from a workflow execution."""
    workflow_name: str
    success: bool
    summary: str
    details: dict
    steps_executed: list[dict]
    alerts_created: int
    duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "workflow_name": self.workflow_name,
            "success": self.success,
            "summary": self.summary,
            "details": self.details,
            "steps_executed": self.steps_executed,
            "alerts_created": self.alerts_created,
            "duration_ms": self.duration_ms,
            "error": self.error
        }


class WorkflowEngine:
    """Execute pre-defined workflows for document analysis."""

    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        es_host: str = "localhost",
        es_port: int = 9200,
        es_scheme: str = "http",
    ):
        """Initialize the workflow engine.

        Args:
            tools: Pre-configured AgentTools instance.
            es_host: Elasticsearch host.
            es_port: Elasticsearch port.
            es_scheme: http or https.
        """
        self.tools = tools or AgentTools(
            host=es_host,
            port=es_port,
            scheme=es_scheme
        )

    def get_available_workflows(self) -> list[dict]:
        """Get list of available workflows.

        Returns:
            List of workflow definitions.
        """
        return [
            {
                "name": "conflict_scan",
                "description": "Full corpus scan for conflicts and inconsistencies",
                "parameters": {
                    "topic": "Optional topic to focus on",
                    "create_alerts": "Whether to create alerts for findings (default: True)"
                }
            },
            {
                "name": "compliance_audit",
                "description": "Comprehensive compliance audit with report generation",
                "parameters": {
                    "focus_areas": "List of areas to focus on (e.g., ['security', 'data'])"
                }
            },
            {
                "name": "document_review",
                "description": "Deep review of a specific document",
                "parameters": {
                    "document_id": "ID of document to review (required)"
                }
            },
            {
                "name": "staleness_audit",
                "description": "Check all documents for staleness and expiration",
                "parameters": {}
            },
            {
                "name": "gap_analysis",
                "description": "Analyze coverage gaps across the document corpus",
                "parameters": {}
            }
        ]

    def execute_workflow(
        self,
        workflow_name: str,
        parameters: Optional[dict] = None
    ) -> WorkflowResult:
        """Execute a workflow by name.

        Args:
            workflow_name: Name of the workflow to execute.
            parameters: Optional parameters for the workflow.

        Returns:
            WorkflowResult with execution details.
        """
        parameters = parameters or {}

        workflow_map = {
            "conflict_scan": self._workflow_conflict_scan,
            "compliance_audit": self._workflow_compliance_audit,
            "document_review": self._workflow_document_review,
            "staleness_audit": self._workflow_staleness_audit,
            "gap_analysis": self._workflow_gap_analysis,
        }

        if workflow_name not in workflow_map:
            return WorkflowResult(
                workflow_name=workflow_name,
                success=False,
                summary="Unknown workflow",
                details={},
                steps_executed=[],
                alerts_created=0,
                duration_ms=0,
                error=f"Unknown workflow: {workflow_name}"
            )

        start_time = datetime.utcnow()
        try:
            result = workflow_map[workflow_name](parameters)
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.duration_ms = duration
            return result
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return WorkflowResult(
                workflow_name=workflow_name,
                success=False,
                summary=f"Workflow failed: {str(e)}",
                details={},
                steps_executed=[],
                alerts_created=0,
                duration_ms=duration,
                error=str(e)
            )

    def _workflow_conflict_scan(self, params: dict) -> WorkflowResult:
        """Full corpus conflict scan workflow.

        Steps:
        1. Get document health overview
        2. Run consistency check (optionally focused on topic)
        3. Generate conflict report
        4. Create alerts for critical/high issues
        """
        steps = []
        alerts_created = 0
        topic = params.get("topic")
        create_alerts = params.get("create_alerts", True)

        # Step 1: Get health overview
        health_result = self.tools.get_document_health(
            include_staleness=False,
            include_gaps=False
        )
        steps.append({
            "step": 1,
            "tool": "get_document_health",
            "success": health_result.success,
            "summary": f"Documents: {health_result.data.get('document_count', 0)}" if health_result.success else health_result.error
        })

        # Step 2: Run consistency check
        consistency_result = self.tools.run_consistency_check(focus_topic=topic)
        inconsistencies = consistency_result.data.get("inconsistencies", []) if consistency_result.success else []
        steps.append({
            "step": 2,
            "tool": "run_consistency_check",
            "success": consistency_result.success,
            "summary": f"Found {len(inconsistencies)} conflicts" if consistency_result.success else consistency_result.error
        })

        # Step 3: Generate conflict report
        report_result = self.tools.generate_report(
            report_type="conflict",
            scope=f"Topic: {topic}" if topic else "Full corpus",
            include_recommendations=True
        )
        steps.append({
            "step": 3,
            "tool": "generate_report",
            "success": report_result.success,
            "summary": "Report generated" if report_result.success else report_result.error
        })

        # Step 4: Create alerts for critical/high issues
        if create_alerts and consistency_result.success:
            for issue in inconsistencies:
                if issue.get("severity") in ["critical", "high"]:
                    alert_result = self.tools.create_alert(
                        severity=issue.get("severity"),
                        title=f"Conflict: {issue.get('type', 'Unknown')}",
                        description=issue.get("description", ""),
                        affected_doc_ids=[]
                    )
                    if alert_result.success:
                        alerts_created += 1

            steps.append({
                "step": 4,
                "tool": "create_alert",
                "success": True,
                "summary": f"Created {alerts_created} alerts"
            })

        # Build summary
        critical = len([i for i in inconsistencies if i.get("severity") == "critical"])
        high = len([i for i in inconsistencies if i.get("severity") == "high"])

        summary = f"Conflict scan complete. Found {len(inconsistencies)} conflicts ({critical} critical, {high} high)."
        if alerts_created > 0:
            summary += f" Created {alerts_created} alerts."

        return WorkflowResult(
            workflow_name="conflict_scan",
            success=True,
            summary=summary,
            details={
                "total_conflicts": len(inconsistencies),
                "critical": critical,
                "high": high,
                "medium": len([i for i in inconsistencies if i.get("severity") == "medium"]),
                "low": len([i for i in inconsistencies if i.get("severity") == "low"]),
                "conflicts": inconsistencies,
                "report": report_result.data if report_result.success else None
            },
            steps_executed=steps,
            alerts_created=alerts_created,
            duration_ms=0  # Will be set by caller
        )

    def _workflow_compliance_audit(self, params: dict) -> WorkflowResult:
        """Comprehensive compliance audit workflow.

        Steps:
        1. Get document health
        2. Run consistency check
        3. Check for staleness
        4. Analyze coverage gaps
        5. Generate compliance report
        """
        steps = []
        focus_areas = params.get("focus_areas", [])

        # Step 1: Get health
        health_result = self.tools.get_document_health(
            include_staleness=True,
            include_gaps=True
        )
        steps.append({
            "step": 1,
            "tool": "get_document_health",
            "success": health_result.success,
            "summary": f"Health score: {health_result.data.get('health_score', 'N/A')}" if health_result.success else health_result.error
        })

        # Step 2: Run consistency checks for each focus area
        all_conflicts = []
        for area in focus_areas or [None]:
            consistency_result = self.tools.run_consistency_check(focus_topic=area)
            if consistency_result.success:
                all_conflicts.extend(consistency_result.data.get("inconsistencies", []))

        steps.append({
            "step": 2,
            "tool": "run_consistency_check",
            "success": True,
            "summary": f"Found {len(all_conflicts)} total conflicts"
        })

        # Step 3: Generate compliance report
        report_result = self.tools.generate_report(
            report_type="compliance",
            scope=f"Focus areas: {', '.join(focus_areas)}" if focus_areas else "Full audit",
            include_recommendations=True
        )
        steps.append({
            "step": 3,
            "tool": "generate_report",
            "success": report_result.success,
            "summary": "Compliance report generated" if report_result.success else report_result.error
        })

        health_data = health_result.data if health_result.success else {}

        return WorkflowResult(
            workflow_name="compliance_audit",
            success=True,
            summary=f"Compliance audit complete. Health score: {health_data.get('health_score', 'N/A')}/100",
            details={
                "health_score": health_data.get("health_score"),
                "status": health_data.get("status"),
                "conflicts": len(all_conflicts),
                "staleness_issues": health_data.get("staleness_issues", 0),
                "coverage_gaps": health_data.get("coverage_gaps", 0),
                "report": report_result.data if report_result.success else None
            },
            steps_executed=steps,
            alerts_created=0,
            duration_ms=0
        )

    def _workflow_document_review(self, params: dict) -> WorkflowResult:
        """Deep review of a specific document.

        Steps:
        1. Search for the document
        2. Check for conflicts with other documents
        3. Check for staleness
        4. Identify coverage gaps
        """
        steps = []
        document_id = params.get("document_id")

        if not document_id:
            return WorkflowResult(
                workflow_name="document_review",
                success=False,
                summary="Missing required parameter: document_id",
                details={},
                steps_executed=[],
                alerts_created=0,
                duration_ms=0,
                error="document_id is required"
            )

        # Step 1: Search for related content
        search_result = self.tools.search_documents(
            query=document_id,
            top_k=10
        )
        steps.append({
            "step": 1,
            "tool": "search_documents",
            "success": search_result.success,
            "summary": f"Found {search_result.data.get('total_hits', 0)} related sections" if search_result.success else search_result.error
        })

        # Step 2: Check for conflicts
        consistency_result = self.tools.run_consistency_check(
            doc_ids=[document_id]
        )
        conflicts = consistency_result.data.get("inconsistencies", []) if consistency_result.success else []
        steps.append({
            "step": 2,
            "tool": "run_consistency_check",
            "success": consistency_result.success,
            "summary": f"Found {len(conflicts)} conflicts involving this document"
        })

        # Step 3: Get health info
        health_result = self.tools.get_document_health()
        steps.append({
            "step": 3,
            "tool": "get_document_health",
            "success": health_result.success,
            "summary": "Health data retrieved" if health_result.success else health_result.error
        })

        return WorkflowResult(
            workflow_name="document_review",
            success=True,
            summary=f"Document review complete. Found {len(conflicts)} conflicts.",
            details={
                "document_id": document_id,
                "conflicts": conflicts,
                "related_sections": search_result.data.get("results", []) if search_result.success else []
            },
            steps_executed=steps,
            alerts_created=0,
            duration_ms=0
        )

    def _workflow_staleness_audit(self, params: dict) -> WorkflowResult:
        """Audit all documents for staleness.

        Steps:
        1. Get health with staleness info
        2. Generate staleness report
        3. Create alerts for expired documents
        """
        steps = []
        alerts_created = 0

        # Step 1: Get health
        health_result = self.tools.get_document_health(
            include_staleness=True,
            include_gaps=False
        )
        steps.append({
            "step": 1,
            "tool": "get_document_health",
            "success": health_result.success,
            "summary": f"Found {health_result.data.get('staleness_issues', 0)} staleness issues" if health_result.success else health_result.error
        })

        # Step 2: Generate report
        report_result = self.tools.generate_report(
            report_type="staleness",
            include_recommendations=True
        )
        steps.append({
            "step": 2,
            "tool": "generate_report",
            "success": report_result.success,
            "summary": "Staleness report generated" if report_result.success else report_result.error
        })

        # Step 3: Create alerts for expired documents
        health_data = health_result.data if health_result.success else {}
        expired = health_data.get("expired_documents", 0)

        if expired > 0:
            alert_result = self.tools.create_alert(
                severity="critical",
                title=f"{expired} Expired Document(s) Found",
                description=f"Found {expired} expired documents that need immediate attention.",
                affected_doc_ids=[]
            )
            if alert_result.success:
                alerts_created += 1

            steps.append({
                "step": 3,
                "tool": "create_alert",
                "success": alert_result.success,
                "summary": f"Alert created for expired documents"
            })

        return WorkflowResult(
            workflow_name="staleness_audit",
            success=True,
            summary=f"Staleness audit complete. {health_data.get('staleness_issues', 0)} issues found, {expired} documents expired.",
            details={
                "staleness_issues": health_data.get("staleness_issues", 0),
                "expired_documents": expired,
                "outdated_documents": health_data.get("outdated_documents", 0),
                "report": report_result.data if report_result.success else None
            },
            steps_executed=steps,
            alerts_created=alerts_created,
            duration_ms=0
        )

    def _workflow_gap_analysis(self, params: dict) -> WorkflowResult:
        """Analyze coverage gaps across the corpus.

        Steps:
        1. Get health with gap info
        2. Generate gap report
        """
        steps = []

        # Step 1: Get health
        health_result = self.tools.get_document_health(
            include_staleness=False,
            include_gaps=True
        )
        steps.append({
            "step": 1,
            "tool": "get_document_health",
            "success": health_result.success,
            "summary": f"Found {health_result.data.get('coverage_gaps', 0)} coverage gaps" if health_result.success else health_result.error
        })

        # Step 2: Generate report
        report_result = self.tools.generate_report(
            report_type="gap",
            include_recommendations=True
        )
        steps.append({
            "step": 2,
            "tool": "generate_report",
            "success": report_result.success,
            "summary": "Gap analysis report generated" if report_result.success else report_result.error
        })

        health_data = health_result.data if health_result.success else {}

        return WorkflowResult(
            workflow_name="gap_analysis",
            success=True,
            summary=f"Gap analysis complete. Found {health_data.get('coverage_gaps', 0)} gaps ({health_data.get('critical_gaps', 0)} critical/high priority).",
            details={
                "coverage_gaps": health_data.get("coverage_gaps", 0),
                "critical_gaps": health_data.get("critical_gaps", 0),
                "report": report_result.data if report_result.success else None
            },
            steps_executed=steps,
            alerts_created=0,
            duration_ms=0
        )
