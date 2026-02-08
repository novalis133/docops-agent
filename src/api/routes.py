"""FastAPI routes for DocOps Agent."""

from pathlib import Path
from typing import Optional
import tempfile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    DocumentResponse,
    HealthResponse,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from ..config import settings
from ..main import (
    create_embedder,
    create_indexer,
    get_corpus_health,
    ingest_document,
    search,
)

app = FastAPI(
    title="DocOps Agent API",
    description="Intelligent document operations platform",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Get corpus health status."""
    try:
        health = get_corpus_health()
        return HealthResponse(**health)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a document file."""
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}",
        )

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Ingest
        result = ingest_document(tmp_path)

        # Clean up
        Path(tmp_path).unlink(missing_ok=True)

        return IngestResponse(
            document_id=result["document_id"],
            filename=file.filename,
            chunk_count=result["chunk_count"],
            indexed_at=result["indexed_at"],
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents."""
    try:
        results = search(
            query=request.query,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            bm25_weight=request.bm25_weight,
            vector_weight=request.vector_weight,
        )

        return SearchResponse(
            query=request.query,
            results=[
                SearchResult(
                    id=r["id"],
                    score=r.get("score", 0.0),
                    document_id=r.get("document_id", ""),
                    document_title=r.get("document_title", ""),
                    section_title=r.get("section_title", ""),
                    content=r.get("content", ""),
                    page_number=r.get("page_number"),
                )
                for r in results
            ],
            total=len(results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get a document by ID."""
    try:
        indexer = create_indexer()
        doc = indexer.get_document(document_id)

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentResponse(
            id=doc["id"],
            filename=doc.get("filename", ""),
            title=doc.get("title", ""),
            file_type=doc.get("file_type", ""),
            page_count=doc.get("page_count", 1),
            section_count=doc.get("section_count", 0),
            chunk_count=doc.get("chunk_count", 0),
            indexed_at=doc.get("indexed_at", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks."""
    try:
        indexer = create_indexer()
        success = indexer.delete_document(document_id, refresh=True)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"message": "Document deleted", "document_id": document_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """Get all chunks for a document."""
    try:
        indexer = create_indexer()
        chunks = indexer.get_document_chunks(document_id)

        return {
            "document_id": document_id,
            "chunks": chunks,
            "total": len(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts")
async def get_alerts(
    document_id: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    status: str = Query("open"),
    top_k: int = Query(100, ge=1, le=1000),
):
    """Get alerts with optional filtering."""
    try:
        indexer = create_indexer()
        alerts = indexer.get_alerts(
            document_id=document_id,
            severity=severity,
            status=status,
            top_k=top_k,
        )

        return {
            "alerts": alerts,
            "total": len(alerts),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get corpus statistics."""
    try:
        indexer = create_indexer()
        stats = indexer.get_corpus_stats()
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Agent Endpoints
# =============================================================================

from pydantic import BaseModel
from typing import List, Dict, Any

from ..agent import DocOpsAgent, WorkflowEngine


class AgentChatRequest(BaseModel):
    """Request for agent chat."""
    message: str


class AgentChatResponse(BaseModel):
    """Response from agent chat."""
    answer: str
    steps: List[Dict[str, Any]]
    tools_used: List[str]
    total_steps: int
    success: bool
    error: Optional[str] = None


class WorkflowRequest(BaseModel):
    """Request to execute a workflow."""
    workflow: str
    parameters: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    """Response from workflow execution."""
    workflow_name: str
    success: bool
    summary: str
    details: Dict[str, Any]
    steps_executed: List[Dict[str, Any]]
    alerts_created: int
    duration_ms: float
    error: Optional[str] = None


# Agent singleton
_agent: Optional[DocOpsAgent] = None
_workflow_engine: Optional[WorkflowEngine] = None


def get_agent() -> DocOpsAgent:
    """Get or create the agent singleton."""
    global _agent
    if _agent is None:
        _agent = DocOpsAgent(
            es_host=settings.elasticsearch.host,
            es_port=settings.elasticsearch.port,
            es_scheme=settings.elasticsearch.scheme,
        )
    return _agent


def get_workflow_engine() -> WorkflowEngine:
    """Get or create the workflow engine singleton."""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine(
            es_host=settings.elasticsearch.host,
            es_port=settings.elasticsearch.port,
            es_scheme=settings.elasticsearch.scheme,
        )
    return _workflow_engine


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """Send a message to the agent and get a response with step trace.

    The agent will:
    1. Analyze your request
    2. Execute appropriate tools (search, compare, check consistency, etc.)
    3. Return a response with the full execution trace
    """
    try:
        agent = get_agent()
        response = agent.chat(request.message)

        return AgentChatResponse(
            answer=response.answer,
            steps=[s.to_dict() for s in response.steps],
            tools_used=response.tools_used,
            total_steps=response.total_steps,
            success=response.success,
            error=response.error,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/workflow", response_model=WorkflowResponse)
async def execute_workflow(request: WorkflowRequest):
    """Execute a pre-defined workflow.

    Available workflows:
    - conflict_scan: Full corpus scan for conflicts
    - compliance_audit: Comprehensive compliance audit
    - document_review: Deep review of a specific document
    - staleness_audit: Check all documents for staleness
    - gap_analysis: Analyze coverage gaps
    """
    try:
        engine = get_workflow_engine()
        result = engine.execute_workflow(
            workflow_name=request.workflow,
            parameters=request.parameters or {}
        )

        return WorkflowResponse(**result.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/workflows")
async def list_workflows():
    """List available workflows and their parameters."""
    try:
        engine = get_workflow_engine()
        workflows = engine.get_available_workflows()
        return {"workflows": workflows}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/tools")
async def list_agent_tools():
    """List available agent tools and their schemas."""
    try:
        agent = get_agent()
        tools = agent.tool_definitions
        return {"tools": tools}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/reset")
async def reset_agent():
    """Reset the agent conversation history."""
    try:
        agent = get_agent()
        agent.reset_conversation()
        return {"message": "Agent conversation reset"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Alert Resolution Lifecycle Endpoints
# =============================================================================

from ..actions.alert_manager import AlertManager


class ResolveAlertRequest(BaseModel):
    """Request to resolve an alert."""
    resolution_status: str  # open, in_progress, resolved, wont_fix
    resolution_notes: Optional[str] = None
    resolved_by: Optional[str] = None


class VerifyAlertRequest(BaseModel):
    """Request to verify an alert resolution."""
    verified_by: Optional[str] = None


# Alert manager singleton
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create the alert manager singleton."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(
            host=settings.elasticsearch.host,
            port=settings.elasticsearch.port,
            scheme=settings.elasticsearch.scheme,
        )
    return _alert_manager


@app.patch("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, request: ResolveAlertRequest):
    """Update the resolution status of an alert.

    Resolution statuses:
    - open: Alert is open and needs attention
    - in_progress: Someone is working on fixing this
    - resolved: The issue has been fixed
    - wont_fix: Decided not to fix (with notes explaining why)
    """
    try:
        manager = get_alert_manager()
        success = manager.update_resolution_status(
            alert_id=alert_id,
            resolution_status=request.resolution_status,
            resolution_notes=request.resolution_notes,
            resolved_by=request.resolved_by
        )

        if not success:
            raise HTTPException(status_code=404, detail="Alert not found or invalid status")

        # Get updated alert
        alert = manager.get_alert(alert_id)

        return {
            "message": f"Alert resolution status updated to {request.resolution_status}",
            "alert_id": alert_id,
            "resolution_status": request.resolution_status,
            "alert": alert.to_dict() if alert else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/{alert_id}/verify")
async def verify_alert(alert_id: str, request: Optional[VerifyAlertRequest] = None):
    """Verify that a resolved alert has been properly fixed.

    Triggers a re-scan using the agent to confirm the issue no longer exists.
    """
    try:
        agent = get_agent()

        # Use the verify_resolution tool
        result = agent.tools.verify_resolution(alert_id)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        return {
            "message": "Verification complete",
            "alert_id": alert_id,
            "verification_result": result.data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/{alert_id}/remediation")
async def get_alert_remediation(alert_id: str):
    """Get the remediation suggestion for an alert.

    If no suggestion exists, generates one using AI.
    """
    try:
        agent = get_agent()

        # Use the get_remediation_suggestion tool
        result = agent.tools.get_remediation_suggestion(alert_id)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        return {
            "alert_id": alert_id,
            "remediation": result.data.get("remediation"),
            "source": result.data.get("source")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/lifecycle")
async def get_alerts_lifecycle():
    """Get alerts grouped by resolution and verification status.

    Returns counts for each lifecycle stage:
    - open: Alerts needing attention
    - in_progress: Being worked on
    - resolved_pending_verification: Fixed but not verified
    - verified: Confirmed fixed
    - verification_failed: Fix attempt failed
    - wont_fix: Decided not to fix
    """
    try:
        manager = get_alert_manager()
        lifecycle = manager.get_alerts_by_lifecycle()

        return {
            "lifecycle": lifecycle,
            "summary": {
                "needs_attention": (
                    lifecycle.get("lifecycle_stages", {}).get("open", 0) +
                    lifecycle.get("lifecycle_stages", {}).get("verification_failed", 0)
                ),
                "in_progress": lifecycle.get("lifecycle_stages", {}).get("in_progress", 0),
                "pending_verification": lifecycle.get("lifecycle_stages", {}).get("resolved_pending_verification", 0),
                "completed": lifecycle.get("lifecycle_stages", {}).get("verified", 0),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/pending-verification")
async def get_pending_verifications(top_k: int = Query(50, ge=1, le=200)):
    """Get alerts that are resolved but pending verification."""
    try:
        manager = get_alert_manager()
        alerts = manager.get_pending_verifications(top_k=top_k)

        return {
            "alerts": [a.to_dict() for a in alerts],
            "total": len(alerts)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Advanced Analytics Endpoints
# =============================================================================

from ..analysis import CorpusAnalyticsEngine


# Analytics engine singleton
_analytics_engine: Optional[CorpusAnalyticsEngine] = None


def get_analytics_engine() -> CorpusAnalyticsEngine:
    """Get or create the analytics engine singleton."""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = CorpusAnalyticsEngine(
            host=settings.elasticsearch.host,
            port=settings.elasticsearch.port,
            scheme=settings.elasticsearch.scheme,
        )
    return _analytics_engine


@app.get("/analytics")
async def get_corpus_analytics():
    """Get comprehensive corpus analytics using advanced Elasticsearch aggregations.

    Returns metrics including:
    - Staleness distribution (using runtime fields)
    - Conflicts over time (date histogram)
    - Document type health (nested aggregations)
    - Significant conflict terms
    - Complexity analysis
    """
    try:
        engine = get_analytics_engine()
        analytics = engine.get_corpus_analytics()

        return {
            "analytics": analytics.to_dict(),
            "elasticsearch_features_used": [
                "runtime_fields",
                "date_histogram",
                "nested_aggregations",
                "significant_text",
                "bucket_script"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/trends")
async def get_trend_analysis(
    metric: str = Query("alerts", enum=["alerts", "documents", "chunks"]),
    period_days: int = Query(30, ge=7, le=365)
):
    """Analyze trends for a specific metric over time.

    Uses Elasticsearch range aggregations to compare current vs previous periods.
    """
    try:
        engine = get_analytics_engine()
        trend = engine.get_trend_analysis(metric=metric, period_days=period_days)

        return {
            "trend_analysis": trend,
            "elasticsearch_features_used": ["range_filter", "period_comparison"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/hotspots")
async def get_hotspot_analysis(top_k: int = Query(10, ge=1, le=50)):
    """Identify document sections that are conflict hotspots.

    Uses multi-level aggregations with risk scoring.
    """
    try:
        engine = get_analytics_engine()
        hotspots = engine.get_hotspot_analysis(top_k=top_k)

        return {
            "hotspots": hotspots,
            "total": len(hotspots),
            "elasticsearch_features_used": ["terms_aggregation", "nested_aggs", "bucket_sort"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
