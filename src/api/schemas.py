"""Pydantic schemas for API request/response models."""

from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request schema."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    use_hybrid: bool = Field(True, description="Use hybrid search (BM25 + vector)")
    bm25_weight: float = Field(0.3, ge=0, le=1, description="Weight for BM25 scoring")
    vector_weight: float = Field(0.7, ge=0, le=1, description="Weight for vector scoring")


class SearchResult(BaseModel):
    """Single search result."""

    id: str
    score: float
    document_id: str
    document_title: str
    section_title: str
    content: str
    page_number: Optional[int] = None


class SearchResponse(BaseModel):
    """Search response schema."""

    query: str
    results: list[SearchResult]
    total: int


class IngestResponse(BaseModel):
    """Ingest response schema."""

    document_id: str
    filename: str
    chunk_count: int
    indexed_at: str


class DocumentResponse(BaseModel):
    """Document response schema."""

    id: str
    filename: str
    title: str
    file_type: str
    page_count: int
    section_count: int
    chunk_count: int
    indexed_at: str


class HealthResponse(BaseModel):
    """Health response schema."""

    status: str
    color: str
    document_count: int
    chunk_count: int
    open_alerts: int
    critical_alerts: int
    high_alerts: int


class AlertCreate(BaseModel):
    """Create alert request schema."""

    document_id: str
    alert_type: str = Field(..., description="Type: conflict, staleness, gap, compliance")
    severity: str = Field(..., description="Severity: critical, high, medium, low")
    title: str
    description: str
    metadata: Optional[dict] = None


class AlertResponse(BaseModel):
    """Alert response schema."""

    id: str
    document_id: str
    alert_type: str
    severity: str
    title: str
    description: str
    status: str
    created_at: str
    metadata: Optional[dict] = None
