"""Document analysis components.

This module provides tools for analyzing documents:
- Entity extraction (dates, amounts, durations, requirements)
- Conflict detection (inconsistencies between documents)
- Staleness checking (expired or outdated content)
- Gap analysis (missing coverage across documents)
"""

from .entity_extractor import (
    Entity,
    EntityExtractor,
    EntityType,
)
from .conflict_detector import (
    Conflict,
    ConflictDetector,
    ConflictLocation,
    ConflictSeverity,
    ConflictType,
)
from .staleness_checker import (
    StalenessChecker,
    StalenessIssue,
    StalenessSeverity,
    StalenessType,
)
from .gap_analyzer import (
    CoverageGap,
    GapAnalyzer,
    GapSeverity,
    GapType,
)
from .remediation_suggester import (
    RecommendedAction,
    RemediationAction,
    RemediationPriority,
    RemediationSuggester,
    RemediationSuggestion,
)
from .analytics import (
    CorpusAnalytics,
    CorpusAnalyticsEngine,
    TimeSeriesDataPoint,
)
from .semantic_conflict_detector import (
    SemanticConflict,
    SemanticConflictDetector,
    SemanticConflictType,
    SemanticSeverity,
)

__all__ = [
    # Entity extraction
    "Entity",
    "EntityExtractor",
    "EntityType",
    # Conflict detection
    "Conflict",
    "ConflictDetector",
    "ConflictLocation",
    "ConflictSeverity",
    "ConflictType",
    # Staleness checking
    "StalenessChecker",
    "StalenessIssue",
    "StalenessSeverity",
    "StalenessType",
    # Gap analysis
    "CoverageGap",
    "GapAnalyzer",
    "GapSeverity",
    "GapType",
    # Remediation suggestions
    "RecommendedAction",
    "RemediationAction",
    "RemediationPriority",
    "RemediationSuggester",
    "RemediationSuggestion",
    # Analytics
    "CorpusAnalytics",
    "CorpusAnalyticsEngine",
    "TimeSeriesDataPoint",
    # Semantic conflict detection
    "SemanticConflict",
    "SemanticConflictDetector",
    "SemanticConflictType",
    "SemanticSeverity",
]
