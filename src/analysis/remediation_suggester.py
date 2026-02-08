"""Remediation suggestion engine for detected conflicts and issues.

Provides intelligent recommendations for resolving conflicts, staleness,
and coverage gaps based on document authority, recency, and policy logic.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .conflict_detector import Conflict, ConflictType, ConflictSeverity


class RemediationAction(str, Enum):
    """Types of remediation actions."""
    UPDATE_DOCUMENT = "update_document"
    RETIRE_DOCUMENT = "retire_document"
    ADD_SECTION = "add_section"
    CLARIFY_POLICY = "clarify_policy"
    MERGE_DOCUMENTS = "merge_documents"
    ESCALATE_TO_OWNER = "escalate_to_owner"


class RemediationPriority(str, Enum):
    """Priority levels for remediation."""
    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RecommendedAction:
    """A recommended action to resolve an issue."""
    action: RemediationAction
    target_document: str
    target_section: Optional[str]
    suggested_change: str
    rationale: str
    priority: RemediationPriority
    estimated_effort: str
    confidence: float = 0.0
    alternative_actions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "target_document": self.target_document,
            "target_section": self.target_section,
            "suggested_change": self.suggested_change,
            "rationale": self.rationale,
            "priority": self.priority.value,
            "estimated_effort": self.estimated_effort,
            "confidence": self.confidence,
            "alternative_actions": [
                alt.to_dict() if hasattr(alt, 'to_dict') else alt
                for alt in self.alternative_actions
            ]
        }


@dataclass
class RemediationSuggestion:
    """Complete remediation suggestion for an alert."""
    alert_id: str
    issue_type: str
    recommendation: RecommendedAction
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "issue_type": self.issue_type,
            "recommendation": self.recommendation.to_dict(),
            "created_at": self.created_at.isoformat()
        }


class RemediationSuggester:
    """Generate remediation suggestions for detected issues.

    Resolution Logic:
    1. Stricter policy wins (e.g., 14 chars > 12 chars for security)
    2. Newer document wins (if modification dates known)
    3. Higher authority wins (Security Policy > Employee Handbook)
    """

    # Document authority hierarchy (higher number = more authoritative)
    AUTHORITY_HIERARCHY = {
        "security policy": 100,
        "security": 95,
        "compliance": 90,
        "legal": 90,
        "policy": 80,
        "standard": 75,
        "procedure": 70,
        "guideline": 60,
        "handbook": 50,
        "guide": 45,
        "manual": 40,
        "faq": 20,
        "readme": 10,
    }

    # Security-related topics where stricter values should win
    STRICTER_WINS_TOPICS = [
        "password", "authentication", "security", "access", "encryption",
        "retention", "backup", "compliance", "audit", "privacy"
    ]

    def __init__(self):
        """Initialize the remediation suggester."""
        pass

    def suggest_remediation(self, conflict: Conflict, alert_id: str = "") -> RemediationSuggestion:
        """Generate remediation suggestion for a conflict.

        Args:
            conflict: The detected conflict.
            alert_id: Optional alert ID to associate with.

        Returns:
            RemediationSuggestion with recommended action.
        """
        if conflict.conflict_type == ConflictType.NUMERIC:
            return self._suggest_numeric_remediation(conflict, alert_id)
        elif conflict.conflict_type == ConflictType.DURATION:
            return self._suggest_duration_remediation(conflict, alert_id)
        elif conflict.conflict_type == ConflictType.POLICY:
            return self._suggest_policy_remediation(conflict, alert_id)
        elif conflict.conflict_type == ConflictType.MONETARY:
            return self._suggest_monetary_remediation(conflict, alert_id)
        else:
            return self._suggest_generic_remediation(conflict, alert_id)

    def suggest_staleness_remediation(
        self,
        document_title: str,
        staleness_type: str,
        expired_date: Optional[str] = None,
        sections_affected: Optional[list[str]] = None,
        alert_id: str = ""
    ) -> RemediationSuggestion:
        """Generate remediation suggestion for staleness issues.

        Args:
            document_title: Title of the stale document.
            staleness_type: Type of staleness (expired, outdated_year, etc.).
            expired_date: The expired date if known.
            sections_affected: Specific sections that are stale.
            alert_id: Optional alert ID.

        Returns:
            RemediationSuggestion for the staleness issue.
        """
        if staleness_type == "expired":
            action = RecommendedAction(
                action=RemediationAction.UPDATE_DOCUMENT,
                target_document=document_title,
                target_section=sections_affected[0] if sections_affected else None,
                suggested_change=f"Update expiration date or retire document (expired: {expired_date})",
                rationale="Document has passed its expiration date and may contain outdated information.",
                priority=RemediationPriority.IMMEDIATE,
                estimated_effort="15-30 minutes",
                confidence=0.95,
                alternative_actions=[
                    {
                        "action": "retire_document",
                        "description": "Mark document as retired if no longer needed"
                    }
                ]
            )
        elif staleness_type == "outdated_year":
            action = RecommendedAction(
                action=RemediationAction.UPDATE_DOCUMENT,
                target_document=document_title,
                target_section=None,
                suggested_change="Review and update year references to current year",
                rationale="Document contains references to past years that may be outdated.",
                priority=RemediationPriority.HIGH,
                estimated_effort="30-60 minutes",
                confidence=0.80
            )
        else:
            action = RecommendedAction(
                action=RemediationAction.UPDATE_DOCUMENT,
                target_document=document_title,
                target_section=None,
                suggested_change="Review document for outdated content",
                rationale="Document flagged for potential staleness.",
                priority=RemediationPriority.MEDIUM,
                estimated_effort="1-2 hours",
                confidence=0.60
            )

        return RemediationSuggestion(
            alert_id=alert_id,
            issue_type="staleness",
            recommendation=action
        )

    def suggest_gap_remediation(
        self,
        topic: str,
        covered_in: list[str],
        missing_from: list[str],
        severity: str,
        alert_id: str = ""
    ) -> RemediationSuggestion:
        """Generate remediation suggestion for coverage gaps.

        Args:
            topic: The topic that has a coverage gap.
            covered_in: Documents that cover this topic.
            missing_from: Documents missing this topic.
            severity: Severity of the gap.
            alert_id: Optional alert ID.

        Returns:
            RemediationSuggestion for the gap.
        """
        # Determine which document should add the coverage
        target_doc = self._select_target_for_gap(missing_from, covered_in, topic)
        source_doc = covered_in[0] if covered_in else "existing documentation"

        priority = RemediationPriority.HIGH if severity in ["critical", "high"] else RemediationPriority.MEDIUM

        action = RecommendedAction(
            action=RemediationAction.ADD_SECTION,
            target_document=target_doc,
            target_section=f"{topic.title()} Section",
            suggested_change=f"Add section covering '{topic}' based on content from {source_doc}",
            rationale=f"Topic '{topic}' is covered in {len(covered_in)} document(s) but missing from {target_doc}.",
            priority=priority,
            estimated_effort="30-60 minutes",
            confidence=0.75
        )

        return RemediationSuggestion(
            alert_id=alert_id,
            issue_type="coverage_gap",
            recommendation=action
        )

    def _suggest_numeric_remediation(self, conflict: Conflict, alert_id: str) -> RemediationSuggestion:
        """Suggest remediation for numeric conflicts."""
        doc_a = conflict.location_a.document_title
        doc_b = conflict.location_b.document_title
        val_a = conflict.value_a
        val_b = conflict.value_b

        # Determine which document should be updated
        authority_a = self._get_document_authority(doc_a)
        authority_b = self._get_document_authority(doc_b)

        # Extract numeric values for comparison
        num_a = self._extract_number(val_a)
        num_b = self._extract_number(val_b)

        # For security topics, stricter (usually higher) values win
        is_security_topic = any(kw in conflict.topic.lower() for kw in self.STRICTER_WINS_TOPICS)

        if authority_a > authority_b:
            target_doc = doc_b
            source_doc = doc_a
            correct_value = val_a
            wrong_value = val_b
            rationale = f"{doc_a} (authoritative source) specifies {val_a}."
        elif authority_b > authority_a:
            target_doc = doc_a
            source_doc = doc_b
            correct_value = val_b
            wrong_value = val_a
            rationale = f"{doc_b} (authoritative source) specifies {val_b}."
        elif is_security_topic and num_a is not None and num_b is not None:
            # For security, stricter (higher) value wins
            if num_a > num_b:
                target_doc = doc_b
                source_doc = doc_a
                correct_value = val_a
                wrong_value = val_b
                rationale = f"For security requirements, stricter value ({val_a}) should apply."
            else:
                target_doc = doc_a
                source_doc = doc_b
                correct_value = val_b
                wrong_value = val_a
                rationale = f"For security requirements, stricter value ({val_b}) should apply."
        else:
            # Default: escalate to document owner
            return self._suggest_escalation(conflict, alert_id,
                "Unable to determine authoritative value. Manual review required.")

        priority = (RemediationPriority.IMMEDIATE if conflict.severity == ConflictSeverity.CRITICAL
                   else RemediationPriority.HIGH if conflict.severity == ConflictSeverity.HIGH
                   else RemediationPriority.MEDIUM)

        action = RecommendedAction(
            action=RemediationAction.UPDATE_DOCUMENT,
            target_document=target_doc,
            target_section=conflict.location_a.section_title if target_doc == doc_a else conflict.location_b.section_title,
            suggested_change=f"Update '{wrong_value}' to '{correct_value}'",
            rationale=rationale,
            priority=priority,
            estimated_effort="5 minutes",
            confidence=0.85 if authority_a != authority_b else 0.70
        )

        return RemediationSuggestion(
            alert_id=alert_id,
            issue_type=f"numeric_contradiction",
            recommendation=action
        )

    def _suggest_duration_remediation(self, conflict: Conflict, alert_id: str) -> RemediationSuggestion:
        """Suggest remediation for duration conflicts."""
        # Duration conflicts follow similar logic to numeric
        return self._suggest_numeric_remediation(conflict, alert_id)

    def _suggest_policy_remediation(self, conflict: Conflict, alert_id: str) -> RemediationSuggestion:
        """Suggest remediation for policy contradictions."""
        doc_a = conflict.location_a.document_title
        doc_b = conflict.location_b.document_title

        authority_a = self._get_document_authority(doc_a)
        authority_b = self._get_document_authority(doc_b)

        if authority_a > authority_b:
            target_doc = doc_b
            source_doc = doc_a
            rationale = f"{doc_a} is the authoritative source for policy decisions."
        elif authority_b > authority_a:
            target_doc = doc_a
            source_doc = doc_b
            rationale = f"{doc_b} is the authoritative source for policy decisions."
        else:
            return self._suggest_escalation(conflict, alert_id,
                "Policy contradiction requires manual review to determine correct policy.")

        action = RecommendedAction(
            action=RemediationAction.CLARIFY_POLICY,
            target_document=target_doc,
            target_section=conflict.location_a.section_title if target_doc == doc_a else conflict.location_b.section_title,
            suggested_change=f"Align policy with {source_doc}: change from '{conflict.value_b if target_doc == doc_b else conflict.value_a}' to '{conflict.value_a if target_doc == doc_b else conflict.value_b}'",
            rationale=rationale,
            priority=RemediationPriority.HIGH,
            estimated_effort="15-30 minutes",
            confidence=0.75
        )

        return RemediationSuggestion(
            alert_id=alert_id,
            issue_type="policy_contradiction",
            recommendation=action
        )

    def _suggest_monetary_remediation(self, conflict: Conflict, alert_id: str) -> RemediationSuggestion:
        """Suggest remediation for monetary conflicts."""
        # Similar to numeric but may have different considerations
        return self._suggest_numeric_remediation(conflict, alert_id)

    def _suggest_generic_remediation(self, conflict: Conflict, alert_id: str) -> RemediationSuggestion:
        """Suggest generic remediation when type is unknown."""
        return self._suggest_escalation(conflict, alert_id,
            "Conflict type requires manual review.")

    def _suggest_escalation(self, conflict: Conflict, alert_id: str, reason: str) -> RemediationSuggestion:
        """Suggest escalation to document owner."""
        action = RecommendedAction(
            action=RemediationAction.ESCALATE_TO_OWNER,
            target_document=conflict.location_a.document_title,
            target_section=conflict.location_a.section_title,
            suggested_change="Review and resolve conflict manually",
            rationale=reason,
            priority=RemediationPriority.HIGH,
            estimated_effort="30-60 minutes",
            confidence=0.50,
            alternative_actions=[
                {
                    "action": "merge_documents",
                    "description": "Consider merging overlapping documents to prevent future conflicts"
                }
            ]
        )

        return RemediationSuggestion(
            alert_id=alert_id,
            issue_type=conflict.conflict_type.value,
            recommendation=action
        )

    def _get_document_authority(self, doc_title: str) -> int:
        """Get the authority score for a document based on its title."""
        doc_lower = doc_title.lower()

        max_authority = 0
        for keyword, authority in self.AUTHORITY_HIERARCHY.items():
            if keyword in doc_lower:
                max_authority = max(max_authority, authority)

        return max_authority if max_authority > 0 else 30  # Default authority

    def _extract_number(self, value: str) -> Optional[float]:
        """Extract a numeric value from a string."""
        match = re.search(r'(\d+(?:\.\d+)?)', value)
        if match:
            return float(match.group(1))
        return None

    def _select_target_for_gap(
        self,
        missing_from: list[str],
        covered_in: list[str],
        topic: str
    ) -> str:
        """Select which document should add coverage for a gap."""
        if not missing_from:
            return "new_document"

        # Prefer documents with higher authority
        best_doc = missing_from[0]
        best_authority = 0

        for doc in missing_from:
            authority = self._get_document_authority(doc)
            if authority > best_authority:
                best_authority = authority
                best_doc = doc

        return best_doc
