"""Check for stale, outdated, or expired content in documents.

Detects:
- Explicit expiration dates that have passed
- Year references that are outdated (e.g., "2023" when current year is 2026)
- References to other documents that may have been updated
- "Last reviewed" dates that are too old
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from elasticsearch import Elasticsearch

from .entity_extractor import EntityExtractor, EntityType


class StalenessSeverity(str, Enum):
    """Severity levels for staleness issues."""
    CRITICAL = "critical"  # Explicitly expired
    HIGH = "high"  # Very outdated (2+ years)
    MEDIUM = "medium"  # Somewhat outdated (1-2 years)
    LOW = "low"  # Potentially outdated


class StalenessType(str, Enum):
    """Types of staleness that can be detected."""
    EXPIRED = "expired"  # Has explicit expiration date that passed
    OUTDATED_YEAR = "outdated_year"  # Year in title/content is old
    STALE_REVIEW = "stale_review"  # Last reviewed date is old
    REFERENCES_STALE = "references_stale"  # References another stale doc
    OLD_VERSION = "old_version"  # Version number suggests newer exists


@dataclass
class StalenessIssue:
    """Represents a detected staleness issue."""
    id: str
    document_id: str
    document_title: str
    staleness_type: StalenessType
    severity: StalenessSeverity
    description: str
    stale_reference: str  # The specific date/year/reference that's stale
    recommended_action: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    affected_sections: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class StalenessChecker:
    """Check documents for staleness and outdated content."""

    # Patterns for expiration language
    EXPIRATION_PATTERNS = [
        r'expires?\s+(?:on\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
        r'expir(?:es?|ation)\s*(?:date)?[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'valid\s+(?:until|through)\s+(\w+\s+\d{1,2},?\s+\d{4})',
        r'effective\s+(?:until|through)\s+(\w+\s+\d{1,2},?\s+\d{4})',
        r'(?:this\s+(?:policy|document)\s+)?expires?\s+(\w+\s+\d{1,2},?\s+\d{4})',
    ]

    # Patterns for last reviewed dates
    REVIEW_PATTERNS = [
        r'last\s+(?:reviewed|updated|revised)\s*[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
        r'(?:reviewed|updated|revised)\s+(?:on\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
        r'version\s+date[:\s]+(\w+\s+\d{1,2},?\s+\d{4})',
        r'as\s+of\s+(\w+\s+\d{1,2},?\s+\d{4})',
    ]

    # Year pattern in titles
    YEAR_IN_TITLE_PATTERN = re.compile(r'\b(20\d{2})\b')

    def __init__(
        self,
        es_client: Optional[Elasticsearch] = None,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        chunks_index: str = "docops-chunks",
        documents_index: str = "docops-documents",
        current_date: Optional[datetime] = None,
        stale_threshold_years: int = 2,
        review_threshold_months: int = 12,
    ):
        """Initialize the staleness checker.

        Args:
            es_client: Existing ES client (optional).
            host: ES host.
            port: ES port.
            scheme: http or https.
            chunks_index: Name of chunks index.
            documents_index: Name of documents index.
            current_date: Override current date for testing.
            stale_threshold_years: Years before content is considered outdated.
            review_threshold_months: Months before review date is considered stale.
        """
        if es_client:
            self.es = es_client
        else:
            self.es = Elasticsearch(f"{scheme}://{host}:{port}")

        self.chunks_index = chunks_index
        self.documents_index = documents_index
        self.current_date = current_date or datetime.utcnow()
        self.stale_threshold_years = stale_threshold_years
        self.review_threshold_months = review_threshold_months
        self.entity_extractor = EntityExtractor()
        self._issue_counter = 0

    def check_all_documents(self) -> list[StalenessIssue]:
        """Check all documents for staleness issues.

        Returns:
            List of detected staleness issues.
        """
        issues = []

        # Get all documents
        query = {
            "query": {"match_all": {}},
            "size": 100
        }
        result = self.es.search(index=self.documents_index, body=query)
        documents = result["hits"]["hits"]

        for doc in documents:
            doc_issues = self.check_document(
                doc["_id"],
                doc["_source"].get("title", ""),
            )
            issues.extend(doc_issues)

        return issues

    def check_document(self, document_id: str, document_title: str) -> list[StalenessIssue]:
        """Check a single document for staleness issues.

        Args:
            document_id: The document ID.
            document_title: The document title.

        Returns:
            List of staleness issues for this document.
        """
        issues = []

        # Check title for outdated year
        year_issue = self._check_title_year(document_id, document_title)
        if year_issue:
            issues.append(year_issue)

        # Get document chunks for content analysis
        chunk_query = {
            "query": {
                "term": {"document_id": document_id}
            },
            "size": 50
        }
        result = self.es.search(index=self.chunks_index, body=chunk_query)
        chunks = result["hits"]["hits"]

        all_content = " ".join(
            chunk["_source"].get("content", "") for chunk in chunks
        )

        # Check for expiration dates
        expiration_issues = self._check_expiration_dates(
            document_id, document_title, all_content, chunks
        )
        issues.extend(expiration_issues)

        # Check for stale review dates
        review_issues = self._check_review_dates(
            document_id, document_title, all_content
        )
        issues.extend(review_issues)

        # Check for references to potentially stale documents
        reference_issues = self._check_stale_references(
            document_id, document_title, all_content
        )
        issues.extend(reference_issues)

        return issues

    def _check_title_year(
        self,
        document_id: str,
        document_title: str
    ) -> Optional[StalenessIssue]:
        """Check if the document title contains an outdated year."""
        match = self.YEAR_IN_TITLE_PATTERN.search(document_title)
        if not match:
            return None

        year = int(match.group(1))
        current_year = self.current_date.year
        years_old = current_year - year

        if years_old < self.stale_threshold_years:
            return None

        self._issue_counter += 1
        severity = self._year_severity(years_old)

        return StalenessIssue(
            id=f"stale-year-{self._issue_counter}",
            document_id=document_id,
            document_title=document_title,
            staleness_type=StalenessType.OUTDATED_YEAR,
            severity=severity,
            description=f"Document title references year {year}, which is {years_old} years old",
            stale_reference=str(year),
            recommended_action=f"Review and update document for {current_year}, or confirm content is still current",
        )

    def _check_expiration_dates(
        self,
        document_id: str,
        document_title: str,
        content: str,
        chunks: list[dict]
    ) -> list[StalenessIssue]:
        """Check for explicit expiration dates that have passed."""
        issues = []

        for pattern_str in self.EXPIRATION_PATTERNS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(content):
                date_str = match.group(1)
                parsed_date = self._parse_date(date_str)

                if parsed_date and parsed_date < self.current_date:
                    self._issue_counter += 1
                    days_expired = (self.current_date - parsed_date).days

                    # Find which section contains this
                    affected_sections = self._find_affected_sections(
                        chunks, match.group(0)
                    )

                    issues.append(StalenessIssue(
                        id=f"stale-expired-{self._issue_counter}",
                        document_id=document_id,
                        document_title=document_title,
                        staleness_type=StalenessType.EXPIRED,
                        severity=ConflictSeverity.CRITICAL if days_expired > 365 else ConflictSeverity.HIGH,
                        description=f"Document expired on {date_str} ({days_expired} days ago)",
                        stale_reference=date_str,
                        recommended_action="Immediately review and either renew or retire this document",
                        affected_sections=affected_sections,
                    ))

        return issues

    def _check_review_dates(
        self,
        document_id: str,
        document_title: str,
        content: str
    ) -> list[StalenessIssue]:
        """Check for stale review/update dates."""
        issues = []

        for pattern_str in self.REVIEW_PATTERNS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(content):
                date_str = match.group(1)
                parsed_date = self._parse_date(date_str)

                if parsed_date:
                    months_since = (self.current_date - parsed_date).days / 30

                    if months_since > self.review_threshold_months:
                        self._issue_counter += 1

                        issues.append(StalenessIssue(
                            id=f"stale-review-{self._issue_counter}",
                            document_id=document_id,
                            document_title=document_title,
                            staleness_type=StalenessType.STALE_REVIEW,
                            severity=self._review_severity(months_since),
                            description=f"Last reviewed {date_str} ({int(months_since)} months ago)",
                            stale_reference=date_str,
                            recommended_action=f"Schedule review - document hasn't been reviewed in {int(months_since)} months",
                        ))

        return issues

    def _check_stale_references(
        self,
        document_id: str,
        document_title: str,
        content: str
    ) -> list[StalenessIssue]:
        """Check if document references other documents that are stale."""
        issues = []

        # Get all document titles to check references
        query = {
            "query": {"match_all": {}},
            "size": 100,
            "_source": ["title"]
        }
        result = self.es.search(index=self.documents_index, body=query)
        all_docs = result["hits"]["hits"]

        stale_doc_titles = []
        for doc in all_docs:
            title = doc["_source"].get("title", "")
            year_match = self.YEAR_IN_TITLE_PATTERN.search(title)
            if year_match:
                year = int(year_match.group(1))
                if self.current_date.year - year >= self.stale_threshold_years:
                    stale_doc_titles.append(title)

        # Check if current document references any stale documents
        content_lower = content.lower()
        for stale_title in stale_doc_titles:
            # Skip self-reference
            if stale_title.lower() == document_title.lower():
                continue

            # Check for reference (simplified - look for title keywords)
            title_words = [w for w in stale_title.lower().split() if len(w) > 3]
            if len(title_words) >= 2:
                if all(word in content_lower for word in title_words[:3]):
                    self._issue_counter += 1

                    issues.append(StalenessIssue(
                        id=f"stale-reference-{self._issue_counter}",
                        document_id=document_id,
                        document_title=document_title,
                        staleness_type=StalenessType.REFERENCES_STALE,
                        severity=StalenessSeverity.MEDIUM,
                        description=f"References potentially stale document: {stale_title}",
                        stale_reference=stale_title,
                        recommended_action=f"Verify reference to '{stale_title}' is still valid or update to current version",
                    ))

        return issues

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse a date string into a datetime object."""
        formats = [
            "%B %d, %Y",  # December 31, 2023
            "%B %d %Y",   # December 31 2023
            "%m/%d/%Y",   # 12/31/2023
            "%m-%d-%Y",   # 12-31-2023
            "%Y-%m-%d",   # 2023-12-31
            "%m/%d/%y",   # 12/31/23
        ]

        # Clean the date string
        date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)  # Remove ordinals
        date_str = date_str.strip()

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def _year_severity(self, years_old: int) -> StalenessSeverity:
        """Determine severity based on how many years old."""
        if years_old >= 4:
            return StalenessSeverity.CRITICAL
        elif years_old >= 3:
            return StalenessSeverity.HIGH
        elif years_old >= 2:
            return StalenessSeverity.MEDIUM
        else:
            return StalenessSeverity.LOW

    def _review_severity(self, months_since: float) -> StalenessSeverity:
        """Determine severity based on months since last review."""
        if months_since >= 36:
            return StalenessSeverity.CRITICAL
        elif months_since >= 24:
            return StalenessSeverity.HIGH
        elif months_since >= 12:
            return StalenessSeverity.MEDIUM
        else:
            return StalenessSeverity.LOW

    def _find_affected_sections(
        self,
        chunks: list[dict],
        search_text: str
    ) -> list[str]:
        """Find which sections contain the given text."""
        sections = []
        search_lower = search_text.lower()

        for chunk in chunks:
            content = chunk["_source"].get("content", "").lower()
            if search_lower in content:
                section = chunk["_source"].get("section_title", "")
                if section and section not in sections:
                    sections.append(section)

        return sections


# Alias for backward compatibility
ConflictSeverity = StalenessSeverity
