"""Analyze coverage gaps across documents.

Identifies:
- Topics covered in some documents but missing from related ones
- Required topics that are not addressed
- Incomplete coverage of regulatory/compliance areas
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from elasticsearch import Elasticsearch


class GapSeverity(str, Enum):
    """Severity levels for coverage gaps."""
    CRITICAL = "critical"  # Required topic completely missing
    HIGH = "high"  # Important topic missing from related doc
    MEDIUM = "medium"  # Topic inconsistently covered
    LOW = "low"  # Minor coverage gap


class GapType(str, Enum):
    """Types of coverage gaps."""
    MISSING_TOPIC = "missing_topic"  # Topic in A but not in related B
    INCOMPLETE_COVERAGE = "incomplete_coverage"  # Topic partially covered
    MISSING_REQUIREMENT = "missing_requirement"  # Required topic not found


@dataclass
class CoverageGap:
    """Represents a detected coverage gap."""
    id: str
    gap_type: GapType
    severity: GapSeverity
    topic: str
    description: str
    covered_in: list[str]  # Document titles where topic IS covered
    missing_from: list[str]  # Document titles where topic is MISSING
    evidence_snippets: list[str]  # Text snippets showing coverage
    recommended_action: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


class GapAnalyzer:
    """Analyze documents for coverage gaps."""

    # Topics that should typically be covered in policy documents
    EXPECTED_POLICY_TOPICS = {
        "security": [
            "password", "authentication", "access control", "encryption",
            "mfa", "multi-factor", "two-factor", "2fa"
        ],
        "hr": [
            "leave", "vacation", "sick", "termination", "onboarding",
            "performance", "compensation", "benefits"
        ],
        "compliance": [
            "data privacy", "gdpr", "retention", "audit", "training",
            "reporting", "confidentiality"
        ],
        "remote_work": [
            "equipment", "security", "communication", "availability",
            "workspace", "vpn", "network"
        ],
        "expense": [
            "travel", "lodging", "meals", "approval", "reimbursement",
            "receipts", "limits"
        ],
    }

    # Document type associations (which doc types should cover what)
    DOC_TYPE_COVERAGE = {
        "handbook": ["hr", "compliance"],
        "security": ["security", "compliance"],
        "remote": ["remote_work", "security"],
        "expense": ["expense", "compliance"],
        "policy": ["compliance"],
    }

    def __init__(
        self,
        es_client: Optional[Elasticsearch] = None,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        chunks_index: str = "docops-chunks",
        documents_index: str = "docops-documents",
    ):
        """Initialize the gap analyzer.

        Args:
            es_client: Existing ES client (optional).
            host: ES host.
            port: ES port.
            scheme: http or https.
            chunks_index: Name of chunks index.
            documents_index: Name of documents index.
        """
        if es_client:
            self.es = es_client
        else:
            self.es = Elasticsearch(f"{scheme}://{host}:{port}")

        self.chunks_index = chunks_index
        self.documents_index = documents_index
        self._gap_counter = 0

    def analyze_all(self) -> list[CoverageGap]:
        """Analyze the entire corpus for coverage gaps.

        Returns:
            List of detected coverage gaps.
        """
        gaps = []

        # Build topic coverage matrix
        coverage_matrix = self._build_coverage_matrix()

        # Find gaps between related documents
        cross_doc_gaps = self._find_cross_document_gaps(coverage_matrix)
        gaps.extend(cross_doc_gaps)

        # Find missing required topics
        requirement_gaps = self._find_missing_requirements(coverage_matrix)
        gaps.extend(requirement_gaps)

        return gaps

    def analyze_document(self, document_id: str) -> list[CoverageGap]:
        """Analyze a specific document for coverage gaps.

        Args:
            document_id: The document to analyze.

        Returns:
            List of coverage gaps related to this document.
        """
        # Get document info
        try:
            doc = self.es.get(index=self.documents_index, id=document_id)
            doc_title = doc["_source"].get("title", "")
        except Exception:
            return []

        # Build coverage matrix
        coverage_matrix = self._build_coverage_matrix()

        if doc_title not in coverage_matrix:
            return []

        gaps = []

        # Check what this document covers vs what related docs cover
        doc_coverage = coverage_matrix[doc_title]
        doc_type = self._infer_doc_type(doc_title)

        for other_title, other_coverage in coverage_matrix.items():
            if other_title == doc_title:
                continue

            other_type = self._infer_doc_type(other_title)

            # If documents are related (similar type), check for gaps
            if self._docs_related(doc_type, other_type):
                for topic in other_coverage:
                    if topic not in doc_coverage:
                        self._gap_counter += 1
                        gaps.append(CoverageGap(
                            id=f"gap-{self._gap_counter}",
                            gap_type=GapType.MISSING_TOPIC,
                            severity=self._topic_severity(topic),
                            topic=topic,
                            description=f"Topic '{topic}' is covered in '{other_title}' but not in '{doc_title}'",
                            covered_in=[other_title],
                            missing_from=[doc_title],
                            evidence_snippets=other_coverage.get(topic, [])[:2],
                            recommended_action=f"Consider adding '{topic}' coverage to '{doc_title}' for consistency",
                        ))

        return gaps

    def _build_coverage_matrix(self) -> dict[str, dict[str, list[str]]]:
        """Build a matrix of which topics each document covers.

        Returns:
            Dict mapping doc_title -> {topic -> [evidence_snippets]}
        """
        coverage: dict[str, dict[str, list[str]]] = {}

        # Get all documents
        doc_query = {
            "query": {"match_all": {}},
            "size": 100
        }
        result = self.es.search(index=self.documents_index, body=doc_query)
        documents = result["hits"]["hits"]

        for doc in documents:
            doc_id = doc["_id"]
            doc_title = doc["_source"].get("title", "Unknown")
            coverage[doc_title] = {}

            # Get document chunks
            chunk_query = {
                "query": {"term": {"document_id": doc_id}},
                "size": 50
            }
            chunks_result = self.es.search(index=self.chunks_index, body=chunk_query)
            chunks = chunks_result["hits"]["hits"]

            all_content = " ".join(
                chunk["_source"].get("content", "") for chunk in chunks
            ).lower()

            # Check coverage for each expected topic
            for category, keywords in self.EXPECTED_POLICY_TOPICS.items():
                for keyword in keywords:
                    if keyword.lower() in all_content:
                        topic = f"{category}:{keyword}"
                        if topic not in coverage[doc_title]:
                            coverage[doc_title][topic] = []

                        # Find evidence snippet
                        snippet = self._extract_topic_snippet(all_content, keyword)
                        if snippet:
                            coverage[doc_title][topic].append(snippet)

        return coverage

    def _find_cross_document_gaps(
        self,
        coverage_matrix: dict[str, dict[str, list[str]]]
    ) -> list[CoverageGap]:
        """Find topics covered in some docs but missing from related docs."""
        gaps = []
        doc_titles = list(coverage_matrix.keys())

        for i, title_a in enumerate(doc_titles):
            type_a = self._infer_doc_type(title_a)

            for title_b in doc_titles[i + 1:]:
                type_b = self._infer_doc_type(title_b)

                if not self._docs_related(type_a, type_b):
                    continue

                coverage_a = set(coverage_matrix[title_a].keys())
                coverage_b = set(coverage_matrix[title_b].keys())

                # Topics in A but not B
                missing_from_b = coverage_a - coverage_b
                for topic in missing_from_b:
                    if self._is_significant_gap(topic, type_b):
                        self._gap_counter += 1
                        gaps.append(CoverageGap(
                            id=f"gap-cross-{self._gap_counter}",
                            gap_type=GapType.MISSING_TOPIC,
                            severity=self._topic_severity(topic),
                            topic=topic.split(":")[-1] if ":" in topic else topic,
                            description=f"'{topic.split(':')[-1]}' is covered in '{title_a}' but not in '{title_b}'",
                            covered_in=[title_a],
                            missing_from=[title_b],
                            evidence_snippets=coverage_matrix[title_a].get(topic, [])[:2],
                            recommended_action=f"Review if '{title_b}' should include {topic.split(':')[-1]} coverage",
                        ))

                # Topics in B but not A
                missing_from_a = coverage_b - coverage_a
                for topic in missing_from_a:
                    if self._is_significant_gap(topic, type_a):
                        self._gap_counter += 1
                        gaps.append(CoverageGap(
                            id=f"gap-cross-{self._gap_counter}",
                            gap_type=GapType.MISSING_TOPIC,
                            severity=self._topic_severity(topic),
                            topic=topic.split(":")[-1] if ":" in topic else topic,
                            description=f"'{topic.split(':')[-1]}' is covered in '{title_b}' but not in '{title_a}'",
                            covered_in=[title_b],
                            missing_from=[title_a],
                            evidence_snippets=coverage_matrix[title_b].get(topic, [])[:2],
                            recommended_action=f"Review if '{title_a}' should include {topic.split(':')[-1]} coverage",
                        ))

        return gaps

    def _find_missing_requirements(
        self,
        coverage_matrix: dict[str, dict[str, list[str]]]
    ) -> list[CoverageGap]:
        """Find required topics that are not covered anywhere."""
        gaps = []

        # Get all covered topics across corpus
        all_covered = set()
        for doc_coverage in coverage_matrix.values():
            all_covered.update(doc_coverage.keys())

        # Check for completely missing required topics
        for category, keywords in self.EXPECTED_POLICY_TOPICS.items():
            for keyword in keywords:
                topic = f"{category}:{keyword}"
                if topic not in all_covered:
                    # Check if any doc should cover this
                    for doc_title in coverage_matrix:
                        doc_type = self._infer_doc_type(doc_title)
                        expected_categories = self.DOC_TYPE_COVERAGE.get(doc_type, [])

                        if category in expected_categories:
                            self._gap_counter += 1
                            gaps.append(CoverageGap(
                                id=f"gap-req-{self._gap_counter}",
                                gap_type=GapType.MISSING_REQUIREMENT,
                                severity=GapSeverity.HIGH,
                                topic=keyword,
                                description=f"Required topic '{keyword}' not found in any document",
                                covered_in=[],
                                missing_from=[doc_title],
                                evidence_snippets=[],
                                recommended_action=f"Add '{keyword}' coverage to '{doc_title}' or create dedicated policy",
                            ))
                            break  # Only report once per missing topic

        return gaps

    def _infer_doc_type(self, title: str) -> str:
        """Infer document type from title."""
        title_lower = title.lower()

        if "handbook" in title_lower:
            return "handbook"
        elif "security" in title_lower:
            return "security"
        elif "remote" in title_lower:
            return "remote"
        elif "expense" in title_lower:
            return "expense"
        elif "retention" in title_lower:
            return "compliance"
        elif "policy" in title_lower:
            return "policy"
        else:
            return "general"

    def _docs_related(self, type_a: str, type_b: str) -> bool:
        """Check if two document types are related."""
        # Direct match
        if type_a == type_b:
            return True

        # Related pairs
        related_pairs = [
            ("handbook", "policy"),
            ("handbook", "security"),
            ("handbook", "remote"),
            ("security", "remote"),
            ("security", "compliance"),
            ("policy", "compliance"),
        ]

        for pair in related_pairs:
            if (type_a in pair and type_b in pair):
                return True

        return False

    def _is_significant_gap(self, topic: str, doc_type: str) -> bool:
        """Check if a gap is significant for the document type."""
        if ":" in topic:
            category = topic.split(":")[0]
        else:
            category = topic

        expected = self.DOC_TYPE_COVERAGE.get(doc_type, [])
        return category in expected

    def _topic_severity(self, topic: str) -> GapSeverity:
        """Determine severity based on topic importance."""
        critical_keywords = ["security", "password", "authentication", "access", "encryption", "compliance"]
        high_keywords = ["data", "privacy", "retention", "audit", "mfa", "vpn"]

        topic_lower = topic.lower()

        if any(kw in topic_lower for kw in critical_keywords):
            return GapSeverity.HIGH  # Not CRITICAL - it's a gap, not a vulnerability

        if any(kw in topic_lower for kw in high_keywords):
            return GapSeverity.MEDIUM

        return GapSeverity.LOW

    def _extract_topic_snippet(self, content: str, keyword: str) -> Optional[str]:
        """Extract a snippet of content around a keyword."""
        keyword_lower = keyword.lower()
        idx = content.find(keyword_lower)

        if idx == -1:
            return None

        # Extract ~100 chars around the keyword
        start = max(0, idx - 50)
        end = min(len(content), idx + len(keyword) + 50)

        snippet = content[start:end].strip()

        # Clean up - find sentence boundaries
        if start > 0:
            # Find start of sentence
            sentence_start = snippet.find(". ")
            if sentence_start != -1 and sentence_start < 30:
                snippet = snippet[sentence_start + 2:]

        return f"...{snippet}..."
