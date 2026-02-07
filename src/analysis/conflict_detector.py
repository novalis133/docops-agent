"""Detect conflicts and inconsistencies across documents.

Uses Elasticsearch to find overlapping content, then analyzes
for contradictions in:
- Numeric values (12 chars vs 14 chars)
- Time durations (30 days vs 60 days)
- Monetary amounts ($100 vs $150)
- Policy statements (allowed vs prohibited)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from elasticsearch import Elasticsearch

from .entity_extractor import EntityExtractor, EntityType, Entity


class ConflictSeverity(str, Enum):
    """Severity levels for detected conflicts."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConflictType(str, Enum):
    """Types of conflicts that can be detected."""
    NUMERIC = "numeric"
    DURATION = "duration"
    MONETARY = "monetary"
    POLICY = "policy"
    DATE = "date"


@dataclass
class ConflictLocation:
    """Location of a conflict within a document."""
    document_id: str
    document_title: str
    chunk_id: str
    section_title: str
    content_snippet: str
    entity: Optional[Entity] = None


@dataclass
class Conflict:
    """Represents a detected conflict between documents."""
    id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    location_a: ConflictLocation
    location_b: ConflictLocation
    value_a: str
    value_b: str
    topic: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


class ConflictDetector:
    """Detect conflicts between documents in Elasticsearch."""

    # Keywords that suggest contradictory policies
    CONTRADICTION_PAIRS = [
        ("required", "optional"),
        ("mandatory", "voluntary"),
        ("must", "may"),
        ("allowed", "prohibited"),
        ("permitted", "forbidden"),
        ("yes", "no"),
        ("always", "never"),
        ("all", "none"),
    ]

    def __init__(
        self,
        es_client: Optional[Elasticsearch] = None,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        chunks_index: str = "docops-chunks",
        documents_index: str = "docops-documents",
    ):
        """Initialize the conflict detector.

        Args:
            es_client: Existing Elasticsearch client (optional).
            host: ES host.
            port: ES port.
            scheme: http or https.
            chunks_index: Name of the chunks index.
            documents_index: Name of the documents index.
        """
        if es_client:
            self.es = es_client
        else:
            self.es = Elasticsearch(f"{scheme}://{host}:{port}")

        self.chunks_index = chunks_index
        self.documents_index = documents_index
        self.entity_extractor = EntityExtractor()
        self._conflict_counter = 0

    def detect_conflicts(self, topic: Optional[str] = None) -> list[Conflict]:
        """Detect conflicts across all documents or for a specific topic.

        Args:
            topic: Optional topic to focus on (e.g., "password", "remote work").

        Returns:
            List of detected Conflict objects.
        """
        conflicts = []

        # Get all chunks, optionally filtered by topic
        if topic:
            query = {
                "query": {
                    "match": {
                        "content": topic
                    }
                },
                "size": 100
            }
        else:
            query = {
                "query": {"match_all": {}},
                "size": 500
            }

        result = self.es.search(index=self.chunks_index, body=query)
        chunks = result["hits"]["hits"]

        if len(chunks) < 2:
            return []

        # Group chunks by document
        doc_chunks: dict[str, list[dict]] = {}
        for chunk in chunks:
            doc_id = chunk["_source"].get("document_id", "")
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(chunk)

        # Compare chunks across different documents
        doc_ids = list(doc_chunks.keys())
        for i, doc_id_a in enumerate(doc_ids):
            for doc_id_b in doc_ids[i + 1:]:
                doc_conflicts = self._compare_documents(
                    doc_chunks[doc_id_a],
                    doc_chunks[doc_id_b],
                    topic
                )
                conflicts.extend(doc_conflicts)

        return conflicts

    def detect_all_conflicts(self) -> list[Conflict]:
        """Scan the entire corpus for conflicts.

        Returns:
            List of all detected conflicts.
        """
        # First, find topics that appear in multiple documents
        topics = self._find_common_topics()

        all_conflicts = []
        seen_conflict_keys = set()

        for topic in topics:
            topic_conflicts = self.detect_conflicts(topic=topic)
            for conflict in topic_conflicts:
                # Deduplicate by document pair and values
                key = (
                    min(conflict.location_a.document_id, conflict.location_b.document_id),
                    max(conflict.location_a.document_id, conflict.location_b.document_id),
                    conflict.value_a,
                    conflict.value_b,
                )
                if key not in seen_conflict_keys:
                    seen_conflict_keys.add(key)
                    all_conflicts.append(conflict)

        return all_conflicts

    def _find_common_topics(self) -> list[str]:
        """Find topics that appear in multiple documents.

        Returns:
            List of topic keywords.
        """
        # Get significant terms across the corpus
        agg_query = {
            "size": 0,
            "aggs": {
                "common_terms": {
                    "significant_terms": {
                        "field": "content",
                        "size": 50
                    }
                }
            }
        }

        try:
            result = self.es.search(index=self.chunks_index, body=agg_query)
            buckets = result.get("aggregations", {}).get("common_terms", {}).get("buckets", [])
            return [b["key"] for b in buckets if len(b["key"]) > 3]
        except Exception:
            # Fallback to common policy topics
            return [
                "password", "security", "remote", "work", "expense",
                "leave", "policy", "data", "retention", "access"
            ]

    def _compare_documents(
        self,
        chunks_a: list[dict],
        chunks_b: list[dict],
        topic: Optional[str] = None
    ) -> list[Conflict]:
        """Compare chunks between two documents for conflicts.

        Args:
            chunks_a: Chunks from document A.
            chunks_b: Chunks from document B.
            topic: Optional topic filter.

        Returns:
            List of conflicts between the documents.
        """
        conflicts = []

        for chunk_a in chunks_a:
            content_a = chunk_a["_source"].get("content", "")
            entities_a = self.entity_extractor.extract(content_a)

            for chunk_b in chunks_b:
                content_b = chunk_b["_source"].get("content", "")

                # Skip if contents don't seem related
                if topic and topic.lower() not in content_a.lower() and topic.lower() not in content_b.lower():
                    continue

                if not self._contents_related(content_a, content_b):
                    continue

                entities_b = self.entity_extractor.extract(content_b)

                # Check for numeric conflicts
                numeric_conflicts = self._find_numeric_conflicts(
                    chunk_a, chunk_b, entities_a, entities_b, topic
                )
                conflicts.extend(numeric_conflicts)

                # Check for duration conflicts
                duration_conflicts = self._find_duration_conflicts(
                    chunk_a, chunk_b, entities_a, entities_b, topic
                )
                conflicts.extend(duration_conflicts)

                # Check for policy contradictions
                policy_conflicts = self._find_policy_contradictions(
                    chunk_a, chunk_b, content_a, content_b, topic
                )
                conflicts.extend(policy_conflicts)

        return conflicts

    def _contents_related(self, content_a: str, content_b: str) -> bool:
        """Check if two content strings discuss related topics."""
        # Extract significant words
        def get_words(text: str) -> set[str]:
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'would', 'should', 'could', 'their', 'there', 'these', 'those', 'about', 'which', 'when', 'where', 'what', 'your', 'more', 'other', 'into', 'only', 'also', 'such', 'than', 'then', 'some', 'make', 'like', 'just', 'over', 'even', 'most', 'after', 'before', 'between'}
            return {w for w in words if w not in stopwords}

        words_a = get_words(content_a)
        words_b = get_words(content_b)

        if not words_a or not words_b:
            return False

        overlap = len(words_a & words_b)
        return overlap >= 3

    def _find_numeric_conflicts(
        self,
        chunk_a: dict,
        chunk_b: dict,
        entities_a: list[Entity],
        entities_b: list[Entity],
        topic: Optional[str]
    ) -> list[Conflict]:
        """Find conflicting numeric requirements."""
        conflicts = []

        numeric_a = [e for e in entities_a if e.entity_type == EntityType.NUMERIC_REQUIREMENT]
        numeric_b = [e for e in entities_b if e.entity_type == EntityType.NUMERIC_REQUIREMENT]

        for ea in numeric_a:
            for eb in numeric_b:
                if ea.normalized_value and eb.normalized_value:
                    if ea.normalized_value != eb.normalized_value:
                        # Check if they're discussing the same thing
                        if self._same_topic_context(ea.context, eb.context):
                            conflict = self._create_conflict(
                                chunk_a, chunk_b,
                                ConflictType.NUMERIC,
                                ea, eb,
                                topic or self._extract_topic(ea.context, eb.context)
                            )
                            conflicts.append(conflict)

        return conflicts

    def _find_duration_conflicts(
        self,
        chunk_a: dict,
        chunk_b: dict,
        entities_a: list[Entity],
        entities_b: list[Entity],
        topic: Optional[str]
    ) -> list[Conflict]:
        """Find conflicting time durations."""
        conflicts = []

        duration_a = [e for e in entities_a if e.entity_type == EntityType.DURATION]
        duration_b = [e for e in entities_b if e.entity_type == EntityType.DURATION]

        for ea in duration_a:
            for eb in duration_b:
                if ea.normalized_value and eb.normalized_value:
                    # Compare normalized values (in days)
                    try:
                        days_a = int(ea.normalized_value)
                        days_b = int(eb.normalized_value)
                        if days_a != days_b and self._same_topic_context(ea.context, eb.context):
                            conflict = self._create_conflict(
                                chunk_a, chunk_b,
                                ConflictType.DURATION,
                                ea, eb,
                                topic or self._extract_topic(ea.context, eb.context)
                            )
                            conflicts.append(conflict)
                    except ValueError:
                        continue

        return conflicts

    def _find_policy_contradictions(
        self,
        chunk_a: dict,
        chunk_b: dict,
        content_a: str,
        content_b: str,
        topic: Optional[str]
    ) -> list[Conflict]:
        """Find contradictory policy statements."""
        conflicts = []

        content_a_lower = content_a.lower()
        content_b_lower = content_b.lower()

        for word_a, word_b in self.CONTRADICTION_PAIRS:
            # Check if A has word_a and B has word_b for same topic
            if word_a in content_a_lower and word_b in content_b_lower:
                # Extract the sentence containing each word
                sentence_a = self._extract_sentence(content_a, word_a)
                sentence_b = self._extract_sentence(content_b, word_b)

                if sentence_a and sentence_b and self._same_topic_context(sentence_a, sentence_b):
                    self._conflict_counter += 1
                    conflict = Conflict(
                        id=f"conflict-policy-{self._conflict_counter}",
                        conflict_type=ConflictType.POLICY,
                        severity=ConflictSeverity.MEDIUM,
                        description=f"Policy contradiction: '{word_a}' vs '{word_b}'",
                        location_a=ConflictLocation(
                            document_id=chunk_a["_source"].get("document_id", ""),
                            document_title=chunk_a["_source"].get("document_title", ""),
                            chunk_id=chunk_a["_id"],
                            section_title=chunk_a["_source"].get("section_title", ""),
                            content_snippet=sentence_a[:200],
                        ),
                        location_b=ConflictLocation(
                            document_id=chunk_b["_source"].get("document_id", ""),
                            document_title=chunk_b["_source"].get("document_title", ""),
                            chunk_id=chunk_b["_id"],
                            section_title=chunk_b["_source"].get("section_title", ""),
                            content_snippet=sentence_b[:200],
                        ),
                        value_a=word_a,
                        value_b=word_b,
                        topic=topic or "policy",
                    )
                    conflicts.append(conflict)

            # Also check reverse
            if word_b in content_a_lower and word_a in content_b_lower:
                sentence_a = self._extract_sentence(content_a, word_b)
                sentence_b = self._extract_sentence(content_b, word_a)

                if sentence_a and sentence_b and self._same_topic_context(sentence_a, sentence_b):
                    self._conflict_counter += 1
                    conflict = Conflict(
                        id=f"conflict-policy-{self._conflict_counter}",
                        conflict_type=ConflictType.POLICY,
                        severity=ConflictSeverity.MEDIUM,
                        description=f"Policy contradiction: '{word_b}' vs '{word_a}'",
                        location_a=ConflictLocation(
                            document_id=chunk_a["_source"].get("document_id", ""),
                            document_title=chunk_a["_source"].get("document_title", ""),
                            chunk_id=chunk_a["_id"],
                            section_title=chunk_a["_source"].get("section_title", ""),
                            content_snippet=sentence_a[:200],
                        ),
                        location_b=ConflictLocation(
                            document_id=chunk_b["_source"].get("document_id", ""),
                            document_title=chunk_b["_source"].get("document_title", ""),
                            chunk_id=chunk_b["_id"],
                            section_title=chunk_b["_source"].get("section_title", ""),
                            content_snippet=sentence_b[:200],
                        ),
                        value_a=word_b,
                        value_b=word_a,
                        topic=topic or "policy",
                    )
                    conflicts.append(conflict)

        return conflicts

    def _same_topic_context(self, context_a: str, context_b: str) -> bool:
        """Check if two contexts are discussing the same topic."""
        # Extract key nouns and look for overlap
        def get_key_words(text: str) -> set[str]:
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            # Filter to likely topic words
            stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'would', 'should', 'could', 'their', 'there', 'these', 'those', 'about', 'which', 'when', 'where', 'what', 'your', 'more', 'other', 'into', 'only', 'also', 'such', 'than', 'then', 'some', 'make', 'like', 'just', 'over', 'even', 'most', 'after', 'before', 'between', 'must', 'shall', 'need', 'each', 'every', 'following'}
            return {w for w in words if w not in stopwords}

        words_a = get_key_words(context_a)
        words_b = get_key_words(context_b)

        if not words_a or not words_b:
            return False

        overlap = words_a & words_b
        return len(overlap) >= 2

    def _extract_sentence(self, text: str, word: str) -> Optional[str]:
        """Extract the sentence containing a word."""
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if word.lower() in sentence.lower():
                return sentence.strip()
        return None

    def _extract_topic(self, context_a: str, context_b: str) -> str:
        """Extract the common topic from two contexts."""
        def get_key_words(text: str) -> set[str]:
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'would', 'should', 'could', 'their', 'there', 'these', 'those', 'about', 'which', 'when', 'where', 'what', 'your', 'more', 'other', 'into', 'must', 'shall'}
            return {w for w in words if w not in stopwords}

        words_a = get_key_words(context_a)
        words_b = get_key_words(context_b)
        common = words_a & words_b

        if common:
            return max(common, key=len)
        return "general"

    def _create_conflict(
        self,
        chunk_a: dict,
        chunk_b: dict,
        conflict_type: ConflictType,
        entity_a: Entity,
        entity_b: Entity,
        topic: str
    ) -> Conflict:
        """Create a Conflict object from detected entities."""
        self._conflict_counter += 1

        # Determine severity based on conflict type and magnitude
        severity = self._determine_severity(conflict_type, entity_a, entity_b)

        return Conflict(
            id=f"conflict-{conflict_type.value}-{self._conflict_counter}",
            conflict_type=conflict_type,
            severity=severity,
            description=f"{conflict_type.value.title()} conflict: {entity_a.text} vs {entity_b.text}",
            location_a=ConflictLocation(
                document_id=chunk_a["_source"].get("document_id", ""),
                document_title=chunk_a["_source"].get("document_title", ""),
                chunk_id=chunk_a["_id"],
                section_title=chunk_a["_source"].get("section_title", ""),
                content_snippet=entity_a.context[:200],
                entity=entity_a,
            ),
            location_b=ConflictLocation(
                document_id=chunk_b["_source"].get("document_id", ""),
                document_title=chunk_b["_source"].get("document_title", ""),
                chunk_id=chunk_b["_id"],
                section_title=chunk_b["_source"].get("section_title", ""),
                content_snippet=entity_b.context[:200],
                entity=entity_b,
            ),
            value_a=entity_a.text,
            value_b=entity_b.text,
            topic=topic,
        )

    def _determine_severity(
        self,
        conflict_type: ConflictType,
        entity_a: Entity,
        entity_b: Entity
    ) -> ConflictSeverity:
        """Determine the severity of a conflict."""
        # Security-related topics are critical
        security_keywords = ['password', 'security', 'authentication', 'access', 'encryption']
        context = (entity_a.context + entity_b.context).lower()

        if any(kw in context for kw in security_keywords):
            return ConflictSeverity.CRITICAL

        # Large differences in values are high severity
        if entity_a.normalized_value and entity_b.normalized_value:
            try:
                val_a = float(entity_a.normalized_value)
                val_b = float(entity_b.normalized_value)
                diff_ratio = abs(val_a - val_b) / max(val_a, val_b)
                if diff_ratio > 0.5:
                    return ConflictSeverity.HIGH
                elif diff_ratio > 0.2:
                    return ConflictSeverity.MEDIUM
            except ValueError:
                pass

        return ConflictSeverity.LOW
