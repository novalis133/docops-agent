"""Semantic Conflict Detection using LLM reasoning.

This is the KEY DIFFERENTIATOR — finding conflicts that require UNDERSTANDING,
not just pattern matching.

Examples of semantic conflicts that rule-based systems miss:
- Policy A allows remote work, Policy B requires secure networks for data access
- Policy A says "manager approval", Policy B says "director approval" for same action
- Policy A has 30-day notice period, Policy B allows "immediate termination"
- Policy A allows flexible hours, Policy B requires "core hours" attendance

These conflicts are invisible to keyword matching but cause real compliance failures.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class SemanticConflictType(str, Enum):
    """Types of semantic conflicts."""
    SCOPE = "scope"  # Different scope of applicability
    AUTHORITY = "authority"  # Different approval/decision chains
    PROCEDURAL = "procedural"  # Contradictory procedures
    TEMPORAL = "temporal"  # Conflicting timelines/deadlines
    CONDITIONAL = "conditional"  # Edge cases where both can't be followed
    IMPLICATION = "implication"  # Logical implications that conflict


class SemanticSeverity(str, Enum):
    """Severity of semantic conflicts."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SemanticConflict:
    """A semantic conflict detected by LLM reasoning."""
    id: str
    conflict_type: SemanticConflictType
    severity: SemanticSeverity
    description: str
    doc_a_id: str
    doc_a_title: str
    doc_a_section: str
    doc_a_content: str
    doc_b_id: str
    doc_b_title: str
    doc_b_section: str
    doc_b_content: str
    example_scenario: str
    affected_roles: list[str]
    recommendation: str
    confidence: float
    similarity_score: float
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "doc_a": {
                "id": self.doc_a_id,
                "title": self.doc_a_title,
                "section": self.doc_a_section,
                "content_snippet": self.doc_a_content[:300] + "..." if len(self.doc_a_content) > 300 else self.doc_a_content
            },
            "doc_b": {
                "id": self.doc_b_id,
                "title": self.doc_b_title,
                "section": self.doc_b_section,
                "content_snippet": self.doc_b_content[:300] + "..." if len(self.doc_b_content) > 300 else self.doc_b_content
            },
            "example_scenario": self.example_scenario,
            "affected_roles": self.affected_roles,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "similarity_score": self.similarity_score,
            "detected_at": self.detected_at.isoformat()
        }


class SemanticConflictDetector:
    """Detect semantic conflicts using LLM reasoning.

    This is the key differentiator — finding conflicts that require understanding,
    not just pattern matching.

    Flow:
    1. Pre-filter: Use embedding similarity to find semantically related sections
    2. Deep Analysis: Use LLM to reason about potential conflicts
    3. Validation: Verify findings and assign confidence scores
    """

    # System prompt for semantic conflict detection
    SYSTEM_PROMPT = """You are a compliance expert specializing in detecting policy conflicts.

Your task is to identify SEMANTIC conflicts — contradictions that require UNDERSTANDING
the meaning and implications of policies, not just matching keywords.

A semantic conflict exists when:
1. Two policies cannot both be followed simultaneously in some scenario
2. Following one policy would violate the spirit or letter of another
3. The policies have contradictory implications (even if not explicitly stated)
4. Different authorities, timelines, or requirements are specified for the same action
5. Edge cases exist where compliance with both is impossible

You must think like an employee trying to follow both policies and identify where they would get stuck."""

    # Analysis prompt template
    ANALYSIS_PROMPT = """Analyze these two document sections for SEMANTIC conflicts.

## Section A
**Document:** {doc_a_title}
**Section:** {section_a_title}
**Content:**
{section_a_content}

## Section B
**Document:** {doc_b_title}
**Section:** {section_b_title}
**Content:**
{section_b_content}

## Analysis Instructions

Think through these questions:
1. Can an employee follow BOTH policies simultaneously in ALL scenarios?
2. Are there edge cases where following A would violate B (or vice versa)?
3. Do they specify different authorities, timelines, or requirements for similar situations?
4. Are there implicit contradictions in what each policy assumes?

## Response Format

Respond with a JSON object (no markdown, just raw JSON):
{{
  "has_conflict": true or false,
  "conflict_type": "scope" | "authority" | "procedural" | "temporal" | "conditional" | "implication" | null,
  "severity": "critical" | "high" | "medium" | "low",
  "description": "Clear explanation of the semantic conflict",
  "example_scenario": "A concrete, specific example where an employee cannot follow both policies",
  "affected_roles": ["list", "of", "job", "roles", "affected"],
  "recommendation": "How to resolve this conflict",
  "confidence": 0.0 to 1.0,
  "reasoning": "Step-by-step reasoning that led to this conclusion"
}}

If no conflict exists, set has_conflict to false and explain in the reasoning why they are compatible."""

    # Keywords that suggest potential semantic conflicts (for pre-filtering boost)
    CONFLICT_INDICATORS = [
        # Authority
        ("manager", "director", "vp", "executive", "approval", "authorize"),
        # Location/Access
        ("remote", "office", "onsite", "location", "network", "secure"),
        # Time
        ("immediate", "notice", "days", "weeks", "deadline", "timeline"),
        # Requirements
        ("required", "optional", "must", "may", "should", "mandatory"),
        # Scope
        ("all employees", "department", "exempt", "non-exempt", "contractor"),
        # Data/Security
        ("pii", "confidential", "sensitive", "classified", "public"),
    ]

    def __init__(
        self,
        es_client: Optional[Elasticsearch] = None,
        host: str = "localhost",
        port: int = 9200,
        scheme: str = "http",
        chunks_index: str = "docops-chunks",
        llm_provider: str = "openai",
        similarity_threshold: float = 0.5,
    ):
        """Initialize the semantic conflict detector.

        Args:
            es_client: Elasticsearch client.
            host: ES host.
            port: ES port.
            scheme: http or https.
            chunks_index: Index containing document chunks.
            llm_provider: LLM provider (openai, anthropic, local).
            similarity_threshold: Minimum similarity to consider for analysis.
        """
        if es_client:
            self.es = es_client
        else:
            self.es = Elasticsearch(f"{scheme}://{host}:{port}")

        self.chunks_index = chunks_index
        self.llm_provider = llm_provider
        self.similarity_threshold = similarity_threshold
        self._conflict_counter = 0

    def detect_semantic_conflicts(
        self,
        doc_ids: Optional[list[str]] = None,
        topic: Optional[str] = None,
        deep_scan: bool = False,
        max_pairs: int = 50,
    ) -> list[SemanticConflict]:
        """Detect semantic conflicts across documents.

        Args:
            doc_ids: Optional list of document IDs to analyze.
            topic: Optional topic to focus on.
            deep_scan: If True, analyze more pairs (slower but thorough).
            max_pairs: Maximum number of section pairs to analyze.

        Returns:
            List of detected SemanticConflict objects.
        """
        logger.info(f"Starting semantic conflict detection (deep_scan={deep_scan})")

        # Step 1: Get relevant sections
        sections = self._get_sections(doc_ids, topic)
        logger.info(f"Found {len(sections)} sections to analyze")

        if len(sections) < 2:
            return []

        # Step 2: Pre-filter to find semantically related pairs
        pairs = self._pre_filter_pairs(sections, deep_scan, max_pairs)
        logger.info(f"Pre-filtered to {len(pairs)} candidate pairs")

        if not pairs:
            return []

        # Step 3: Analyze pairs with LLM
        conflicts = self._analyze_pairs_with_llm(pairs)
        logger.info(f"Detected {len(conflicts)} semantic conflicts")

        return conflicts

    def _get_sections(
        self,
        doc_ids: Optional[list[str]],
        topic: Optional[str]
    ) -> list[dict]:
        """Get document sections from Elasticsearch."""
        must_clauses = []

        if doc_ids:
            must_clauses.append({"terms": {"document_id": doc_ids}})

        if topic:
            must_clauses.append({"match": {"content": topic}})

        query = {
            "bool": {"must": must_clauses} if must_clauses else {"match_all": {}}
        }

        try:
            result = self.es.search(
                index=self.chunks_index,
                query=query,
                size=500,
                _source=["document_id", "document_title", "section_title", "content", "embedding"]
            )

            sections = []
            for hit in result["hits"]["hits"]:
                source = hit["_source"]
                sections.append({
                    "id": hit["_id"],
                    "doc_id": source.get("document_id", ""),
                    "doc_title": source.get("document_title", ""),
                    "section_title": source.get("section_title", ""),
                    "content": source.get("content", ""),
                    "embedding": source.get("embedding", []),
                })

            return sections

        except Exception as e:
            logger.error(f"Failed to get sections: {e}")
            return []

    def _pre_filter_pairs(
        self,
        sections: list[dict],
        deep_scan: bool,
        max_pairs: int
    ) -> list[dict]:
        """Pre-filter section pairs using embedding similarity.

        This reduces O(n²) comparisons to a manageable subset.
        """
        pairs = []
        threshold = self.similarity_threshold if not deep_scan else self.similarity_threshold * 0.7

        for i, section_a in enumerate(sections):
            for section_b in sections[i + 1:]:
                # Skip same document
                if section_a["doc_id"] == section_b["doc_id"]:
                    continue

                # Skip if either has no embedding
                if not section_a.get("embedding") or not section_b.get("embedding"):
                    # Fall back to keyword overlap check
                    if self._has_conflict_indicators(section_a["content"], section_b["content"]):
                        pairs.append({
                            "section_a": section_a,
                            "section_b": section_b,
                            "similarity": 0.5,  # Default similarity for keyword match
                        })
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(
                    section_a["embedding"],
                    section_b["embedding"]
                )

                # Boost similarity if conflict indicators present
                if self._has_conflict_indicators(section_a["content"], section_b["content"]):
                    similarity = min(1.0, similarity + 0.2)

                if similarity >= threshold:
                    pairs.append({
                        "section_a": section_a,
                        "section_b": section_b,
                        "similarity": similarity,
                    })

        # Sort by similarity and take top pairs
        pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return pairs[:max_pairs]

    def _has_conflict_indicators(self, content_a: str, content_b: str) -> bool:
        """Check if two sections have overlapping conflict indicator keywords."""
        content_a_lower = content_a.lower()
        content_b_lower = content_b.lower()

        for indicator_group in self.CONFLICT_INDICATORS:
            matches_a = sum(1 for word in indicator_group if word in content_a_lower)
            matches_b = sum(1 for word in indicator_group if word in content_b_lower)

            # If both sections have keywords from same indicator group
            if matches_a >= 1 and matches_b >= 1:
                return True

        return False

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a = np.array(vec_a)
            b = np.array(vec_b)

            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return float(dot_product / (norm_a * norm_b))
        except Exception:
            return 0.0

    def _analyze_pairs_with_llm(self, pairs: list[dict]) -> list[SemanticConflict]:
        """Analyze section pairs using LLM reasoning."""
        conflicts = []

        for pair in pairs:
            try:
                result = self._analyze_single_pair(pair)
                if result:
                    conflicts.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze pair: {e}")
                continue

        return conflicts

    def _analyze_single_pair(self, pair: dict) -> Optional[SemanticConflict]:
        """Analyze a single section pair for semantic conflicts."""
        section_a = pair["section_a"]
        section_b = pair["section_b"]
        similarity = pair["similarity"]

        # Build prompt
        prompt = self.ANALYSIS_PROMPT.format(
            doc_a_title=section_a["doc_title"],
            section_a_title=section_a["section_title"],
            section_a_content=section_a["content"][:2000],  # Limit content length
            doc_b_title=section_b["doc_title"],
            section_b_title=section_b["section_title"],
            section_b_content=section_b["content"][:2000],
        )

        # Call LLM
        response = self._call_llm(prompt)
        if not response:
            return None

        # Parse response
        try:
            result = self._parse_llm_response(response)
            if not result or not result.get("has_conflict"):
                return None

            self._conflict_counter += 1

            return SemanticConflict(
                id=f"semantic-{self._conflict_counter}",
                conflict_type=SemanticConflictType(result.get("conflict_type", "implication")),
                severity=SemanticSeverity(result.get("severity", "medium")),
                description=result.get("description", ""),
                doc_a_id=section_a["doc_id"],
                doc_a_title=section_a["doc_title"],
                doc_a_section=section_a["section_title"],
                doc_a_content=section_a["content"],
                doc_b_id=section_b["doc_id"],
                doc_b_title=section_b["doc_title"],
                doc_b_section=section_b["section_title"],
                doc_b_content=section_b["content"],
                example_scenario=result.get("example_scenario", ""),
                affected_roles=result.get("affected_roles", []),
                recommendation=result.get("recommendation", ""),
                confidence=result.get("confidence", 0.7),
                similarity_score=similarity,
            )

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM for analysis.

        Supports multiple providers with fallback to heuristic analysis.
        """
        # Try OpenAI
        if self.llm_provider == "openai":
            return self._call_openai(prompt)

        # Try Anthropic
        elif self.llm_provider == "anthropic":
            return self._call_anthropic(prompt)

        # Fallback to heuristic analysis
        else:
            return self._heuristic_analysis(prompt)

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, using heuristic analysis")
            return self._heuristic_analysis(prompt)

        try:
            import httpx

            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}, using heuristic analysis")
            return self._heuristic_analysis(prompt)

    def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Call Anthropic API."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, using heuristic analysis")
            return self._heuristic_analysis(prompt)

        try:
            import httpx

            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1000,
                    "system": self.SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"]

        except Exception as e:
            logger.warning(f"Anthropic call failed: {e}, using heuristic analysis")
            return self._heuristic_analysis(prompt)

    def _heuristic_analysis(self, prompt: str) -> str:
        """Fallback heuristic analysis when LLM is not available.

        Uses advanced rule-based detection for common semantic conflict patterns.
        """
        # Extract section contents from prompt
        section_a_match = re.search(r"\*\*Content:\*\*\n(.*?)\n\n## Section B", prompt, re.DOTALL)
        section_b_match = re.search(r"Section B.*?\*\*Content:\*\*\n(.*?)\n\n## Analysis", prompt, re.DOTALL)

        if not section_a_match or not section_b_match:
            return json.dumps({"has_conflict": False, "reasoning": "Could not parse sections"})

        content_a = section_a_match.group(1).lower()
        content_b = section_b_match.group(1).lower()

        # Check for location-based conflicts (remote vs secure location)
        if self._check_location_conflict(content_a, content_b):
            return json.dumps({
                "has_conflict": True,
                "conflict_type": "scope",
                "severity": "high",
                "description": "Location flexibility conflicts with location-restricted requirements",
                "example_scenario": "An employee working remotely cannot access resources that require on-site secure network access",
                "affected_roles": ["remote workers", "data handlers"],
                "recommendation": "Clarify which roles can work remotely and ensure secure remote access solutions",
                "confidence": 0.85,
                "reasoning": "Detected pattern: remote work policy vs secure location requirement"
            })

        # Check for authority conflicts (different approval chains)
        if self._check_authority_conflict(content_a, content_b):
            return json.dumps({
                "has_conflict": True,
                "conflict_type": "authority",
                "severity": "medium",
                "description": "Different approval authorities specified for similar actions",
                "example_scenario": "Employee unsure whether to seek manager or director approval for the same action",
                "affected_roles": ["all employees"],
                "recommendation": "Establish clear approval hierarchy and document which authority applies to each scenario",
                "confidence": 0.75,
                "reasoning": "Detected pattern: conflicting approval authority requirements"
            })

        # Check for timeline conflicts
        if self._check_timeline_conflict(content_a, content_b):
            return json.dumps({
                "has_conflict": True,
                "conflict_type": "temporal",
                "severity": "high",
                "description": "Conflicting timeline or notice period requirements",
                "example_scenario": "One policy requires 30 days notice while another allows immediate action for the same scenario",
                "affected_roles": ["all employees", "managers"],
                "recommendation": "Reconcile timeline requirements and specify which applies in each case",
                "confidence": 0.80,
                "reasoning": "Detected pattern: conflicting timeline/notice requirements"
            })

        # Check for scope conflicts (all vs some employees)
        if self._check_scope_conflict(content_a, content_b):
            return json.dumps({
                "has_conflict": True,
                "conflict_type": "scope",
                "severity": "medium",
                "description": "Different scopes of applicability for related policies",
                "example_scenario": "Policy applies to 'all employees' in one document but only 'exempt employees' in another",
                "affected_roles": ["non-exempt employees", "contractors"],
                "recommendation": "Clarify scope and ensure consistent applicability definitions",
                "confidence": 0.70,
                "reasoning": "Detected pattern: conflicting scope definitions"
            })

        return json.dumps({
            "has_conflict": False,
            "reasoning": "No semantic conflict patterns detected through heuristic analysis"
        })

    def _check_location_conflict(self, content_a: str, content_b: str) -> bool:
        """Check for remote work vs secure location conflicts."""
        remote_keywords = ["remote", "work from home", "any location", "flexible location", "wfh"]
        secure_keywords = ["secure network", "office only", "on-site", "onsite", "approved location", "company premises"]

        has_remote = any(kw in content_a for kw in remote_keywords) or any(kw in content_b for kw in remote_keywords)
        has_secure = any(kw in content_a for kw in secure_keywords) or any(kw in content_b for kw in secure_keywords)

        # Check if they're in opposite documents
        remote_in_a = any(kw in content_a for kw in remote_keywords)
        secure_in_b = any(kw in content_b for kw in secure_keywords)
        remote_in_b = any(kw in content_b for kw in remote_keywords)
        secure_in_a = any(kw in content_a for kw in secure_keywords)

        return (remote_in_a and secure_in_b) or (remote_in_b and secure_in_a)

    def _check_authority_conflict(self, content_a: str, content_b: str) -> bool:
        """Check for conflicting approval authorities."""
        authorities = ["manager", "director", "vp", "executive", "ceo", "hr", "supervisor", "team lead"]

        authorities_in_a = [auth for auth in authorities if auth in content_a]
        authorities_in_b = [auth for auth in authorities if auth in content_b]

        # Check if different authorities are mentioned for similar context
        if authorities_in_a and authorities_in_b:
            # If they mention different authorities and both have "approval" context
            if "approv" in content_a and "approv" in content_b:
                if set(authorities_in_a) != set(authorities_in_b):
                    return True

        return False

    def _check_timeline_conflict(self, content_a: str, content_b: str) -> bool:
        """Check for conflicting timeline requirements."""
        immediate = ["immediate", "instantly", "same day", "effective immediately"]
        notice_patterns = [r"\d+\s*days?\s*notice", r"\d+\s*weeks?\s*notice", r"\d+\s*months?\s*notice"]

        has_immediate = any(kw in content_a for kw in immediate) or any(kw in content_b for kw in immediate)
        has_notice = any(re.search(pat, content_a) or re.search(pat, content_b) for pat in notice_patterns)

        # Check if they're in opposite documents
        immediate_in_a = any(kw in content_a for kw in immediate)
        notice_in_b = any(re.search(pat, content_b) for pat in notice_patterns)
        immediate_in_b = any(kw in content_b for kw in immediate)
        notice_in_a = any(re.search(pat, content_a) for pat in notice_patterns)

        return (immediate_in_a and notice_in_b) or (immediate_in_b and notice_in_a)

    def _check_scope_conflict(self, content_a: str, content_b: str) -> bool:
        """Check for conflicting scope definitions."""
        universal = ["all employees", "everyone", "all staff", "entire organization"]
        limited = ["exempt only", "non-exempt", "contractors excluded", "full-time only", "department only"]

        universal_in_a = any(kw in content_a for kw in universal)
        limited_in_b = any(kw in content_b for kw in limited)
        universal_in_b = any(kw in content_b for kw in universal)
        limited_in_a = any(kw in content_a for kw in limited)

        return (universal_in_a and limited_in_b) or (universal_in_b and limited_in_a)

    def _parse_llm_response(self, response: str) -> Optional[dict]:
        """Parse LLM response JSON."""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = re.sub(r"```json?\n?", "", response)
                response = re.sub(r"```\n?$", "", response)

            return json.loads(response)

        except json.JSONDecodeError:
            # Try to find JSON object in response
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Could not parse LLM response as JSON: {response[:200]}...")
            return None
