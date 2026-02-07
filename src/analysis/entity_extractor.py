"""Extract structured entities from document text.

Extracts:
- Monetary amounts ($100, $1,500.00, etc.)
- Time durations (30 days, 2 weeks, 3 months, etc.)
- Dates (December 31, 2023, 01/15/2024, etc.)
- Numeric requirements (12 characters, 14 characters, etc.)
- Percentages (50%, 10.5%, etc.)
- Policy references (Policy #123, Section 4.2, etc.)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    MONETARY = "monetary"
    DURATION = "duration"
    DATE = "date"
    NUMERIC_REQUIREMENT = "numeric_requirement"
    PERCENTAGE = "percentage"
    POLICY_REFERENCE = "policy_reference"


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    entity_type: EntityType
    normalized_value: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""
    metadata: dict = field(default_factory=dict)


class EntityExtractor:
    """Extract structured entities from text using regex patterns."""

    # Monetary patterns: $100, $1,500.00, USD 500, etc.
    MONETARY_PATTERNS = [
        r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:per|/)\s*(?:day|night|month|year|person|employee))?',
        r'(?:USD|EUR|GBP)\s*[\d,]+(?:\.\d{2})?',
        r'[\d,]+(?:\.\d{2})?\s*(?:dollars|USD)',
    ]

    # Duration patterns: 30 days, 2 weeks, 3 months, etc.
    DURATION_PATTERNS = [
        r'\d+\s*(?:day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes)',
        r'(?:one|two|three|four|five|six|seven|eight|nine|ten|thirty|sixty|ninety)\s*(?:day|days|week|weeks|month|months|year|years)',
        r'\d+\s*(?:business\s+)?day(?:s)?',
        r'(?:every|each)\s+\d+\s+(?:day|days|week|weeks|month|months)',
    ]

    # Date patterns: December 31, 2023, 01/15/2024, 2024-01-15, etc.
    DATE_PATTERNS = [
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}',
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{4}-\d{2}-\d{2}',
        r'(?:Q[1-4]|FY)\s*\d{4}',
    ]

    # Numeric requirement patterns: 12 characters, 14 characters, minimum 8, etc.
    NUMERIC_REQUIREMENT_PATTERNS = [
        r'(?:at\s+least\s+|minimum\s+|max(?:imum)?\s+|up\s+to\s+)?\d+\s*(?:character|characters|char|chars)',
        r'\d+\s*(?:password|passwords)',
        r'(?:minimum|maximum|at\s+least|up\s+to)\s+\d+',
        r'\d+\s*(?:attempt|attempts|try|tries)',
        r'\d+\s*(?:level|levels|tier|tiers)',
    ]

    # Percentage patterns: 50%, 10.5%, etc.
    PERCENTAGE_PATTERNS = [
        r'\d+(?:\.\d+)?%',
        r'\d+(?:\.\d+)?\s*percent',
    ]

    # Policy reference patterns: Policy #123, Section 4.2, Article III, etc.
    POLICY_REFERENCE_PATTERNS = [
        r'(?:Policy|Section|Article|Clause|Paragraph|Chapter)\s*(?:#|No\.?|Number)?\s*[\d.]+(?:\([a-z]\))?',
        r'(?:Policy|Section|Article)\s+[IVX]+',
        r'Appendix\s+[A-Z]',
    ]

    def __init__(self):
        """Initialize the entity extractor with compiled patterns."""
        self._patterns = {
            EntityType.MONETARY: [re.compile(p, re.IGNORECASE) for p in self.MONETARY_PATTERNS],
            EntityType.DURATION: [re.compile(p, re.IGNORECASE) for p in self.DURATION_PATTERNS],
            EntityType.DATE: [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS],
            EntityType.NUMERIC_REQUIREMENT: [re.compile(p, re.IGNORECASE) for p in self.NUMERIC_REQUIREMENT_PATTERNS],
            EntityType.PERCENTAGE: [re.compile(p, re.IGNORECASE) for p in self.PERCENTAGE_PATTERNS],
            EntityType.POLICY_REFERENCE: [re.compile(p, re.IGNORECASE) for p in self.POLICY_REFERENCE_PATTERNS],
        }

    def extract(self, text: str) -> list[Entity]:
        """Extract all entities from text.

        Args:
            text: The text to extract entities from.

        Returns:
            List of extracted Entity objects.
        """
        if not text or not text.strip():
            return []

        entities = []
        seen_spans = set()  # Avoid duplicate extractions

        for entity_type, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    span = (match.start(), match.end())

                    # Skip if we've already extracted this span
                    if span in seen_spans:
                        continue

                    # Skip if this span overlaps with an existing extraction
                    overlaps = any(
                        not (span[1] <= existing[0] or span[0] >= existing[1])
                        for existing in seen_spans
                    )
                    if overlaps:
                        continue

                    seen_spans.add(span)

                    # Extract context (50 chars before and after)
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end].strip()

                    entity = Entity(
                        text=match.group().strip(),
                        entity_type=entity_type,
                        normalized_value=self._normalize(match.group(), entity_type),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=context,
                    )
                    entities.append(entity)

        # Sort by position in text
        entities.sort(key=lambda e: e.start_pos)
        return entities

    def extract_by_type(self, text: str, entity_type: EntityType) -> list[Entity]:
        """Extract entities of a specific type.

        Args:
            text: The text to extract from.
            entity_type: The type of entity to extract.

        Returns:
            List of entities of the specified type.
        """
        all_entities = self.extract(text)
        return [e for e in all_entities if e.entity_type == entity_type]

    def _normalize(self, text: str, entity_type: EntityType) -> str:
        """Normalize entity value for comparison.

        Args:
            text: The raw extracted text.
            entity_type: The type of entity.

        Returns:
            Normalized string value.
        """
        text = text.strip().lower()

        if entity_type == EntityType.MONETARY:
            # Extract numeric value
            numbers = re.findall(r'[\d,]+(?:\.\d+)?', text)
            if numbers:
                return numbers[0].replace(',', '')

        elif entity_type == EntityType.DURATION:
            # Normalize to days
            text_lower = text.lower()
            numbers = re.findall(r'\d+', text)
            if numbers:
                num = int(numbers[0])
                if 'year' in text_lower:
                    return str(num * 365)
                elif 'month' in text_lower:
                    return str(num * 30)
                elif 'week' in text_lower:
                    return str(num * 7)
                else:
                    return str(num)
            # Handle word numbers
            word_to_num = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'thirty': 30, 'sixty': 60, 'ninety': 90
            }
            for word, num in word_to_num.items():
                if word in text_lower:
                    if 'year' in text_lower:
                        return str(num * 365)
                    elif 'month' in text_lower:
                        return str(num * 30)
                    elif 'week' in text_lower:
                        return str(num * 7)
                    else:
                        return str(num)

        elif entity_type == EntityType.NUMERIC_REQUIREMENT:
            numbers = re.findall(r'\d+', text)
            if numbers:
                return numbers[0]

        elif entity_type == EntityType.PERCENTAGE:
            numbers = re.findall(r'[\d.]+', text)
            if numbers:
                return numbers[0]

        return text

    def find_conflicting_values(
        self,
        entities_a: list[Entity],
        entities_b: list[Entity],
        entity_type: EntityType,
    ) -> list[tuple[Entity, Entity]]:
        """Find entities of the same type with different values.

        Useful for detecting conflicts between documents.

        Args:
            entities_a: Entities from document A.
            entities_b: Entities from document B.
            entity_type: The type of entity to compare.

        Returns:
            List of (entity_a, entity_b) tuples that have conflicting values.
        """
        a_of_type = [e for e in entities_a if e.entity_type == entity_type]
        b_of_type = [e for e in entities_b if e.entity_type == entity_type]

        conflicts = []
        for ea in a_of_type:
            for eb in b_of_type:
                if ea.normalized_value and eb.normalized_value:
                    if ea.normalized_value != eb.normalized_value:
                        # Check if contexts are similar (same topic)
                        if self._contexts_related(ea.context, eb.context):
                            conflicts.append((ea, eb))

        return conflicts

    def _contexts_related(self, context_a: str, context_b: str) -> bool:
        """Check if two contexts discuss related topics.

        Simple keyword overlap check.
        """
        # Extract significant words (3+ chars, no stopwords)
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'has', 'her', 'was', 'one', 'our', 'out'}

        def get_words(text: str) -> set[str]:
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            return {w for w in words if w not in stopwords}

        words_a = get_words(context_a)
        words_b = get_words(context_b)

        if not words_a or not words_b:
            return False

        overlap = len(words_a & words_b)
        min_size = min(len(words_a), len(words_b))

        # At least 20% word overlap
        return overlap >= max(2, min_size * 0.2)
