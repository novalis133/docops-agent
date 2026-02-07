"""Section-aware document chunker."""

import re
from dataclasses import dataclass, field
from typing import Optional

from .parser import ParsedDocument, ParsedSection


@dataclass
class Chunk:
    """Represents a chunk of document content."""

    id: str
    content: str
    document_id: str
    document_title: str
    section_title: str
    section_level: int
    chunk_index: int
    total_chunks_in_section: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Return the character count of this chunk."""
        return len(self.content)


class SectionChunker:
    """Chunk documents while respecting section boundaries."""

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 100,
        respect_sections: bool = True,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.respect_sections = respect_sections

        # Sentence boundary patterns
        self._sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        self._paragraph_break = re.compile(r"\n\s*\n")

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk a parsed document into smaller pieces."""
        chunks: list[Chunk] = []
        document_id = self._generate_document_id(document)

        for section in document.sections:
            section_chunks = self._chunk_section(section, document_id, document.title)
            chunks.extend(section_chunks)

        return chunks

    def chunk_text(
        self,
        text: str,
        document_id: str = "doc",
        document_title: str = "Document",
        section_title: str = "Content",
    ) -> list[Chunk]:
        """Chunk raw text into smaller pieces."""
        section = ParsedSection(title=section_title, content=text, level=1)
        return self._chunk_section(section, document_id, document_title)

    def _chunk_section(
        self, section: ParsedSection, document_id: str, document_title: str
    ) -> list[Chunk]:
        """Chunk a single section."""
        content = section.content.strip()

        if not content:
            return []

        # If content is small enough, return as single chunk
        if len(content) <= self.max_chunk_size:
            return [
                Chunk(
                    id=f"{document_id}_{section.title}_0",
                    content=content,
                    document_id=document_id,
                    document_title=document_title,
                    section_title=section.title,
                    section_level=section.level,
                    chunk_index=0,
                    total_chunks_in_section=1,
                    start_char=0,
                    end_char=len(content),
                    page_number=section.page_number,
                    metadata=section.metadata,
                )
            ]

        # Split into chunks respecting boundaries
        raw_chunks = self._split_content(content)
        chunks: list[Chunk] = []

        char_offset = 0
        for i, chunk_content in enumerate(raw_chunks):
            chunk_id = f"{document_id}_{self._sanitize_title(section.title)}_{i}"

            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=chunk_content,
                    document_id=document_id,
                    document_title=document_title,
                    section_title=section.title,
                    section_level=section.level,
                    chunk_index=i,
                    total_chunks_in_section=len(raw_chunks),
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_content),
                    page_number=section.page_number,
                    metadata=section.metadata,
                )
            )

            # Account for overlap in offset calculation
            char_offset += len(chunk_content) - self.overlap

        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks_in_section = len(chunks)

        return chunks

    def _split_content(self, content: str) -> list[str]:
        """Split content into chunks respecting sentence and paragraph boundaries."""
        chunks: list[str] = []

        # First, split by paragraphs
        paragraphs = self._paragraph_break.split(content)

        current_chunk: list[str] = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If paragraph alone exceeds max size, split by sentences
            if len(paragraph) > self.max_chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split long paragraph by sentences
                sentence_chunks = self._split_by_sentences(paragraph)
                chunks.extend(sentence_chunks)
            elif current_size + len(paragraph) + 2 <= self.max_chunk_size:
                # Add to current chunk
                current_chunk.append(paragraph)
                current_size += len(paragraph) + 2  # +2 for paragraph separator
            else:
                # Flush current chunk and start new one
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))

                # Add overlap from previous chunk
                if chunks and self.overlap > 0:
                    overlap_text = self._get_overlap_text(chunks[-1])
                    current_chunk = [overlap_text, paragraph] if overlap_text else [paragraph]
                    current_size = sum(len(p) for p in current_chunk) + 2
                else:
                    current_chunk = [paragraph]
                    current_size = len(paragraph)

        # Add remaining content
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # Filter out chunks that are too small (except if it's the only chunk)
        if len(chunks) > 1:
            chunks = [c for c in chunks if len(c) >= self.min_chunk_size]

        return chunks if chunks else [content]

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text by sentences when it exceeds max chunk size."""
        sentences = self._sentence_end.split(text)
        chunks: list[str] = []

        current_chunk: list[str] = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If single sentence exceeds max, split by words
            if len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                word_chunks = self._split_by_words(sentence)
                chunks.extend(word_chunks)
            elif current_size + len(sentence) + 1 <= self.max_chunk_size:
                current_chunk.append(sentence)
                current_size += len(sentence) + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                # Add overlap
                if chunks and self.overlap > 0:
                    overlap_text = self._get_overlap_text(chunks[-1])
                    current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                    current_size = sum(len(s) for s in current_chunk) + 1
                else:
                    current_chunk = [sentence]
                    current_size = len(sentence)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_by_words(self, text: str) -> list[str]:
        """Split text by words as a last resort."""
        words = text.split()
        chunks: list[str] = []

        current_chunk: list[str] = []
        current_size = 0

        for word in words:
            if current_size + len(word) + 1 <= self.max_chunk_size:
                current_chunk.append(word)
                current_size += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_overlap_text(self, previous_chunk: str) -> str:
        """Get overlap text from the end of the previous chunk."""
        if len(previous_chunk) <= self.overlap:
            return previous_chunk

        # Try to break at sentence boundary
        overlap_region = previous_chunk[-self.overlap * 2 :]
        sentences = self._sentence_end.split(overlap_region)

        if len(sentences) > 1:
            return sentences[-1].strip()

        # Fall back to word boundary
        words = overlap_region.split()
        result: list[str] = []
        size = 0

        for word in reversed(words):
            if size + len(word) + 1 <= self.overlap:
                result.insert(0, word)
                size += len(word) + 1
            else:
                break

        return " ".join(result)

    def _generate_document_id(self, document: ParsedDocument) -> str:
        """Generate a unique document ID."""
        import hashlib

        content = f"{document.filename}:{document.title}:{len(document.raw_text)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _sanitize_title(self, title: str) -> str:
        """Sanitize a title for use in IDs."""
        sanitized = re.sub(r"[^\w\s-]", "", title.lower())
        sanitized = re.sub(r"[\s_-]+", "_", sanitized)
        return sanitized[:50]
