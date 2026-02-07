#!/usr/bin/env python3
"""Ingest documents into DocOps Agent."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import ingest_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_file(file_path: Path) -> bool:
    """Ingest a single file."""
    try:
        result = ingest_document(file_path)
        logger.info(
            f"Successfully ingested: {file_path.name} "
            f"(ID: {result['document_id']}, {result['chunk_count']} chunks)"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to ingest {file_path.name}: {e}")
        return False


def ingest_directory(dir_path: Path, recursive: bool = False) -> tuple[int, int]:
    """Ingest all documents in a directory."""
    extensions = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}
    success_count = 0
    failure_count = 0

    pattern = "**/*" if recursive else "*"

    for file_path in dir_path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            if ingest_file(file_path):
                success_count += 1
            else:
                failure_count += 1

    return success_count, failure_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest documents into DocOps Agent")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to ingest",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively process directories",
    )

    args = parser.parse_args()

    total_success = 0
    total_failure = 0

    for path_str in args.paths:
        path = Path(path_str)

        if not path.exists():
            logger.error(f"Path not found: {path}")
            total_failure += 1
            continue

        if path.is_file():
            if ingest_file(path):
                total_success += 1
            else:
                total_failure += 1
        elif path.is_dir():
            success, failure = ingest_directory(path, args.recursive)
            total_success += success
            total_failure += failure
        else:
            logger.error(f"Invalid path: {path}")
            total_failure += 1

    # Print summary
    logger.info(f"Ingestion complete: {total_success} succeeded, {total_failure} failed")

    if total_failure > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
