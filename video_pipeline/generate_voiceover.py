#!/usr/bin/env python3
"""
Generate voiceover audio using edge-tts (free, no API key needed).
"""

import asyncio
from pathlib import Path
import edge_tts

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
AUDIO_PATH = OUTPUT_DIR / "voiceover.mp3"
DURATION_PATH = OUTPUT_DIR / "voiceover_duration.txt"

# Voice options: en-US-GuyNeural (professional male), en-US-JennyNeural (female)
VOICE = "en-US-GuyNeural"
RATE = "-5%"  # Slightly slower for clarity

# Voiceover script (approximately 3 minutes at 150 words/minute)
SCRIPT = """
Your company has a problem you don't even know about.

Your Employee Handbook says passwords need twelve characters. Your Security Policy says fourteen.

Which one is right? Until today, you'd never know.

Introducing DocOps Agent.

This isn't a chatbot. This is an intelligent agent that thinks.

Watch. I click one button, and DocOps launches a multi-step analysis.

First, it searches your entire document corpus using hybrid search, combining keyword matching with semantic understanding.

Then it compares related sections across documents, looking for conflicts that simple pattern-matching would miss.

Five tools. One agent. Zero manual work.

Here's what it found. Twenty-three issues across fifty documents. In under two minutes.

Look at this conflict. The Remote Work Policy says employees can work from anywhere. But the Data Security Policy requires customer data work on approved networks only.

A remote employee handling customer data would violate security policy, but there's no obvious keyword conflict.

DocOps understood the implication. That's semantic conflict detection. No other tool does this.

Under the hood, this is pure Elasticsearch power.

Hybrid search combines BM25 keyword matching with dense vector embeddings. Every document is chunked, embedded, and indexed for lightning-fast retrieval.

The agent uses six custom tools, all powered by Elasticsearch queries. Aggregations analyze conflict patterns. Runtime fields calculate staleness dynamically.

This is what Elasticsearch Agent Builder was made for. Multi-step reasoning. Tool orchestration. Real intelligence.

But finding problems is only half the battle.

DocOps generates compliance reports automatically. Conflicts grouped by severity. Staleness analysis. Coverage gaps.

Every alert includes remediation suggestions. Not just here's a problem, but here's how to fix it.

And everything is logged for audit trails. Timestamp. Action. Result. Ready for compliance review.

Twenty-three critical issues. Fifty documents analyzed. Two minutes.

A compliance team would spend weeks doing this manually. And they'd still miss the semantic conflicts.

DocOps Agent. Built with Elasticsearch Agent Builder. Intelligent document operations for the enterprise.

Find conflicts. Fix compliance. Sleep better.
"""


async def generate_voiceover():
    """Generate voiceover audio from script."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Generating voiceover with voice: {VOICE}")
    print(f"Script length: {len(SCRIPT.split())} words")

    # Create TTS communicate object
    communicate = edge_tts.Communicate(
        text=SCRIPT.strip(),
        voice=VOICE,
        rate=RATE,
    )

    # Save audio file
    print(f"Saving to: {AUDIO_PATH}")
    await communicate.save(str(AUDIO_PATH))

    # Calculate approximate duration (150 words/minute)
    word_count = len(SCRIPT.split())
    estimated_duration = word_count / 150 * 60  # seconds

    # Save duration info
    with open(DURATION_PATH, "w") as f:
        f.write(f"word_count: {word_count}\n")
        f.write(f"estimated_duration_seconds: {estimated_duration:.1f}\n")
        f.write(f"estimated_duration_formatted: {int(estimated_duration // 60)}:{int(estimated_duration % 60):02d}\n")

    print(f"Voiceover generated successfully!")
    print(f"Estimated duration: {int(estimated_duration // 60)}:{int(estimated_duration % 60):02d}")

    return AUDIO_PATH


if __name__ == "__main__":
    asyncio.run(generate_voiceover())
