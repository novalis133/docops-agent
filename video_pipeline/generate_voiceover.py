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

# Voiceover script matching the demo video flow
SCRIPT = """
Welcome to DocOps Agent — AI-powered document analysis with multi-step reasoning.

The platform detects three types of issues: Conflict Detection finds contradictions between documents. Staleness Analysis identifies outdated content. Coverage Gaps discovers inconsistent topic coverage.

Here's how it works in four simple steps. Upload your documents. Ask the AI agent questions. View conflicts side by side. Export reports and resolve issues.

Let's see it in action. I'll load our demo corpus — twenty-five enterprise policy documents.

The dashboard immediately shows the health of our document ecosystem.

Twenty-five documents indexed. One hundred ninety-seven searchable chunks. Sixty-one open alerts. All critical severity.

Look at the impact comparison. Manual audit takes around fifty hours. DocOps Agent? Just one minute. That's ninety-nine point nine percent time saved. Sixty-one issues automatically detected.

The health score is zero out of one hundred — critical status. The pie chart shows seventy-seven percent are conflicts, twenty percent are coverage gaps, and three percent are staleness issues.

Now let's look at the Document Analytics — powered by Elasticsearch aggregations. Top document sections by chunk count. A treemap showing document sizes. The agent is running a staleness audit in the background.

Here are the detected alerts. Every single one flagged as critical. Numeric conflicts about password requirements. Duration mismatches. All discovered automatically by the agent.

This is the Agent Chat. Watch how it reasons through the conflicts.

It found a mismatch between Information Security Policy and Security Policy Update. One says sixty days, the other says forty-five days.

But here's what makes DocOps different — it doesn't just find problems. It suggests resolutions. "Both documents have similar authority. Schedule a review meeting to determine the correct value."

Now the Conflict Viewer. We can filter by topic and severity. Sixty-seven critical issues. Four high severity.

Let's examine a specific conflict. Security Policy Update says minimum sixteen characters for passwords. But the Employee Handbook says at least twelve characters.

Same topic. Different requirements. This is a critical compliance risk.

Click "Get Suggestion" and watch the AI analyze it.

Immediate priority. Estimated five minutes to fix. Eighty-five percent confidence.

The suggested change: Update "at least twelve character" to "minimum sixteen character."

The rationale: Security Policy Update twenty twenty-six is the authoritative source, so the stricter requirement should apply.

You can accept the suggestion or dismiss it. Full control with intelligent guidance.

Alert created! The issue is now tracked in our system.

The Alert Lifecycle page tracks resolution status. Open alerts, pending verification, resolved. Every action logged with timestamps for audit compliance.

Now the Search page. Hybrid search combining BM25 keyword matching with semantic vectors.

Search for "password" — ten results instantly. Relevance scores shown. Keywords highlighted. Export to Excel with one click.

This is Elasticsearch hybrid search in action.

Finally, Reports. Choose report type — Conflict Analysis or Staleness Report. Select export format. Include recommendations and executive summary. Filter by severity.

Generate the report. Found eight staleness issues.

The recommendations are actionable: Update or retire expired documents. Replace outdated year references. Establish a quarterly review cadence. Add metadata for tracking.

Download as Markdown, Excel, or PDF. Ready for stakeholders.

Twenty-five documents analyzed. Sixty-six issues detected. Conflicts found. Resolutions suggested. Reports generated.

All powered by Elasticsearch Agent Builder.

DocOps Agent. Find conflicts. Fix compliance. Sleep better.
"""


async def generate_voiceover():
    """Generate voiceover audio from script."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Generating voiceover with voice: {VOICE}")
    print(f"Script length: {len(SCRIPT.split())} words")

    communicate = edge_tts.Communicate(
        text=SCRIPT.strip(),
        voice=VOICE,
        rate=RATE,
    )

    print(f"Saving to: {AUDIO_PATH}")
    await communicate.save(str(AUDIO_PATH))

    word_count = len(SCRIPT.split())
    estimated_duration = word_count / 150 * 60

    with open(DURATION_PATH, "w") as f:
        f.write(f"word_count: {word_count}\n")
        f.write(f"estimated_duration_seconds: {estimated_duration:.1f}\n")
        f.write(f"estimated_duration_formatted: {int(estimated_duration // 60)}:{int(estimated_duration % 60):02d}\n")

    print(f"Voiceover generated successfully!")
    print(f"Estimated duration: {int(estimated_duration // 60)}:{int(estimated_duration % 60):02d}")

    return AUDIO_PATH


if __name__ == "__main__":
    asyncio.run(generate_voiceover())
