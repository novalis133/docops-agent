# DocOps Agent - Demo Video Pipeline

Automated video creation for the Elasticsearch Agent Builder Hackathon submission.

## Quick Start

### Prerequisites

1. **Python 3.11+** with pip
2. **FFmpeg** - Install with:
   - Windows: `winget install ffmpeg`
   - Mac: `brew install ffmpeg`
   - Linux: `apt install ffmpeg`

### Generate the Demo Video

**Windows:**
```cmd
cd video_pipeline
run_pipeline.bat
```

**Mac/Linux:**
```bash
cd video_pipeline
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### What It Does

1. **Generates voiceover** - Uses edge-tts (free, no API key) with professional voice
2. **Records screen** - Playwright automates browser to demo the app
3. **Combines everything** - FFmpeg merges video + audio + captions

## Output Files

```
output/
├── voiceover.mp3        # 3-minute voiceover
├── screen_recording.webm # Automated screen recording
└── final_demo.mp4       # Final video for DevPost
```

## Customization

### Change the Voiceover Script

Edit `generate_voiceover.py` and modify the `SCRIPT` variable:

```python
SCRIPT = """
Your new voiceover text here...
"""
```

Available voices:
- `en-US-GuyNeural` - Professional male (default)
- `en-US-JennyNeural` - Professional female
- `en-US-AriaNeural` - Conversational female

### Adjust Recording Timings

Edit `record_demo.py` and modify the `TIMINGS` dictionary:

```python
TIMINGS = {
    "initial_load": 3,      # Wait for page load
    "dashboard_view": 4,    # How long to show dashboard
    "conflict_scan_wait": 6, # Wait for scan to complete
    # ... etc
}
```

### Update Captions

Edit `generate_captions.srt` - standard SRT format:

```srt
1
00:00:00,000 --> 00:00:03,500
Your caption text here
```

## Troubleshooting

### Playwright can't find buttons

The Streamlit app generates dynamic IDs. Update selectors in `record_demo.py`:

```python
# Instead of:
page.get_by_role("button", name="Run Conflict Scan")

# Try:
page.get_by_text("Run Conflict Scan")
# or
page.locator("button:has-text('Run Conflict Scan')")
```

### Video and audio out of sync

1. Check durations match (video ~3 min, audio ~3 min)
2. Adjust `TIMINGS` in `record_demo.py`
3. Re-run the pipeline

### FFmpeg subtitle error

If captions fail, the script automatically retries without them. You can add captions later in video editing software.

### Browser doesn't open

Make sure DocOps is running first:
```bash
cd docops-agent
streamlit run frontend/app.py
```

Then run the video pipeline.

## Manual Alternative

If automation is too complex:

1. Run `python generate_voiceover.py` to get the audio
2. Play voiceover through headphones
3. Use OBS to screen record while manually clicking through the demo
4. Import both into DaVinci Resolve (free) and sync manually

## Demo Script Flow

The recording follows this sequence:

1. **Welcome Screen** (0:00 - 0:20)
   - Show DocOps Agent branding
   - Click "Get Started"

2. **Dashboard** (0:20 - 0:50)
   - Show health metrics
   - Click "Run Conflict Scan"
   - View results

3. **Conflict Viewer** (0:50 - 1:30)
   - Navigate to conflicts
   - Show side-by-side comparison
   - Expand conflict details

4. **Agent Chat** (1:30 - 2:10)
   - Type a query
   - Watch multi-step reasoning
   - View response

5. **Reports** (2:10 - 2:45)
   - Generate a compliance report
   - Show export options

6. **Final Dashboard** (2:45 - 3:00)
   - Return to dashboard
   - Hold on health score

## Tips for Best Results

1. **Clean data first**: Clear old alerts before recording
2. **Load demo documents**: Ensure sample data is indexed
3. **Close other apps**: Reduce CPU usage during recording
4. **Test run first**: Do a dry run without recording
5. **Review output**: Watch the video and adjust timings if needed
