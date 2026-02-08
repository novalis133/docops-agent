#!/bin/bash
#
# DocOps Agent Demo Video Pipeline
# Creates a complete demo video with voiceover and captions
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  DocOps Agent - Video Pipeline"
echo "========================================"
echo ""

# Check dependencies
echo "Checking dependencies..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found!"
    exit 1
fi
echo "  Python: OK"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: FFmpeg not found!"
    echo "Install with:"
    echo "  - Windows: winget install ffmpeg"
    echo "  - Mac: brew install ffmpeg"
    echo "  - Linux: apt install ffmpeg"
    exit 1
fi
echo "  FFmpeg: OK"

# Check/install Python dependencies
echo "  Checking Python packages..."
pip install -q playwright edge-tts 2>/dev/null || {
    echo "Installing Python dependencies..."
    pip install playwright edge-tts
}
echo "  Python packages: OK"

# Install Playwright browsers if needed
python -m playwright install chromium 2>/dev/null || true
echo "  Playwright browsers: OK"

echo ""
echo "All dependencies satisfied!"
echo ""

# Step 1: Generate voiceover
echo "========================================"
echo "  Step 1: Generating Voiceover"
echo "========================================"
python generate_voiceover.py

echo ""

# Step 2: Record screen demo
echo "========================================"
echo "  Step 2: Recording Screen Demo"
echo "========================================"
echo "IMPORTANT: Make sure DocOps is running at http://localhost:8501"
echo ""
read -p "Press Enter when DocOps is ready, or Ctrl+C to cancel..."
echo ""
python record_demo.py

echo ""

# Step 3: Combine video and audio
echo "========================================"
echo "  Step 3: Combining Video & Audio"
echo "========================================"
python combine_video.py

echo ""
echo "========================================"
echo "  Pipeline Complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - Voiceover: output/voiceover.mp3"
echo "  - Recording: output/screen_recording.webm"
echo "  - Final:     output/final_demo.mp4"
echo ""
echo "Upload output/final_demo.mp4 to DevPost!"
echo ""
