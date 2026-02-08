#!/usr/bin/env python3
"""
Combine screen recording with voiceover and captions using FFmpeg.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
SCREEN_RECORDING = OUTPUT_DIR / "screen_recording.webm"
VOICEOVER = OUTPUT_DIR / "voiceover.mp3"
CAPTIONS = Path(__file__).parent / "generate_captions.srt"
FINAL_OUTPUT = OUTPUT_DIR / "final_demo.mp4"


def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_duration(file_path: Path) -> float:
    """Get duration of a media file in seconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def combine_video():
    """Combine video, audio, and captions."""
    print("Checking FFmpeg installation...")
    if not check_ffmpeg():
        print("ERROR: FFmpeg not found!")
        print("Install FFmpeg:")
        print("  - Windows: winget install ffmpeg")
        print("  - Mac: brew install ffmpeg")
        print("  - Linux: apt install ffmpeg")
        sys.exit(1)

    print("FFmpeg found!")

    # Check input files
    if not SCREEN_RECORDING.exists():
        print(f"ERROR: Screen recording not found: {SCREEN_RECORDING}")
        print("Run record_demo.py first!")
        sys.exit(1)

    if not VOICEOVER.exists():
        print(f"ERROR: Voiceover not found: {VOICEOVER}")
        print("Run generate_voiceover.py first!")
        sys.exit(1)

    if not CAPTIONS.exists():
        print(f"WARNING: Captions not found: {CAPTIONS}")
        print("Proceeding without captions...")
        has_captions = False
    else:
        has_captions = True

    # Get durations
    video_duration = get_duration(SCREEN_RECORDING)
    audio_duration = get_duration(VOICEOVER)

    print(f"Video duration: {video_duration:.1f}s")
    print(f"Audio duration: {audio_duration:.1f}s")

    # Determine output duration (use longer of the two)
    output_duration = max(video_duration, audio_duration)

    # Build FFmpeg command
    if has_captions:
        # With burned-in captions
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(SCREEN_RECORDING),
            "-i", str(VOICEOVER),
            "-vf", f"subtitles={CAPTIONS}:force_style='FontSize=24,FontName=Arial,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Shadow=1'",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",  # Stop when shortest stream ends
            "-movflags", "+faststart",  # Web optimization
            str(FINAL_OUTPUT)
        ]
    else:
        # Without captions
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(SCREEN_RECORDING),
            "-i", str(VOICEOVER),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(FINAL_OUTPUT)
        ]

    print("\nCombining video and audio...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")

            # Try without captions if that failed
            if has_captions:
                print("\nRetrying without captions...")
                cmd_simple = [
                    "ffmpeg",
                    "-y",
                    "-i", str(SCREEN_RECORDING),
                    "-i", str(VOICEOVER),
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    "-movflags", "+faststart",
                    str(FINAL_OUTPUT)
                ]
                result = subprocess.run(cmd_simple, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"FFmpeg error: {result.stderr}")
                    sys.exit(1)

        print(f"\nSuccess! Final video saved to: {FINAL_OUTPUT}")
        print(f"File size: {FINAL_OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        print(f"Error running FFmpeg: {e}")
        sys.exit(1)


def combine_video_simple():
    """Simple combination without captions (fallback)."""
    print("Using simple combination (no captions)...")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(SCREEN_RECORDING),
        "-i", str(VOICEOVER),
        "-map", "0:v",  # Video from first input
        "-map", "1:a",  # Audio from second input
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(FINAL_OUTPUT)
    ]

    subprocess.run(cmd, check=True)
    print(f"Video saved to: {FINAL_OUTPUT}")


if __name__ == "__main__":
    combine_video()
