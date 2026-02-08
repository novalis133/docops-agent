@echo off
REM DocOps Agent Demo Video Pipeline (Windows)
REM Creates a complete demo video with voiceover and captions

echo ========================================
echo   DocOps Agent - Video Pipeline
echo ========================================
echo.

cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    exit /b 1
)
echo   Python: OK

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ERROR: FFmpeg not found!
    echo Install with: winget install ffmpeg
    exit /b 1
)
echo   FFmpeg: OK

REM Install Python dependencies
echo   Installing Python packages...
pip install -q playwright edge-tts 2>nul
echo   Python packages: OK

REM Install Playwright browsers
python -m playwright install chromium 2>nul
echo   Playwright browsers: OK

echo.
echo All dependencies satisfied!
echo.

REM Step 1: Generate voiceover
echo ========================================
echo   Step 1: Generating Voiceover
echo ========================================
python generate_voiceover.py
if errorlevel 1 (
    echo ERROR: Voiceover generation failed!
    exit /b 1
)

echo.

REM Step 2: Record screen demo
echo ========================================
echo   Step 2: Recording Screen Demo
echo ========================================
echo IMPORTANT: Make sure DocOps is running at http://localhost:8501
echo.
pause
echo.
python record_demo.py
if errorlevel 1 (
    echo ERROR: Screen recording failed!
    exit /b 1
)

echo.

REM Step 3: Combine video and audio
echo ========================================
echo   Step 3: Combining Video and Audio
echo ========================================
python combine_video.py
if errorlevel 1 (
    echo ERROR: Video combination failed!
    exit /b 1
)

echo.
echo ========================================
echo   Pipeline Complete!
echo ========================================
echo.
echo Output files:
echo   - Voiceover: output\voiceover.mp3
echo   - Recording: output\screen_recording.webm
echo   - Final:     output\final_demo.mp4
echo.
echo Upload output\final_demo.mp4 to DevPost!
echo.
pause
