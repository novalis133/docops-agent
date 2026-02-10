#!/usr/bin/env python3
"""
Automated screen recording for DocOps Agent demo.
Uses Playwright to navigate the Streamlit app and record the session.
"""

import time
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

# Configuration
BASE_URL = "http://localhost:8501"
OUTPUT_DIR = Path(__file__).parent / "output"
VIDEO_PATH = OUTPUT_DIR / "screen_recording.webm"

# Timing configuration (in seconds) - adjust to sync with voiceover
TIMINGS = {
    "initial_load": 4,
    "welcome_pause": 3,
    "after_get_started": 3,
    "dashboard_view": 5,
    "before_conflict_scan": 2,
    "conflict_scan_wait": 8,
    "after_scan_results": 4,
    "navigate_conflict_viewer": 3,
    "conflict_viewer_pause": 5,
    "expand_conflict": 4,
    "conflict_detail_view": 6,
    "navigate_agent_chat": 3,
    "agent_chat_pause": 3,
    "type_query_delay": 0.08,
    "agent_thinking": 10,
    "agent_response_view": 6,
    "navigate_reports": 3,
    "reports_pause": 3,
    "generate_report_wait": 6,
    "report_view": 5,
    "scroll_pause": 2,
    "final_hold": 5,
}


async def record_demo():
    """Main recording function."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    async with async_playwright() as p:
        # Launch browser with video recording
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=400,
        )

        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(OUTPUT_DIR),
            record_video_size={"width": 1920, "height": 1080},
        )

        page = await context.new_page()

        try:
            print("Starting demo recording...")

            # ===== SCENE 1: Welcome Screen =====
            print("Scene 1: Welcome screen")
            await page.goto(BASE_URL)
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(TIMINGS["initial_load"])
            await asyncio.sleep(TIMINGS["welcome_pause"])

            # Click "Get Started" button
            print("Clicking Get Started...")
            try:
                get_started = page.locator("button:has-text('Get Started')")
                await get_started.wait_for(state="visible", timeout=5000)
                await get_started.click()
                await asyncio.sleep(TIMINGS["after_get_started"])
            except Exception as e:
                print(f"Get Started not found: {e}")

            # ===== SCENE 2: Dashboard =====
            print("Scene 2: Dashboard")
            await asyncio.sleep(3)

            # Click Dashboard in sidebar
            try:
                dashboard = page.locator("label:has-text('Dashboard')")
                await dashboard.click()
                await asyncio.sleep(TIMINGS["dashboard_view"])
            except Exception as e:
                print(f"Dashboard nav failed: {e}")

            # Scroll to show charts
            await page.mouse.wheel(0, 400)
            await asyncio.sleep(TIMINGS["scroll_pause"])

            # Click "Run Conflict Scan"
            print("Running conflict scan...")
            try:
                scan_btn = page.locator("button:has-text('Run Conflict Scan')")
                await scan_btn.wait_for(state="visible", timeout=5000)
                await scan_btn.click()
                await asyncio.sleep(TIMINGS["conflict_scan_wait"])
            except Exception as e:
                print(f"Conflict scan failed: {e}")

            await asyncio.sleep(TIMINGS["after_scan_results"])

            # ===== SCENE 3: Conflict Viewer =====
            print("Scene 3: Conflict Viewer")
            try:
                conflict_nav = page.locator("label:has-text('Conflict Viewer')")
                await conflict_nav.click()
                await asyncio.sleep(TIMINGS["navigate_conflict_viewer"])
            except Exception as e:
                print(f"Conflict Viewer nav failed: {e}")

            await asyncio.sleep(TIMINGS["conflict_viewer_pause"])

            # Scroll to show conflicts
            await page.mouse.wheel(0, 500)
            await asyncio.sleep(TIMINGS["scroll_pause"])

            # Expand a conflict
            try:
                expander = page.locator("[data-testid='stExpander']").first
                await expander.click()
                await asyncio.sleep(TIMINGS["expand_conflict"])
            except Exception as e:
                print(f"Could not expand conflict: {e}")

            await asyncio.sleep(TIMINGS["conflict_detail_view"])

            # ===== SCENE 4: Agent Chat =====
            print("Scene 4: Agent Chat")
            try:
                chat_nav = page.locator("label:has-text('Agent Chat')")
                await chat_nav.click()
                await asyncio.sleep(TIMINGS["navigate_agent_chat"])
            except Exception as e:
                print(f"Agent Chat nav failed: {e}")

            await asyncio.sleep(TIMINGS["agent_chat_pause"])

            # Type a query
            print("Typing agent query...")
            try:
                chat_input = page.locator("textarea").first
                await chat_input.wait_for(state="visible", timeout=5000)

                query = "Check for conflicts about password requirements"
                await chat_input.fill("")
                for char in query:
                    await chat_input.type(char, delay=TIMINGS["type_query_delay"] * 1000)

                await asyncio.sleep(0.5)
                await chat_input.press("Enter")
                await asyncio.sleep(TIMINGS["agent_thinking"])
            except Exception as e:
                print(f"Chat interaction failed: {e}")

            await asyncio.sleep(TIMINGS["agent_response_view"])
            await page.mouse.wheel(0, 400)
            await asyncio.sleep(TIMINGS["scroll_pause"])

            # ===== SCENE 5: Reports =====
            print("Scene 5: Reports")
            try:
                reports_nav = page.locator("label:has-text('Reports')")
                await reports_nav.click()
                await asyncio.sleep(TIMINGS["navigate_reports"])
            except Exception as e:
                print(f"Reports nav failed: {e}")

            await asyncio.sleep(TIMINGS["reports_pause"])

            # Generate report
            print("Generating report...")
            try:
                gen_btn = page.locator("button:has-text('Generate Report')")
                await gen_btn.wait_for(state="visible", timeout=5000)
                await gen_btn.click()
                await asyncio.sleep(TIMINGS["generate_report_wait"])
            except Exception as e:
                print(f"Generate report failed: {e}")

            await asyncio.sleep(TIMINGS["report_view"])
            await page.mouse.wheel(0, 600)
            await asyncio.sleep(TIMINGS["scroll_pause"])

            # ===== SCENE 6: Final =====
            print("Scene 6: Final view")
            try:
                dashboard_nav = page.locator("label:has-text('Dashboard')")
                await dashboard_nav.click()
                await asyncio.sleep(3)
            except:
                pass

            print("Holding final frame...")
            await asyncio.sleep(TIMINGS["final_hold"])
            print("Recording complete!")

        except Exception as e:
            print(f"Error during recording: {e}")
            raise
        finally:
            await context.close()
            await browser.close()

    # Rename video file
    print("Finalizing video file...")
    video_files = list(OUTPUT_DIR.glob("*.webm"))
    if video_files:
        latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
        if latest_video != VIDEO_PATH:
            latest_video.rename(VIDEO_PATH)
        print(f"Video saved to: {VIDEO_PATH}")
    else:
        print("Warning: No video file found!")


if __name__ == "__main__":
    asyncio.run(record_demo())
