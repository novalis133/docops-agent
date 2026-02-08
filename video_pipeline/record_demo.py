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
    "initial_load": 3,
    "welcome_pause": 2,
    "after_get_started": 2,
    "dashboard_view": 4,
    "before_conflict_scan": 1,
    "conflict_scan_wait": 6,
    "after_scan_results": 3,
    "navigate_conflict_viewer": 2,
    "conflict_viewer_pause": 4,
    "expand_conflict": 3,
    "conflict_detail_view": 5,
    "navigate_agent_chat": 2,
    "agent_chat_pause": 2,
    "type_query_delay": 0.05,  # Per character
    "agent_thinking": 8,
    "agent_response_view": 5,
    "navigate_reports": 2,
    "reports_pause": 2,
    "generate_report_wait": 5,
    "report_view": 4,
    "scroll_pause": 1,
    "final_hold": 5,
}


async def record_demo():
    """Main recording function."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    async with async_playwright() as p:
        # Launch browser with video recording
        browser = await p.chromium.launch(
            headless=False,  # Show browser for debugging
            slow_mo=300,  # Slow down actions for smooth recording
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
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(TIMINGS["initial_load"])

            # Pause on welcome screen to show the product
            await asyncio.sleep(TIMINGS["welcome_pause"])

            # Click "Get Started" button
            print("Clicking Get Started...")
            try:
                get_started = page.get_by_role("button", name="Get Started")
                await get_started.wait_for(state="visible", timeout=5000)
                await get_started.click()
                await asyncio.sleep(TIMINGS["after_get_started"])
            except Exception as e:
                print(f"Get Started button not found, may already be past welcome: {e}")

            # ===== SCENE 2: Dashboard =====
            print("Scene 2: Dashboard")
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(TIMINGS["dashboard_view"])

            # Scroll down to show metrics
            await page.mouse.wheel(0, 300)
            await asyncio.sleep(TIMINGS["scroll_pause"])

            # Click "Run Conflict Scan" button
            print("Running conflict scan...")
            await asyncio.sleep(TIMINGS["before_conflict_scan"])
            try:
                scan_button = page.get_by_role("button", name="Run Conflict Scan")
                await scan_button.wait_for(state="visible", timeout=5000)
                await scan_button.click()
                await asyncio.sleep(TIMINGS["conflict_scan_wait"])
            except Exception as e:
                print(f"Conflict scan button not found: {e}")

            # View scan results
            await asyncio.sleep(TIMINGS["after_scan_results"])

            # ===== SCENE 3: Conflict Viewer =====
            print("Scene 3: Conflict Viewer")
            # Navigate via sidebar
            try:
                conflict_nav = page.get_by_text("Conflict Viewer")
                await conflict_nav.click()
                await asyncio.sleep(TIMINGS["navigate_conflict_viewer"])
            except Exception as e:
                print(f"Could not navigate to Conflict Viewer: {e}")

            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(TIMINGS["conflict_viewer_pause"])

            # Scroll to show conflicts
            await page.mouse.wheel(0, 400)
            await asyncio.sleep(TIMINGS["scroll_pause"])

            # Try to expand a conflict detail
            print("Expanding conflict details...")
            try:
                # Look for expander elements
                expanders = page.locator("[data-testid='stExpander']")
                count = await expanders.count()
                if count > 0:
                    await expanders.first.click()
                    await asyncio.sleep(TIMINGS["expand_conflict"])
            except Exception as e:
                print(f"Could not expand conflict: {e}")

            await asyncio.sleep(TIMINGS["conflict_detail_view"])

            # ===== SCENE 4: Agent Chat =====
            print("Scene 4: Agent Chat")
            try:
                chat_nav = page.get_by_text("Agent Chat")
                await chat_nav.click()
                await asyncio.sleep(TIMINGS["navigate_agent_chat"])
            except Exception as e:
                print(f"Could not navigate to Agent Chat: {e}")

            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(TIMINGS["agent_chat_pause"])

            # Type a query in the chat
            print("Typing agent query...")
            try:
                chat_input = page.locator("textarea[placeholder*='Ask about']")
                await chat_input.wait_for(state="visible", timeout=5000)

                query = "Check for conflicts about password requirements"
                for char in query:
                    await chat_input.type(char, delay=TIMINGS["type_query_delay"] * 1000)

                await asyncio.sleep(0.5)

                # Press Enter to submit
                await chat_input.press("Enter")
                await asyncio.sleep(TIMINGS["agent_thinking"])

            except Exception as e:
                print(f"Could not interact with chat: {e}")
                # Try clicking a quick query button instead
                try:
                    quick_query = page.get_by_role("button", name="Check for conflicts about passwords")
                    await quick_query.click()
                    await asyncio.sleep(TIMINGS["agent_thinking"])
                except:
                    pass

            # View agent response
            await asyncio.sleep(TIMINGS["agent_response_view"])

            # Scroll to see full response
            await page.mouse.wheel(0, 300)
            await asyncio.sleep(TIMINGS["scroll_pause"])

            # ===== SCENE 5: Reports =====
            print("Scene 5: Reports")
            try:
                reports_nav = page.get_by_text("Reports")
                await reports_nav.click()
                await asyncio.sleep(TIMINGS["navigate_reports"])
            except Exception as e:
                print(f"Could not navigate to Reports: {e}")

            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(TIMINGS["reports_pause"])

            # Generate a report
            print("Generating report...")
            try:
                generate_btn = page.get_by_role("button", name="Generate Report")
                await generate_btn.wait_for(state="visible", timeout=5000)
                await generate_btn.click()
                await asyncio.sleep(TIMINGS["generate_report_wait"])
            except Exception as e:
                print(f"Could not generate report: {e}")

            # View generated report
            await asyncio.sleep(TIMINGS["report_view"])

            # Scroll through report
            await page.mouse.wheel(0, 500)
            await asyncio.sleep(TIMINGS["scroll_pause"])

            # ===== SCENE 6: Final Dashboard View =====
            print("Scene 6: Final view")
            try:
                dashboard_nav = page.get_by_text("Dashboard")
                await dashboard_nav.click()
                await asyncio.sleep(2)
            except:
                pass

            await page.wait_for_load_state("networkidle")

            # Final hold
            print("Holding final frame...")
            await asyncio.sleep(TIMINGS["final_hold"])

            print("Recording complete!")

        except Exception as e:
            print(f"Error during recording: {e}")
            raise

        finally:
            await context.close()
            await browser.close()

    # Find the recorded video and rename it
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
