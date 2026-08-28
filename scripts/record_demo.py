"""Record the demo GIF: one real question, end to end, in the real UI.

    RADREPORT_DEMO=1 streamlit run app.py --server.port 8511 &
    python scripts/record_demo.py

Why a GIF and not just the live link. The deployment runs on Streamlit
Community Cloud, which puts an app to sleep after 12 hours without traffic, so
a link in a post has a good chance of showing a reader a sleep screen instead of
the project. A GIF has no cold start, no quota, no API key and no daemon: it is
the one artefact that cannot be broken by someone else's infrastructure.

Why headless Chromium and not a screen recording. `screencapture -v` would film
the actual desktop -- whatever else is on it -- and depends on a permission
prompt and on windows being where they were last time. Driving a headless
browser records only the page, produces the same frames on any machine, and can
be re-run after a UI change without anyone having to sit and click.

The sequence is chosen to show the thing the README claims is the point: not
that a model answered, but that the answer is checkable. Question -> tools ->
cited answer -> the trace panel showing the exact tool output the model saw.

Dev-only. playwright is not in requirements.txt; install it with:

    pip install playwright && playwright install chromium
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Streamlit's own toolbar ("Deploy", the hamburger) is chrome, not the app.
# Hidden so the GIF shows the project rather than the hosting platform.
HIDE_CHROME = """
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], footer, #MainMenu { display: none !important; }
"""


def record(url: str, out_dir: Path, width: int, height: int) -> Path:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(out_dir),
            record_video_size={"width": width, "height": height},
            device_scale_factor=1,
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=60_000)
        page.add_style_tag(content=HIDE_CHROME)
        page.wait_for_timeout(2500)          # let a reader take in the setup

        run = page.get_by_role("button", name="Run agent")
        run.scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        run.click()

        # The agent really runs: tools, a live LLM, however long that takes.
        # Waiting on the Answer heading rather than a fixed sleep means the GIF
        # never captures a half-rendered page, and never pads a fast run.
        answer = page.get_by_text("Answer", exact=True).first
        answer.wait_for(timeout=180_000)

        # Park ON the answer and hold. The answer renders below the fold, so
        # without this the viewport is still showing the input form when the
        # result arrives, and the first recording scrolled past the prose fast
        # enough that only its last bullet was ever readable. The cited quote is
        # the thing the whole project is about; it gets time on screen.
        answer.scroll_into_view_if_needed()
        page.wait_for_timeout(3500)

        for _ in range(5):
            page.mouse.wheel(0, 200)
            page.wait_for_timeout(900)
        page.wait_for_timeout(1200)

        # Open compute_ctr to show the exact numbers the model received. This
        # is the whole argument of the project in one frame: not "a model said
        # cardiomegaly" but "here is the ratio, computed by geometry, that the
        # sentence rests on".
        #
        # compute_ctr specifically, not "the first tool expander". Matching
        # `compute_ctr or get_report_by_image` picked get_report_by_image simply
        # because it is rendered first, and the run then scrolled past it -- the
        # recording ended on a page of collapsed rows and the payoff was missing.
        summaries = page.locator('[data-testid="stExpander"] summary')
        target = None
        for i in range(summaries.count()):
            if "compute_ctr" in (summaries.nth(i).inner_text() or ""):
                target = i
                break
        if target is not None:
            summaries.nth(target).scroll_into_view_if_needed()
            page.wait_for_timeout(600)
            summaries.nth(target).click()
            page.wait_for_timeout(1200)
            details = page.locator('[data-testid="stExpander"] details').nth(target)
            if details.get_attribute("open") is None:      # click landed nowhere
                summaries.nth(target).click()
                page.wait_for_timeout(1200)
            # Keep it on screen. Scrolling on past the thing just opened is how
            # the first recording ended up showing nothing.
            summaries.nth(target).scroll_into_view_if_needed()
            page.wait_for_timeout(400)
            page.mouse.wheel(0, 220)
            page.wait_for_timeout(4500)
        else:
            page.wait_for_timeout(2000)

        video = page.video
        context.close()                      # finalises the file
        browser.close()
        return Path(video.path())


def to_gif(webm: Path, gif: Path, fps: int, scale: int, speed: float) -> None:
    """Two-pass palette conversion: one shared palette, then apply it.

    ffmpeg's default is a 256-colour palette guessed per frame, which makes
    flat UI backgrounds shimmer. palettegen/paletteuse costs one extra pass and
    removes it.
    """
    palette = gif.with_suffix(".palette.png")
    chain = f"setpts={1 / speed:.4f}*PTS,fps={fps},scale={scale}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-i", str(webm), "-vf", f"{chain},palettegen=stats_mode=diff",
                    str(palette)], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(webm), "-i", str(palette),
                    "-lavfi", f"{chain}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    str(gif)], check=True, capture_output=True)
    palette.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8511")
    ap.add_argument("--out", default=str(REPO / "docs" / "demo.gif"))
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=860)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--scale", type=int, default=900, help="GIF width in px")
    ap.add_argument("--speed", type=float, default=1.6)
    ap.add_argument("--keep-video", action="store_true")
    ns = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found; brew install ffmpeg")

    raw_dir = REPO / ".cache" / "demo-recording"
    print(f"recording {ns.url} ...")
    webm = record(ns.url, raw_dir, ns.width, ns.height)
    print(f"  raw video: {webm} ({webm.stat().st_size / 1e6:.1f} MB)")

    gif = Path(ns.out)
    gif.parent.mkdir(parents=True, exist_ok=True)
    to_gif(webm, gif, ns.fps, ns.scale, ns.speed)
    size_mb = gif.stat().st_size / 1e6
    print(f"  gif: {gif} ({size_mb:.1f} MB)")
    # LinkedIn's limit for an animated GIF is 8 MB; the project's own shipping
    # checklist says under 10. Warn rather than silently hand over something
    # that will be rejected at upload.
    if size_mb > 8:
        print("  ! over 8 MB: lower --fps, --scale, or raise --speed")

    if not ns.keep_video:
        shutil.rmtree(raw_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
