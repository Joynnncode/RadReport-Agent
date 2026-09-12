"""Check a deployed instance from the outside, the way a stranger would.

WHY THIS EXISTS

The deployment has two modes that look nearly identical in a screenshot and
mean different things: live inference, and precomputed results served from
data/demo_cache.json. The difference is the whole point of the demo-mode design,
so "the app loads" is not a useful check. What matters is which mode is actually
running, and whether the pieces that only exist in the cloud (the mounted image
share, the API key in a Container App secret) are really wired up.

    python scripts/verify_deployment.py https://<azure-host>            # expects live
    python scripts/verify_deployment.py https://<app>.streamlit.app --expect demo
    python scripts/verify_deployment.py https://<host> --no-agent       # skip the LLM call

The agent check spends real tokens, which is why it can be turned off, and why
it is last: everything cheaper runs first and reports before it starts.

Exit status is 0 only if every check passed, so this can gate a deploy.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

CASE_ID = re.compile(r"\d+_IM-[\d-]+")
TITLE = "text=RadReport Agent"


class Checks:
    """Collects results so one failure does not hide the others."""

    def __init__(self) -> None:
        self.failed = 0

    def __call__(self, name: str, passed: bool, detail: str = "") -> bool:
        self.failed += not passed
        print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""),
              flush=True)
        return passed


def app_root(page, timeout_s: int):
    """Return whatever actually holds the app: the page, or a child frame.

    Streamlit Community Cloud wraps a deployed app in an iframe served from
    `<host>/~/+/`, so every selector against the main frame misses and the app
    looks unreachable while a screenshot of the same moment shows it rendered
    perfectly. A direct deployment, like Container Apps, has no wrapper. Rather
    than special-casing hosts, find the frame with the title in it.

    Both free tiers also start cold and neither delay is the app's fault:
    Container Apps scales to zero and takes ~34s to pull the image, and
    Streamlit Community Cloud sleeps after 12 hours and can take several
    minutes to come back. It wakes on its own; there is no button to press.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for frame in page.frames:
            try:
                if frame.locator(TITLE).count():
                    return frame
            except Exception:
                continue        # a frame can be navigating; try it again next pass
        page.wait_for_timeout(2_000)
    return None


def selected_case(app) -> str:
    """The chosen case id.

    Streamlit keeps a selectbox's value in an input element, not in the widget's
    text, so inner_text() on the widget returns only its label. Read the input,
    and fall back to the question box, which is seeded with the same id.
    """
    for locator in ('div[data-testid="stSelectbox"] input', "textarea"):
        target = app.locator(locator).first
        if target.count():
            found = CASE_ID.search(target.input_value() or "")
            if found:
                return found.group(0)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--expect", choices=("live", "demo"), default="live",
                    help="which mode the deployment should be serving")
    ap.add_argument("--no-agent", action="store_true",
                    help="skip the end-to-end agent run, which costs tokens")
    ap.add_argument("--wait", type=int, default=600,
                    help="seconds to allow for a cold start or a sleeping app")
    ap.add_argument("--screenshot", default="artifacts/deployment.png")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright      # dev-only dependency

    live = args.expect == "live"
    check = Checks()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1600})
        page.set_default_timeout(120_000)
        page.goto(args.url, wait_until="domcontentloaded")

        app = app_root(page, args.wait)
        if app is None:
            check("app reachable", False, f"no frame showed the title within {args.wait}s")
            page.screenshot(path=args.screenshot, full_page=True)
            browser.close()
            print(f"\nscreenshot: {args.screenshot}")
            print(f"{check.failed} CHECK(S) FAILED")
            return 1

        body = app.inner_text("body")
        check("safety banner present", "Not a medical device" in body)

        # The banner is the app's own statement about which mode it is in, and
        # it is generated from DEMO_MODE, not from configuration we can see here.
        demo_banner = "Precomputed demo" in body
        check(f"serving {args.expect}", demo_banner is not live,
              "precomputed banner " + ("shown" if demo_banner else "absent"))

        case = selected_case(app)
        check("a case is loaded", bool(case), case or "no case id found")

        # Segmentation is the step that needs the real image and ~1.8 GB of RAM.
        # Waiting on the caption text rather than the <img>: Streamlit renders
        # captions as divs and leaves the alt attribute empty.
        app.get_by_text("Show segmentation overlay", exact=False).first.click()
        try:
            app.wait_for_selector("text=Lungs (blue), heart (red)", timeout=300_000)
            segmented = check("segmentation overlay rendered", True)
        except Exception as exc:
            segmented = check("segmentation overlay rendered", False, str(exc)[:80])

        if segmented and live:
            # In demo mode every tool result carries this note. Live output must
            # not, or the deployment is quietly serving the cache.
            check("no precomputed note in output", "Precomputed result" not in app.inner_text("body"))

        if not args.no_agent:
            # Only a real run proves the provider key reached the container and
            # works from there. A wrong key fails here and nowhere earlier.
            app.wait_for_selector("text=API_KEY found", timeout=60_000)
            app.get_by_role("button", name="Run agent").click()
            try:
                app.wait_for_selector("text=Tools called", timeout=300_000)
                answered = check("agent run completed", True)
            except Exception as exc:
                answered = check("agent run completed", False, str(exc)[:80])

            if answered:
                body = app.inner_text("body")
                calls = re.search(r"Tools called\s*\n?(\d+)", body)
                check("the agent called tools", bool(calls) and calls.group(1) != "0",
                      f"{calls.group(1)} call(s)" if calls else "no count rendered")
                check("the loop converged", "did not converge" not in body)

        page.screenshot(path=args.screenshot, full_page=True)
        print(f"\nscreenshot: {args.screenshot}")
        browser.close()

    print("VERIFIED" if not check.failed else f"{check.failed} CHECK(S) FAILED")
    return 0 if not check.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
