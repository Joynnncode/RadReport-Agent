"""Smoke tests for the Streamlit UI, via AppTest.

These do not call an LLM. They assert the things that would embarrass the
project if they broke silently: the safety banner rendering, the app starting at
all, and the adversarial example buttons doing what they claim.

The banner test is the one that matters. Every other surface of this project
carries the not-a-medical-device framing, and the UI is the surface a stranger
actually sees. A refactor that quietly drops it should fail the build.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

# Absolute, resolved from the package rather than the CWD: pytest can be invoked
# from anywhere and a relative path breaks the moment it is.
from radreport.config import REPO_ROOT  # noqa: E402

APP = str(REPO_ROOT / "app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def test_app_starts_without_exception(app):
    assert not app.exception, app.exception


def test_safety_banner_is_present_and_unmissable(app):
    """Not a medical device, on the first screen, not behind an expander."""
    banners = " ".join(e.value for e in app.error)
    assert "Not a medical device" in banners
    assert "Not for clinical use" in banners
    assert "Research prototype" in banners


def test_disclaimer_also_closes_the_page(app):
    captions = " ".join(c.value for c in app.caption)
    assert "Not a medical device" in captions


def test_core_controls_exist(app):
    assert "RadReport Agent" in [t.value for t in app.title]
    labels = {s.label for s in app.selectbox}
    assert "Provider" in labels
    assert any(b.label == "Run agent" for b in app.button)


def test_adversarial_examples_populate_the_question_box(app):
    """These buttons are how a reviewer discovers the safety behaviour without
    having to think of an adversarial question themselves."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()

    at.button[0].click().run()
    assert "CXR9999999" in at.text_area[0].value

    at.button[1].click().run()
    assert "prescribe" in at.text_area[0].value.lower()

    at.button[2].click().run()
    assert "quote" in at.text_area[0].value.lower()


def test_no_deprecated_width_api():
    """use_container_width was removed after 2025-12-31; catch a reintroduction."""
    assert "use_container_width" not in open(APP, encoding="utf-8").read()


def test_no_unguarded_tool_calls_in_the_ui():
    """Any tool call outside a try/except can take the whole page down.

    The deployed app crashed with a full traceback because
    get_report_by_image was called bare and the corpus was missing. Streamlit
    renders an uncaught exception as a red wall in place of the app, so one
    missing file removes every other working feature from the page.
    """
    import ast

    TOOLS = {"get_report_by_image", "segment_lungs", "classify_xray", "compute_ctr",
             "search_reports", "search_literature", "image_bytes"}

    tree = ast.parse(open(APP, encoding="utf-8").read())
    guarded = {sub.lineno
               for node in ast.walk(tree) if isinstance(node, ast.Try)
               for sub in ast.walk(node) if isinstance(sub, ast.Call)}

    unguarded = [f"{n.func.id} (line {n.lineno})"
                 for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id in TOOLS and n.lineno not in guarded]

    assert not unguarded, f"unguarded tool calls: {unguarded}"
