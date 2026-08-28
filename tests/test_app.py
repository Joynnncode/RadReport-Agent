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


def _case_selectbox(at):
    """The case picker, or a skip.

    Without a corpus the app renders an explanatory st.info and no selectbox, so
    a bare next() raises StopIteration and the test reports a confusing crash
    instead of "you have no data". Found by running the suite inside the Docker
    image, where data/ is a mount rather than a checkout.
    """
    box = next((s for s in at.selectbox if s.label == "Pick a case"), None)
    if box is None or len(box.options) < 2:
        pytest.skip("needs a corpus with at least two cases; run scripts/fetch_data.py")
    return box


def test_switching_case_reseeds_the_untouched_question():
    """A stale case id in the box is worse than an empty box: the agent would
    go and analyse the case the user just navigated away from."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()

    box = _case_selectbox(at)
    first = box.value
    assert first in at.text_area[0].value

    other = next(o for o in box.options if o != first)
    box.select(other).run()

    assert other in at.text_area[0].value
    assert first not in at.text_area[0].value


def test_switching_case_never_clobbers_a_typed_question():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()

    at.text_area[0].set_value("What does the corpus say about pleural effusion?").run()

    box = _case_selectbox(at)
    box.select(next(o for o in box.options if o != box.value)).run()

    assert at.text_area[0].value == "What does the corpus say about pleural effusion?"


def test_selected_case_is_passed_to_the_agent():
    """The sidebar selection is context the model cannot otherwise see. Without
    it, 'does this X-ray show anything?' reaches the agent with no image and it
    correctly refuses instead of calling a tool."""
    source = open(APP, encoding="utf-8").read()
    assert "UI context: the user is viewing case" in source
    assert "entry(message," in source
