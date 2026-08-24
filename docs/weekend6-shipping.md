# Weekend 6: ship it and narrate it

Five tasks. Roughly 6 hours. Do them in this order — the deploy has to exist
before you can film it, and the film is what you link from the write-up.

---

## Before you start: the deploy blocker, and what was done about it

Profiling the app before deploying:

| | Measured | Streamlit Community Cloud |
|---|---|---|
| Peak RAM, PSPNet segmentation | **1,816 MB** | ~1 GB |
| torch + cached weights on disk | **820 MB** | tight |
| Dataset | gitignored, so not in the repo | — |

The app would install, start, and be killed by the OOM reaper the first time
anyone toggled the overlay.

**The fix already in the repo:** `scripts/precompute_demo.py` runs the real models
once over 40 cases and writes `data/demo_cache.json` (10 MB, committed).
`RADREPORT_DEMO=1` makes `radreport/tools/__init__.py` import
`radreport/tools/demo.py` instead of the torch-backed tools, so **torch is never
imported at all**. Peak RSS drops from 1,816 MB to **71 MB**, and the served
values are byte-identical to the real model outputs, because they are the real
model outputs.

Retrieval, PubMed and the agent loop stay fully live — none of them need torch.

> **Say this in interviews.** "I profiled it before deploying, found segmentation
> peaked at 1.8 GB against a 1 GB limit, and shipped a precomputed-inference
> variant so the public demo keeps the full user experience within the free
> tier." That is an engineering answer. "I couldn't deploy it" is not.

The honesty requirement is enforced in code: every demo result carries
`precomputed: True`, and the app shows a banner saying so. *"This model runs on
your image in 40 ms"* and *"I ran this model last Tuesday on 40 images"* are
different claims, and a demo that blurs them misrepresents the system.

---

## Task 1 — Deploy (about 45 minutes)

### 1.1 Push to GitHub

```bash
./scripts/check_fresh_clone.sh   # RUN THIS FIRST. See below.
git add -A
git status                       # READ THIS. Confirm no .env, no data/images/
git commit -m "..."              # your words, see the one-paragraph rule
gh repo create radreport-agent --public --source=. --push
```

**`check_fresh_clone.sh` is not optional.** A green local suite does not mean a
green CI: your working tree contains gitignored files that tests can silently
depend on. The first push of this repo failed with six errors that all passed
locally, for exactly that reason. The script copies only the files git would ship
into a temp directory and runs the suite there, in about ten seconds.

Before you push, verify nothing secret is going:

```bash
git check-ignore -q .env && echo "safe" || echo "STOP"
git ls-files | grep -E "\.env$|data/images" && echo "STOP" || echo "safe"
```

### 1.2 Point Streamlit Cloud at it

1. Go to **share.streamlit.io**, sign in with GitHub.
2. **New app** → your repo → branch `main` → main file `app.py`.
3. **Advanced settings** → set **Python 3.11**. Not 3.12+; some wheels lag.

### 1.3 Requirements

Streamlit Cloud installs `requirements.txt` by default, which includes torch and
will fail or OOM. Two options:

- **Simplest:** rename for the deploy — commit `requirements-deploy.txt` as
  `requirements.txt` on a `deploy` branch, and point the app at that branch.
- **Cleaner:** keep `main` as is and create a `deploy` branch whose
  `requirements.txt` is the deploy one. Rebase it forward when you change deps.

Use the first. You can explain the trade-off if asked; do not spend the weekend
on branch hygiene.

### 1.4 Secrets

App settings → **Secrets**, paste TOML (not `KEY=value`):

```toml
GROQ_API_KEY = "gsk_..."
RADREPORT_DEMO = "1"
NCBI_EMAIL = "you@example.com"
```

`RADREPORT_DEMO = "1"` is not optional. Without it the app tries to import torch,
which is not installed, and you get an ImportError on the dashboard. The repo
raises a message telling you exactly this, but set it and skip the detour.

Use **Groq, not Gemini**, as the deployed provider. Measured this weekend: Groq
answers in ~1.9 s and Gemini in ~38 s on the same question, and Gemini's free
tier exhausted after roughly 20 agent runs. A demo that times out is worse than
no demo.

### 1.5 Verify like a stranger

Open the URL in a private window and check, in order:

- [ ] Safety banner is the first thing visible
- [ ] Precomputed-demo banner is visible
- [ ] A case loads and shows an image
- [ ] The overlay toggle works
- [ ] "Missing case" button → agent says the case is not in the corpus
- [ ] "Out of scope" button → agent refuses
- [ ] The trace panel expands and shows real tool JSON
- [ ] No key appears anywhere in the page source

If it sleeps after inactivity, that is normal on the free tier. Note it in the
README so a reviewer knows to wait 30 seconds rather than assume it is broken.

---

## Task 2 — The 60-second video (about 1 hour)

No narration, same format as your cooking app video. Screen recording only.

**Why no narration:** a recruiter watches on mute. Text overlays survive that;
your voice does not. It is also far faster to reshoot.

### Shot list, 60 seconds

| Time | Shot | Overlay text |
|---|---|---|
| 0:00–0:05 | App loads, safety banner visible | "Clinical imaging agent — research prototype" |
| 0:05–0:15 | Pick a case, toggle segmentation overlay | "6 tools: classification, segmentation, CTR, retrieval, PubMed" |
| 0:15–0:30 | Ask the cardiomegaly question, answer appears | "Agent chooses which tools to call" |
| 0:30–0:42 | **Expand the trace panel, scroll it** | "Every claim traceable to a tool call" |
| 0:42–0:52 | Click "Missing case", show the refusal | "Refuses to substitute another patient's report" |
| 0:52–0:60 | Terminal: `pytest` → 133 passed | "133 tests · 43-case eval harness" |

The trace panel is the shot that matters. Give it the most time. It is the thing
no bootcamp project has.

**Recording:** macOS `Cmd+Shift+5`. Record a small region, not the full screen —
text stays legible on a phone. Trim in QuickTime. Export at 1080p. Keep it under
10 MB so it uploads to LinkedIn without re-encoding to mush.

**Do a dry run first.** Warm the app up so nothing loads mid-take, and use a case
you know produces a clean answer.

---

## Task 3 — The 900-word write-up (about 2 hours)

Structure, with your actual material. Do not write this from memory — open
`DECISIONS.md` and mine it.

**1. The problem (120 words).** Not "I wanted to learn agents". Something like:
clinical imaging AI has to combine model outputs with the written record, and the
dangerous failures are not crashes, they are confident wrong answers. State what
you built in one sentence.

**2. The architecture (200 words).** The loop, the six tools, why no LangChain.
Use the Mermaid diagram from the README. The one detail worth spelling out: masks
go to disk, handles go in the conversation, and why that keeps the context window
from exploding.

**3. One thing that surprised you (250 words).** *This is the section people
remember.* Use the **fabricated citation**: the agent invented direct quotations
from PubMed papers whose abstracts the tool never fetched, with named authors and
citation markers. Nothing crashed, tool selection was correct, and it would have
passed a demo. Only the groundedness metric caught it. Say what you changed.

Alternative if you would rather lead with a technical bug: the **double sigmoid**,
where torchxrayvision's model already applies sigmoid internally and my extra one
squashed every pathology into [0.5, 0.73] — a silent miscalibration that looked
like a working model.

Pick one. Do not cram both.

**4. The evaluation (200 words).** 43 cases, four metrics, the numbers:
89.5% tool selection, 97.2% deterministic checks, $0.00083/query, 17s median.
Then the finding: the model answers cardiomegaly questions from the report text
without looking at the image, 50% on that category. **Include a limitation** —
43 cases is small, the labels are derived from the reports themselves, one
reviewer wrote the gold answers. Stating limits reads as competence, not weakness.

**5. What I would do next (130 words).** Radiologist-measured CTR ground truth
(the current cases verify the agent reports what the tool computed, not that the
tool is right). Multi-sample runs to separate regression from reroll. A confidence
gate that refuses to answer when the report and the measurement disagree.

**Rules.** No em dashes. First person. One concrete number per section minimum.
Link the repo, the live app, and the video.

---

## Task 4 — Three STAR stories (about 1 hour)

`DECISIONS.md` has 35 entries; you need three you can tell cold. Here is one
worked through, so you can see the shape. **Do the other two yourself** — the
value is in the retelling, not the transcript.

### Worked example: the fabricated citation

> **Situation.** I had built a clinical imaging agent with six tools and a
> 43-case evaluation harness. It was passing tool-selection checks at 89.5% and
> every adversarial safety case.
>
> **Task.** I was validating the groundedness metric, which checks that every
> quoted span in an answer appears verbatim in what a tool actually returned. It
> came back at 52.9%, which looked like the metric was broken.
>
> **Action.** I read the failures one by one instead of loosening the check. Most
> were cosmetic: the model rewrote punctuation as it quoted, non-breaking hyphens
> and curly apostrophes, and inserted markdown emphasis inside the quotation
> marks. I normalised that typographic layer away. But two failures were
> different. Asked why the cardiothoracic ratio is unreliable on AP films, the
> agent had produced a sentence in quotation marks attributed to named authors,
> with a citation marker. My PubMed tool returns titles and PMIDs only — I had
> deliberately not fetched abstracts, to save context. The agent had never read
> those papers. It fabricated the quotation.
>
> **Result.** I added an explicit system-prompt rule about what that tool returns
> and what may therefore be quoted, and a gold-set case that asks directly for
> quotations and flags "et al. reported" plus citation markers, verified against
> the real fabricated output. The wider lesson was about the metric: my instinct
> was that 52.9% meant the check was broken, and if I had loosened it without
> reading the failures I would have deleted the only thing that caught this.
> Nothing crashed, tool selection was correct, and it would have passed a demo.

### Your other two — pick from these

- **The double sigmoid** (2026-08-18). Answers "tell me about a bug that was hard
  to find". Strong because the failure was silent and plausible.
- **Four failures in a row were my checks, not the agent** (2026-08-21). Answers
  "tell me about a time you were wrong". Strong because you corrected your own
  process, not just a line of code.
- **A quota wall is not a failing test case** (2026-08-21). Answers "tell me about
  measurement going wrong". Strong because you nearly published a false claim
  about a model and caught it.
- **Exact lookup is a separate tool from search** (2026-08-18). Answers "what is
  the most dangerous failure mode?" — the single best question for a clinical AI
  employer.

Write each as four labelled paragraphs. Say them out loud once. If a paragraph is
hard to say, it is too long.

---

## Task 5 — Rehearse the walkthrough (about 1 hour)

Twice, recorded, out loud. This is the task most likely to be skipped and the one
that fixes the actual problem in your feedback.

**Round 1 — the loop.** Screen-share `radreport/agent.py` and talk through it for
five minutes without notes. `docs/agent-walkthrough.md` has the eight questions;
answer them aloud. The three you must not fumble:

1. Why append the model's tool-call turn *as well as* the result?
2. What stops it looping forever, and why is hitting the cap a failure rather
   than an answer?
3. What is the most dangerous failure mode, and what did you do about it?

**Round 2 — the evaluation.** Five minutes on how you know it works. The gold
set, the four metrics, why the judge is a different provider, and — crucially —
what your evaluation does *not* cover. That last part is where you sound like an
engineer rather than a candidate.

**Watch the recordings.** You are listening for two things: places you recite
rather than explain, and places you say "basically" or "sort of" because you are
unsure. Both mark a section to re-read.

---

## Definition of done

- [ ] Public repo, no secrets, no data committed
- [ ] Live Streamlit app, verified in a private window
- [ ] 60-second video, under 10 MB, trace panel featured
- [ ] 900-word write-up posted, linking repo + app + video
- [ ] Three STAR stories written out
- [ ] Two recorded walkthroughs watched back
- [ ] CV lines updated with the real numbers

## The CV lines, with this weekend's actual figures

> **RadReport Agent** | Python, Gemini and Groq APIs, PyTorch, TorchXRayVision, Streamlit, Docker, GitHub Actions
> Built a tool-calling clinical imaging agent from scratch, implementing the reasoning loop, tool schemas, tracing and structured output without an agent framework.
> Designed six tools including CPU inference for chest X-ray classification and lung segmentation, deterministic cardiothoracic ratio calculation, and BM25 retrieval over 3,826 radiologist reports.
> Built a 43-case evaluation harness measuring tool selection accuracy, groundedness, task success and cost per query, reaching 89.5% tool selection accuracy at $0.0008 per query, and used it to detect the agent fabricating citations from an external API.
> Deployed as a public app with 133 tests in CI, containerised setup, and an explicit non-diagnostic safety boundary.
