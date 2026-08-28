# Decisions and challenges log

Three lines per entry: what happened, what I tried, what I chose and why.
Written as I go. These become STAR stories for interviews.

---

## 2026-08-18 — Python 3.15 alpha broke the install before it started

**What happened.** My default `python3` was 3.15.0a7. PyTorch publishes no wheels
for alpha releases, so `pip install torch` would have tried to build from source
and failed after a long wait with an opaque compiler error.

**What I tried.** Checked what else was on the machine before installing
anything: `/usr/bin/python3` (3.9.6, too old for some deps), and Homebrew's
`python@3.11`.

**What I chose and why.** Pinned the venv to Python 3.11 explicitly rather than
using `python3 -m venv`. Lesson: on a project with heavy compiled dependencies,
pin the interpreter version deliberately instead of inheriting whatever `python3`
happens to point at. A one-line change now, a lost evening later.

---

## 2026-08-18 — Double sigmoid: the bug that looked like a working model

**What happened.** The first run of `classify_xray` returned every one of the 18
pathologies above 0.5, all bunched between 0.50 and 0.65. It looked like a
working model with an unremarkable image. It was a bug.

**What I tried.** The clustering was the clue: real multi-label outputs spread
out, they do not all sit just above the threshold. I read
`torchxrayvision.models.DenseNet.forward` and found:

```python
if hasattr(self, "op_threshs") and (self.op_threshs is not None):
    out = torch.sigmoid(out)
    out = op_norm(out, self.op_threshs)
```

The model already applies sigmoid AND operating-point normalisation internally.
My extra `torch.sigmoid()` mapped the already-calibrated `[0,1]` outputs through
sigmoid again, which compresses `[0,1]` into `[0.5, 0.73]`.

**What I chose and why.** Removed my sigmoid and used `model(img)` directly. The
values now spread 0.01 to 0.62 with Cardiomegaly ranked first, which matches the
NIH ground-truth label for that sample image.

Two lasting changes came out of this. First, a regression test that asserts the
*spread* of probabilities rather than specific values, because asserting values
would break every time the weights change while asserting spread catches exactly
this class of bug. Second, a rule: before wrapping any pretrained model, read its
`forward()`. The interface is not the documentation.

**Why this is the most important entry in this file.** The failure was silent.
Nothing crashed, nothing warned, and the output was plausible enough to build on
for weeks. In a clinical tool, a silent miscalibration is worse than a crash.

---

## 2026-08-18 — Tools raise, the agent catches

**What happened.** Needed to decide whether tools return `{"error": ...}` or
raise exceptions.

**What I tried.** Returning error dicts is convenient for the agent, which needs
to feed failures back to the model as text. But it means every caller must
remember to check `result["ok"]`, and a forgotten check silently produces a wrong
clinical answer downstream.

**What I chose and why.** Tools raise a single `ToolError` type carrying a
message and a `recoverable` flag. The agent's dispatcher is the ONE place that
converts an exception into a tool result the model can read. This keeps failures
loud in tests and in the CLI, while still letting the agent recover.
`recoverable=False` tells the model not to bother retrying a bad file path.

---

## 2026-08-18 — Masks go on disk, handles go in the conversation

**What happened.** `segment_lungs` produces a 14x512x512 array. `compute_ctr`
needs it. But tool results are serialised into the LLM's context.

**What I tried.** Considered returning mask summary statistics only and having
`compute_ctr` re-run segmentation. That doubles inference cost for every CTR.

**What I chose and why.** `segment_lungs` writes an `.npz` to `artifacts/` and
returns the path as a `mask_handle`. `compute_ctr` takes that handle. Big data on
disk, small handle in the conversation. This is the pattern that stops an agent's
context window exploding, and it makes the two tools independently testable.

---

## 2026-08-18 — CTR denominator is an approximation, and it is biased

**What happened.** Textbook CTR divides maximum cardiac diameter by maximum
INNER RIB CAGE diameter. Our segmentation model gives lung fields, not ribs.

**What I tried.** Considered using the `Spine` and rib-adjacent channels the
PSPNet also outputs to estimate the rib margin. They are noisier than the lung
masks and would add a failure mode for a small accuracy gain.

**What I chose and why.** Use the widest lung-field extent as the denominator,
and state the bias direction explicitly in the tool's own output. The lung fields
sit just inside the ribs, so our denominator is slightly small and our CTR is
slightly HIGH. That errs toward over-calling cardiomegaly rather than missing it,
which is the safer direction for a screening-flavoured tool. Documenting the
direction of a known bias is more useful than pretending it is not there.

---

## 2026-08-18 — Refusing to report physiologically impossible ratios

**What happened.** If segmentation fails, the geometry still returns a number.
A CTR of 0.95 is not a remarkable patient, it is a broken mask.

**What I chose and why.** `compute_ctr` flags anything outside 0.20-0.85 as
`plausible: false` and its interpretation string says "Do not report this value."
The tool refuses at the tool layer rather than hoping the prompt catches it.
General principle: a safety constraint that can be enforced deterministically
should never be delegated to a language model.

---

## 2026-08-18 — CTR is only meaningful on a frontal film

**What happened.** CTR is invalid on a lateral view and unreliable on AP/supine
films because the heart is magnified. I initially planned to hedge in prose.

**What I tried.** Looked at what the IU dataset actually ships:
`indiana_projections.csv` has a `projection` column labelling every image
Frontal or Lateral.

**What I chose and why.** The fetch script joins on it and keeps only frontal
images in the working set, and carries `projection` into the report corpus. A
constraint I can enforce from data beats a caveat I have to write in a prompt.
Still an open limitation: Frontal covers both PA and AP, and the dataset does not
distinguish them, so the magnification caveat genuinely remains.

---

## 2026-08-18 — Stripping "XXXX" from the index but not from quotes

**What happened.** The NLM de-identified these reports by replacing names, dates
and some clinical terms with the literal token `XXXX`. It appears thousands of
times across the corpus.

**What I tried.** Considered stripping it from the stored text entirely.

**What I chose and why.** Two different texts for two different jobs. `index_text`
strips `XXXX` and is what BM25 indexes, because the token carries no signal and
inflates document length, which BM25's length normalisation then penalises.
`text` keeps it, and is what gets quoted as evidence. A quotation must match the
source record exactly, including its redactions, or the citation is not honest.

---

## 2026-08-18 — Exact lookup is a separate tool from search

**What happened.** "What does the report for CXR3821 say?" and "find reports
mentioning effusion" feel like the same tool. They are not.

**What I chose and why.** `get_report_by_image` does an exact join and returns
`found: false` when the case is absent. `search_reports` does fuzzy BM25 ranking.
If a lookup went through the fuzzy path, asking about a patient we do not have
would return the most similar OTHER patient's report, and the model would very
plausibly present it as that patient's. That is the single most dangerous failure
mode I have thought of so far, and it is designed out at the tool boundary rather
than mitigated in a prompt. There is an explicit test for it.

---

## 2026-08-18 — numpy 2 removed the `.ptp()` method

**What happened.** `array.ptp()` raised `AttributeError` on numpy 2.4.

**What I chose and why.** Use the free function `np.ptp(array)`, which still
exists. Minor, but worth logging: pinning `numpy>=1.26` without testing on 2.x
would have shipped a crash in the overlay renderer that no fast test covered.

---

## 2026-08-21 — The provider abstraction had to become real

**What happened.** The first `llm.py` passed the message list straight to
Gemini. That only worked by accident for a single-turn call and would have
broken the moment a tool result went back.

**What I tried.** Listed what the two providers actually disagree about:
assistant is `"model"` vs `"assistant"`; the system prompt is config vs a
message; tool results are a `function_response` Part vs a `role="tool"` message;
Gemini supplies no tool-call id; Groq returns arguments as a JSON *string*.

**What I chose and why.** One neutral message format, and each provider
translates in and out of it. None of that vendor noise reaches `agent.py`. This
is the concrete answer to "why not write directly against the SDK": Weekend 4
must run the identical eval against both providers, and that is only possible
because the differences are contained in one file. Groq is spoken over raw HTTP
rather than its SDK, which is about forty lines and one fewer dependency that
can break CI.

---

## 2026-08-21 — FakeProvider, or: how to test a loop at all

**What happened.** Needed to test the agent loop. Testing against a live model
tests the model, not the loop: slow, costs quota, non-deterministic, so a red
test tells you nothing.

**What I chose and why.** A `FakeProvider` that replays a scripted list of
responses and records what the agent sent it. That gives deterministic,
zero-cost tests, and lets me construct the situations that matter and that a
real model produces only occasionally: a hallucinated tool name, a wrong keyword
argument, a model that never stops calling tools. It raises rather than
returning when its script runs out, because a test that overruns its script is a
broken test and should say so. 20 loop tests, 2 seconds, no network.

---

## 2026-08-21 — Case id normalisation: a false negative in the safety-critical tool

**What happened.** `get_report_by_image("34_IM-1644-1001")` returned
`found: false` for a case that IS in the corpus.

**What I tried.** Traced the shapes. The file is `34_IM-1644-1001.dcm.png`; the
corpus stored `Path(filename).stem`, which leaves the `.dcm` on, giving
`34_IM-1644-1001.dcm`; the user and the model say `34_IM-1644-1001`. Three
spellings of one study, compared raw.

**What I chose and why.** One `normalise_id()` that strips stacked image
suffixes and case, applied to both the stored id and the query. Fixed the
generation side too so the corpus holds the clean id.

**Why it matters more than it looks.** This is the tool whose entire job is
exact matching, and it was silently failing to match. The direction was safe —
a false negative, "I don't have that case", rather than returning the wrong
patient — but it would also have tanked the Weekend 4 eval, and I would have
spent that weekend blaming the model. There is now a regression test covering
all four spellings plus three near-misses that must NOT match.

---

## 2026-08-21 — Validators are guarantees; prompts are requests

**What happened.** Needed the agent to stop short of diagnosing. The obvious
move is to write "do not diagnose" in the system prompt.

**What I chose and why.** Put it in the Pydantic model instead. `AgentAnswer`
rejects diagnostic and prescriptive language via a regex, `present` has no
"confirmed" member, evidence with `source="report"` must carry a verbatim quote,
a `ctr_value` must cite measurement evidence, and an answer with no findings
must explicitly set `unanswerable` or give a refusal reason.

A prompt is a request the model can decline; a validator is a guarantee. Both
are in place, but only one of them is testable, and there are now 14 tests that
assert the safety rules hold. One of those tests deliberately checks the rule is
not TOO broad: "the report states the cardiac silhouette is enlarged" must still
be allowed, or the agent becomes useless.

---

## 2026-08-21 — Repair with the error, not with a retry

**What happened.** When the model returns JSON that fails validation, the naive
fix is to ask again.

**What I chose and why.** Asking again with the same prompt just rerolls the
dice. Instead the actual Pydantic error is fed back, which names the field and
the rule that failed, and that is exactly the feedback needed to fix the output.
Budget is one repair, then give up and report the validation error rather than
looping. There is a test asserting the repair prompt contains the real error
text and not a generic nudge.

---

## 2026-08-21 — Grounding: str(dict) quietly breaks quote matching

**What happened.** The grounding check compares a quoted report line against
what the tools returned. My first version joined tool results with `str(r)`.

**What I tried.** A test with a report containing a newline failed. `str()` on a
dict renders values with `repr()`, so a real newline inside a report becomes the
two characters backslash-n. Any quote spanning a line break would fail to match.

**What I chose and why.** Walk the nested structure and collect string leaves,
keeping text as text. The failure mode this avoided is nasty: a GENUINE citation
reported as fabricated. A grounding check that cries wolf gets switched off, and
then the real fabrications get through too.

---

## 2026-08-21 — Tool results go in the trace

**What happened.** For grounding I needed what the tools returned. The trace only
recorded arguments, so my first version re-ran the tools.

**What I chose and why.** Record the full result in the trace event instead. Two
reasons. Re-running DenseNet just to verify a quote is wasteful, and worse, it
verifies the answer against a FRESH tool run rather than against what the model
actually saw. When those differ the grounding check is meaningless. Self-
contained traces also mean Weekend 4 can score groundedness offline from trace
files alone. Costs a few KB per run.

---

## 2026-08-21 — Both default models had been retired

**What happened.** First live call failed. `gemini-2.0-flash`: 404, "no longer
available, please update to models/gemini-3.6-flash". Groq:
`llama-3.3-70b-versatile` gone too.

**What I tried.** Rather than guessing replacements, asked each API what it
offers: `client.models.list()` and `GET /openai/v1/models`. Then probed the
candidates with a trivial call. `gemini-3.7-flash` returned 504
DEADLINE_EXCEEDED on every attempt (25s and 30s waits), as did the
`gemini-flash-latest` alias pointing at it. `gemini-3.6-flash` answered in ~2s.

**What I chose and why.** `gemini-3.6-flash` and `openai/gpt-oss-120b`, both
pinned to explicit versions rather than `-latest` aliases. An eval table saying
"gemini-flash-latest scored 0.86" is worthless three weeks later when the alias
moves: you cannot tell whether your numbers changed because of your code or
because of someone else's deploy. Pin it, record the pin in the results, bump
deliberately.

**Wider lesson.** Model availability is a runtime dependency that rots. Ask the
API rather than trusting a constant written weeks ago.

---

## 2026-08-21 — Gemini 3.x requires thought signatures to round-trip

**What happened.** With a working model, the first multi-step run died on:
`400 INVALID_ARGUMENT: Function call is missing a thought_signature in
functionCall parts`.

**What I tried.** Gemini 3.x thinking models attach an opaque
`thought_signature` to each function-call part, and reject the request if it is
absent when that call is replayed in the history. My translation layer built a
fresh `FunctionCall` from name and arguments and silently dropped it.

**What I chose and why.** Added `provider_meta: dict` to the neutral `ToolCall`
and round-trip it verbatim. The neutral layer deliberately does NOT know what a
thought signature is; it is opaque vendor data. That keeps the abstraction
honest: Groq needs nothing here, and the next provider's equivalent quirk needs
no change above that line. Signature bytes are stored latin-1 encoded because
they pass through the JSON response cache and bytes are not serialisable.

---

## 2026-08-21 — The cache hid the fix for its own bug

**What happened.** After fixing thought signatures, the run failed with the
byte-identical 400. Spent a confusing few minutes convinced the fix was wrong.

**What I tried.** Inspected `.cache/`. One entry held three parallel tool calls,
all with no signature: written during the *pre-fix* run and replayed straight
back into the fixed code.

**What I chose and why.** Added `CACHE_VERSION` to the cache key. When the SHAPE
of what we store changes, the version bumps and every old entry misses.

**Why this one is worth remembering.** The bug was fixed and invisible at the
same time, and the cache made a correct fix look like a failed one. Any cache
keyed only on inputs, never on the format of what it stores, can do this. It is
also an argument for `--no-cache` existing as a first-class flag rather than
something you comment out.

---

## 2026-08-21 — No timeout on the LLM call

**What happened.** A probe hung for two full minutes with no error.

**What I tried.** Groq had `timeout=60`. The Gemini client had none, so a 504-ing
model wedged the process indefinitely.

**What I chose and why.** One `LLM_TIMEOUT_S = 60` applied to both providers. I
had written "a tool that hangs forever hangs the whole agent loop" in the PubMed
tool on day one and then failed to apply it to the LLM call, which is the thing
every single iteration depends on. The principle was right; I just did not carry
it far enough.

---

## 2026-08-21 — First live runs: the two providers disagree about how much to check

**What happened.** Same question, both providers, both correct.

Gemini called four tools: `get_report_by_image`, `classify_xray`,
`segment_lungs`, `compute_ctr`. It cross-checked the report against the
classifier (0.0077) and the geometry (CTR 0.402), and reported both caveats.
38s, 3 LLM calls, 6,347 input tokens.

Groq called one: `get_report_by_image`. It answered from the report alone,
correctly. 1.9s, 2 LLM calls, 2,607 input tokens.

**What this tells me.** Both answers are defensible; they differ in thoroughness,
and the cheap one is 20x faster. Which is *better* depends on the question, and
"the report says the heart is normal" is weaker evidence than "the report, the
classifier and the geometry all agree". This is precisely the trade-off Weekend
4's tool-selection metric has to capture, and it is why the gold set needs an
expected tool *sequence* rather than just an expected answer. Noted now so I
build the metric with this case in mind.

**Adversarial checks, both passed on the first try.** Asking for `CXR9999999`
returned "not available in this corpus" with no substitution. Asking what to
prescribe produced a refusal plus an offer of what the tool can actually do.

---

## 2026-08-21 — The eval set the default fetch would have given me was useless

**What happened.** Started building the gold set from the 40 images I had. They
were the first 40 frontal images in corpus order: 15 normals, 2 cardiomegalies.

**What I tried.** Considered writing the gold set anyway and noting the
imbalance. But two cardiomegaly cases cannot say anything about the cardiomegaly
path, which is the project's headline capability, and a metric computed over
them would still print a confident percentage.

**What I chose and why.** `scripts/fetch_stratified.py` samples per label from
the dataset's own MeSH-derived `problems` column against a fixed seed: 30
cardiomegaly, 20 effusion, 15 atelectasis, 15 opacity, 40 normal. The seed makes
the eval set reproducible by anyone who clones the repo.

**Lesson.** An unbalanced eval set does not fail loudly. It produces a confident
number that hides the failure you care about most.

---

## 2026-08-21 — Ground truth is computed, not written

**What happened.** Needed expected answers for 30 derived gold cases.

**What I chose and why.** `evals/compute_ground_truth.py` runs the deterministic
tools over real cases and records what they actually produce; expected quotes are
copied verbatim from real reports. Nothing in the gold set is invented. If a gold
answer and the system disagree, exactly one is wrong and re-running the script
tells me which.

**What this surfaced immediately.** Mean CTR is 0.530 for cardiomegaly-labelled
cases and 0.464 for normals, so the measure separates — but three normal cases
measured 0.533, 0.529 and 0.525, i.e. above the 0.50 threshold. That is the
documented high-bias caveat appearing with numbers behind it. The classifier
separates far better (0.522 vs 0.134). Those three became `difficulty: hard`
cases where the report and the measurement genuinely conflict and the agent must
surface the disagreement rather than pick a side.

---

## 2026-08-21 — Free-tier rate limits are an engineering constraint, not an annoyance

**What happened.** The first full adversarial sweep on Gemini managed 4 cases in
10 minutes. Server-supplied `retryDelay` values were 25-59 seconds, on nearly
every call.

**What I tried.** First reaction was to reduce the case count. That is solving
the wrong problem: it would shrink the eval to fit the quota rather than make the
eval survive the quota.

**What I chose and why.** Three changes.

1. `with_rate_limit_retry` in the provider layer, honouring the server's own
   `retryDelay` hint and falling back to exponential backoff. Crucially, a 429 is
   retried rather than surfacing to the harness, because **a rate limit is not a
   system failure and must not be scored as one.** Before this, quota exhaustion
   showed up in the results as the agent failing the case.
2. Every scored case is appended to `<provider>_partial.jsonl` immediately, and
   `--resume` skips what is already there. A run that dies at case 38 keeps 37.
3. A `--delay` flag for deliberate pacing.

**Lesson.** The interesting bug was not the 429. It was that the 429 was being
recorded as a failed test case, which would have made the agent look worse than
it is and sent me debugging the agent instead of the harness.

---

## 2026-08-21 — A metric that cannot assess a case must not score it as a pass

**What happened.** Writing the groundedness scorer, the obvious implementation
returns "pass" when an answer contains no quotes to check.

**What I chose and why.** It returns `applicable: false` instead, and the
summary averages only over applicable cases and reports the n. An answer with no
quotes is not grounded; it is unassessed by this metric. Scoring it as a pass
would inflate the headline number with exactly the cases the metric is blind to,
and the blindness would be invisible in the output.

Same reasoning for `score_deterministic_checks`, which returns `None` rather
than `True` when a case declares no checks.

---

## 2026-08-21 — Cross-provider judging

**What happened.** Needed an LLM judge for the one metric that resists
determinism: did the answer actually address the question within its boundary.

**What I chose and why.** The judge is always the *other* provider — Groq grades
Gemini, Gemini grades Groq. A model grading its own output scores it generously.
This is a cheap mitigation, not a solution: both are LLMs and their blind spots
correlate. That is precisely why the deterministic checks stand alongside it and
why 10 cases still get read by hand. Written into `docs/evaluation.md` as a
stated limitation rather than left as an implementation detail.

---

## 2026-08-21 — The first eval failure was my gold set, not the agent

**What happened.** First CTR case came back FAIL. The agent had called the right
tools in the right order, so the failure was on the answer checks.

**What I tried.** Read the answer. It was excellent: CTR 0.656 exactly right, the
method note about lung-field versus inner rib margins, the PA/AP magnification
caveat, and an explicit "this cannot be concluded from the current data". Then I
read my own check: `expected_concepts: ["caveat"]`.

The agent had written a textbook caveat without using the word "caveat". Nobody
does. I had written a check for the label rather than the substance.

**What I chose and why.** Match the substance: `magnif`, `rib margin`,
`approximat`, `pa vs`, `ap film`, `projection`, `view is unknown`. The rejected
answer now matches six of them.

**What it changed structurally.** Fixing a *scoring* bug should not cost another
full sweep of the agent, because in practice that means you stop fixing scoring
bugs. So I split them: `evals/rescore.py` re-scores saved answers against the
current gold set with no LLM calls at all. Re-scoring took one second and moved
deterministic pass rate from 75% to 100% on the cases run so far; re-running
would have cost twenty minutes and another slice of the daily quota.

**The lesson worth keeping.** An eval's first job is to be wrong loudly enough
that you check it. My instinct on seeing FAIL was to look at the agent. The
answer was better than my test. Always read the failing output before believing
the metric, especially early, when the harness is younger and buggier than the
thing it is measuring.

---

## 2026-08-21 — Four failures in a row were my checks, not the agent

**What happened.** The first sweep produced a run of failures. In order:

| Case | Why it "failed" | What the agent actually did |
|---|---|---|
| `ctr-*` | required the literal word "caveat" | wrote a textbook caveat without the word |
| `cardio-yes-1031` | regex `diagnos\w*` matched "diagnosis" | wrote "does not constitute a formal diagnosis" |
| `classify-17` | required the cardiomegaly probability | correctly said nothing was flagged, so did not quote it |
| `search-effusion` | required the phrase "other patients" | quoted three real reports with UIDs and image ids |

Four failures, four correct answers, four badly written checks.

**The pattern.** Every one of them tested for a **label describing the behaviour**
rather than **text that would actually appear**. I wrote "caveat" when I meant
"mentions magnification or the rib-margin approximation"; "other patients" when I
meant "cites the cases it found". A person reading these answers would pass all
four instantly.

**What I chose and why.** Rewrote every concept check to match substance:
`magnif`, `rib margin`, `approximat`, `im-`, `uid`. Where the check was
reasonable but the QUESTION was wrong, I fixed the question instead: the classify
cases now ask "what probability does it give for cardiomegaly" rather than the
open "what does it flag", because the old question and its check were asking for
different things.

**The general rule I am taking from this.** *The check must test what the
question asked.* If the question is open-ended, the check must be too. My checks
had drifted into asserting one specific correct answer to questions that had
several.

**The bug that mattered.** The `diagnos\w*` one was not confined to the eval. The
identical over-broad pattern was in `radreport/schema.py`, the shipped
`AgentAnswer` validator, where it would have forced repair loops on good answers
and slowly taught me that the safety check was noise. The eval found a production
safety bug that 97 unit tests did not, because the unit tests only asserted what
I had already thought of. That is the argument for having both.

The pattern now handles polarity and negation, is verified against 19 phrasings
(8 that must flag, 11 that must pass, including "the patient has no acute
cardiopulmonary disease"), and the gold set imports it from `schema.py` rather
than keeping a second copy that could drift.

**Uncomfortable meta-lesson.** For the first hour, every red result made me look
at the agent. The harness was younger and buggier than the system it was
measuring, and I should have expected that. Read the failing output before
believing the metric.

---

## 2026-08-21 — The eval caught a fabricated citation

**What happened.** Groundedness came back at 52.9%, which reads like rampant
fabrication. Most of it was not. Checking the unsupported quotes one by one
found two completely different causes hiding behind one number.

**Cause one, cosmetic (8 of 10).** Models rewrite punctuation as they quote.
`well-aerated` came back with a U+2011 non-breaking hyphen, apostrophes came
back curly, and markdown emphasis was inserted INSIDE the quotation marks:
`"The lungs are clear. **Heart size within normal limits**."` Not one word had
changed. I now normalise the typographic layer away (unicode dash/quote folding,
markdown stripping) before comparing, in both the eval scorer and the shipped
`verify_grounding`. A metric that flags eight harmless reformattings for every
real problem is one people learn to ignore.

**Cause two, serious (2 of 10).** Asked why CTR is unreliable on AP films, the
agent produced:

> Sahin et al. reported that AP films "systematically overestimate cardiac
> silhouette size compared with PA films" 【2†L2-L4】

`search_literature` returns titles, journals, years, authors and PMIDs. It does
not return abstracts — I decided that on Weekend 1 to save context. So the agent
had never read a word of those papers. It invented direct quotations, attributed
them to named authors, and generated citation markers to match.

**Why this is the most important finding of the weekend.** This is the failure
mode I would least want in a clinical tool and the hardest for a human to catch,
because a fabricated citation reads exactly like a real one. Nothing crashed.
The tool selection was correct. It would have passed a demo.

**What I chose and why.** Three changes.

1. The system prompt now states explicitly that `search_literature` returns no
   abstracts, that the agent has therefore not read the papers, that it may cite
   a title and PMID but must never put words in quotation marks and attribute
   them to a paper, and what to say instead.
2. A new gold case, `adv-fabricated-citation`, asks directly for quotations from
   papers. Its check flags `et al. reported/found/showed` and the 【 citation
   markers. Verified against the real fabricated output.
3. Kept the prose-level groundedness check, because this is exactly what it is
   for. The structured path already had the defence: `verify_grounding` compares
   every quote against actual tool output, and would have rejected this.

**The wider lesson about the metric.** My instinct on seeing 52.9% was that the
number was broken. It was, partly — but underneath the noise was a real and
severe defect. Fix the noise so the signal is legible; do not dismiss the whole
metric because most of its output is noise. If I had "fixed" groundedness by
loosening it without reading the failures, I would have deleted the only check
that caught this.

True verbatim rate against the report corpus, after typographic normalisation:
41 of 51 quotes. Of the remaining 10, 8 were quotes from tool output rather than
report text (correctly grounded, wrong corpus in my ad-hoc check) and 2 were the
fabrication above.

---

## 2026-08-21 — A quota wall is not a failing test case

**What happened.** The Gemini adversarial run finished with 8 of 12 cases
"failing". Reading the failures: every one carried
`ClientError: 429 RESOURCE_EXHAUSTED`. The daily free-tier quota was gone.

**What I tried.** I already had `with_rate_limit_retry`, added earlier the same
day for exactly this. It was not enough: it absorbs *transient* 429s, but a
genuinely exhausted daily quota outlasts a five-attempt budget, and then the
error propagates into the harness and lands in the results as a failed case.

**What I chose and why.** `is_quota_error()` classifies the error, and such
cases are marked `excluded` and removed from every metric, counted separately.
The summary prints the exclusion count and refuses to present a result computed
from fewer than 10 cases.

Retroactive too: `rescore` applies the rule to records written before it
existed, so the earlier Gemini run reports honestly as "4 cases scored, 8
excluded" rather than 75% / 36.4%.

**Why this matters more than it sounds.** Those numbers were about to go in the
README as Gemini's performance. They measure Google's rate limiter. Publishing
them would have been a false claim about a model, and worse, believing them
would have sent me looking for an agent bug that does not exist. The general
form: *an evaluation must distinguish "the system was wrong" from "the system
was never asked".* Anything that cannot be measured must be reported as
unmeasured, never as zero.

I had written almost this exact sentence in this file eight entries ago, about
groundedness scoring unassessable cases as passes. Same principle, opposite
direction, and I still had to be shown it twice.

---

## 2026-08-22 — The UI is the trace, not the chat

**What happened.** Building the Streamlit app, the obvious design is a chat box
and an answer.

**What I chose and why.** The answer is the least interesting thing on the page.
Anyone can put a text box over an LLM. What is worth showing is the trace panel
underneath: every tool call, its arguments, its latency, and the exact JSON the
model received back, expandable. Plus three one-click adversarial examples
(missing case, out of scope, fabrication bait), because a reviewer will not think
to ask an adversarial question, and the safety behaviour is the part worth
seeing.

Without a visible trace a reviewer has no basis for trusting the answer. Neither
do I.

---

## 2026-08-22 — Three defects the browser would not have shown me

**What happened.** The app served HTTP 200 and looked fine. A curl only fetches
the Streamlit shell, though; the page renders over a websocket, so "200 OK" says
almost nothing about whether the app works.

**What I tried.** `streamlit.testing.v1.AppTest`, which runs the script headless
and exposes the widget tree. It found three real problems:

1. `use_container_width` is deprecated with a removal date of 2025-12-31, which
   has already passed. Still working, but on borrowed time.
2. A dead line: `has_key = bool(...) or key_name in st.secrets if hasattr(...)
   else False`. Tangled precedence, and the variable was never read. Replaced by
   asking the provider whether it can construct itself, which is the thing that
   actually knows.
3. The adversarial example buttons did not work. They assigned to a local
   `question` variable **after** the text area had already rendered, so clicking
   one did nothing visible. Fixed by writing `st.session_state` before the widget
   is created.

**Why number 3 is the interesting one.** It is invisible in code review, invisible
in a screenshot, and would only surface when someone clicked the button during a
demo. `AppTest` caught it in about a second. There is now a test asserting each
button populates the box, and one asserting the safety banner is present, because
the banner is the single thing on that page that must never silently disappear
in a refactor.

---

## 2026-08-22 — Docker: what I verified and what I did not

**What happened.** Wrote the Dockerfile. The Docker daemon was not running on
this machine, so I could not build the image.

**What I tried.** Verified everything that could be verified without a daemon:
`docker compose config` parses, the weight-caching command runs correctly under
the local interpreter, and `/_stcore/health` really does return `ok` (both are
the steps most likely to be silently wrong in a Dockerfile you cannot build).

**What I chose and why.** Shipped it with the limitation stated plainly in the
README rather than implying it was tested, and added a CI job that builds the
image on every push. The image is the one artefact unit tests cannot cover, and
it breaks quietly: a wheel index change or a stale weights URL surfaces only at
build time.

Three decisions inside it worth keeping: torch from the CPU wheel index (the
default Linux wheel bundles CUDA, ~2 GB this project never touches); weights
baked in at build so the first request does not block on a download; and the
dataset mounted read-only rather than copied, because baking 500 MB of medical
images into a distributable layer is how licence terms get violated by accident.

---

## 2026-08-22 — Committed evidence must come from the same computation as the claim

**What happened.** Exported `results_summary.json` as the committed evidence for
the README's numbers. It said deterministic checks 75.0%. The README said 97.2%.

**What I tried.** Both were "true": the summary was exported from the raw records,
the README figure came after re-scoring against a fixed gold set. Two honest
numbers, one of them stale.

**What I chose and why.** `export_summary` now re-scores against the current gold
set before writing. Evidence that disagrees with the claim it supports is worse
than no evidence: it makes the whole results section look careless, and a reader
who checks is right to stop trusting the rest.

---

## 2026-08-22 — Profiling before deploying, and what it forced

**What happened.** About to deploy to Streamlit Community Cloud. Measured the
app first rather than pushing and hoping.

**What I found.** PSPNet segmentation peaks at **1,816 MB** RSS. The free tier
gives roughly 1 GB. torch plus cached weights are ~820 MB on disk. And the
dataset is gitignored, so a fresh clone has no images at all. The app would have
installed, started, and been killed by the OOM reaper the first time anyone
toggled the overlay — after which I would have been debugging a hosting
dashboard instead of a profiler.

**What I tried.** Considered dropping the imaging tools from the public demo and
shipping a retrieval-only version. That guts the part of the project that is
actually distinctive.

**What I chose and why.** Run the real models ONCE, offline, over 40 cases;
commit the results; serve those in the deployment. `RADREPORT_DEMO=1` switches
`radreport/tools/__init__.py` to import `demo.py`, so torch is never imported —
the import is conditional rather than a runtime branch precisely so the deployed
environment does not need the dependency installed. Peak RSS 1,816 MB → **71 MB**,
and the values are byte-identical to the real outputs because they ARE the real
outputs (verified: Cardiomegaly 0.5849, CTR 0.520 on both paths).

**The honesty constraint, enforced in code.** Every demo result carries
`precomputed: True` and a note, and the UI shows a banner. "This model runs on
your image in 40 ms" and "I ran this model last Tuesday on 40 images" are
different claims. A case outside the cached set raises a clear error rather than
falling back to anything — a demo that fabricates is worse than no demo.

**The deploy trap I built a guard for.** If the deployment installs
`requirements-deploy.txt` (no torch) without setting `RADREPORT_DEMO=1`, the
result is a bare `No module named 'torch'` on a hosting dashboard. The package
now catches that ImportError and explains exactly what to set and why.

**The interview version.** "I profiled it before deploying, found segmentation
peaked at 1.8 GB against a 1 GB limit, and shipped a precomputed-inference
variant so the public demo keeps the full user experience within the free tier."
That is an engineering answer. "I couldn't deploy it" is not.

---

## 2026-08-24 — First push failed CI: works on my machine, in its purest form

**What happened.** Pushed to GitHub. The `fast` job failed with six errors, all
of which pass locally. The `docker` job passed, which at least resolved the one
thing I could not verify before: the image builds.

**What I tried.** Read the log rather than re-running it. Two causes, one class:

1. Five failures: `SystemExit: No gold set at .../evals/gold_set.jsonl`. Those
   tests validate real properties of the gold set — every named-case question
   forbids fuzzy search, the absence cases use regexes rather than brittle
   substrings — and I had gitignored the file they read.
2. One failure: `test_trace_records_tool_results_for_offline_scoring` asserted
   `result["ok"] is True` on a `search_reports` call. That needs
   `data/reports.csv`, also gitignored.

Every one of them depended on a file my working tree had and a clone does not.

**What I chose and why.**

*Committing the gold set*, reversing my own earlier decision. I ignored it for
consistency with the corpus, but rebuilding it needs the images AND the models,
so a fresh clone genuinely cannot regenerate it — the file was unreachable in
CI, not merely absent. Symbolic consistency was costing real test coverage. What
actually ships is ~8 single sentences from a public-domain, de-identified
collection with no data use agreement.

*Fixing the trace test to test its own claim.* It asserts that the trace records
the tool RESULT, not that the tool succeeded. A tool that errors must have its
result recorded too — that is exactly the case offline scoring needs to see. It
now asserts the result is present and well-formed, and checks `hits` only when
the corpus is there.

*A guard so it cannot recur.* `scripts/check_fresh_clone.sh` copies precisely
the files git would ship into a temp directory and runs the fast suite there.
Ten seconds, and it reproduces CI exactly. Run before every push.

**The lesson.** I had a green 133-test suite and pushed with confidence. The
suite was green because my working directory held four gitignored files that the
tests silently depended on. A test suite validates the tree you have, not the
tree you ship, and nothing warns you when those diverge. The fix is not
discipline, it is a script that makes the shipped tree the thing under test.

---

## 2026-08-24 — The README diagram shipped broken

**What happened.** GitHub replaced the architecture diagram with
`Lexical error on line 11. Unrecognized text.` The first visual on the repo's
front page was a parser error.

**What I tried.** Reproduced it locally rather than guessing, by running the real
Mermaid parser under jsdom. Got the identical error, caret and all. The offending
line:

```
T2 -.mask_handle .npz.-> T3
```

Mermaid's dotted-link-with-label syntax is `-. text .->`. My label contained a
literal `.` in `.npz`, which collides with the `.->` terminator, so the lexer
gave up mid-label.

**What I chose and why.** Dropped the dots: `T2 -. mask_handle .-> T3`. Then
added `scripts/check_mermaid.mjs` and a CI job that parses every Mermaid block in
every Markdown file on each push.

**Why it is worth a CI job.** No Python test can see this. 133 tests passed, the
fresh-clone check passed, and the front page was still broken. Documentation
rendering is a separate failure surface from code, and on a portfolio repo it is
the surface a reviewer hits first — before a single test result, before any code.
The one artefact everyone sees was the one artefact nothing verified.

Same shape as the CI failure two entries up: I verified the thing I was thinking
about and not the thing the audience actually encounters.

---

## 2026-08-25 — The data policy conflated licence with size, three times

**What happened.** With demo mode finally working, the deployed app crashed on a
full traceback: `get_report_by_image` -> `load_corpus` -> ToolError, because
`data/reports.csv` is not in the repo.

**What I tried.** This was the third bug from the same root, and I had patched
around it twice already rather than fixing the rule:

1. Five CI tests failed because `evals/gold_set.jsonl` was ignored.
2. One CI test failed because `data/reports.csv` was ignored.
3. Now the deployment itself, for the same missing file.

The original rule was "never commit medical images or the report corpus". Safe-
sounding, and wrong, because it treats two different reasons as one.

**What I chose and why.** Wrote the policy out properly, once:

- **MIMIC-CXR**: credentialed, has a data use agreement. Never commit. This
  project does not touch it, deliberately.
- **IU reports**: public domain (NLM Open-i), de-identified with XXXX
  placeholders, no DUA, 1.3 MB. **Committed.** Without it the deployment has no
  corpus at all, so BM25 retrieval over 3,826 reports and exact case lookup are
  both dead, and the sidebar advertises them as features.
- **IU images**: same licence, but ~500 MB. Excluded for **size**, not licence.
  40 downscaled thumbnails ship inside `data/demo_cache.json`.

Licence and size are different constraints and deserve different answers. One
blanket rule covering both produced a repo that was cautious about the wrong
thing while shipping a broken app.

---

## 2026-08-25 — One bare tool call took down the whole page

**What happened.** The corpus error did not degrade a panel, it replaced the
entire app with a red traceback. The image had rendered, the case list worked,
the agent was ready. All of it vanished because one call in an expander was
unguarded.

**What I chose and why.** Guarded that call, then went looking for the others
with an AST walk rather than by eye, and found two more (`image_bytes` in both
image helpers). There is now a test that parses `app.py` and fails if any tool
call sits outside a try/except.

**Why the test rather than just the fix.** Streamlit renders an uncaught
exception as a wall in place of the app, so the blast radius of any single
missing file is the entire page. In a CLI a bad tool call loses you one command;
in a Streamlit app it loses you everything the user could otherwise still do.
That asymmetry is worth encoding as a rule rather than remembering.

The agent's own `dispatch()` has had exactly this property since Weekend 2 -- it
never raises, so one failing tool becomes a message the model recovers from. I
wrote that principle down, then failed to apply it one layer up in the UI.

---

## 2026-08-27 — The worst number in the README was measuring punctuation

**What happened.** Groundedness read 52.9% — the single ugliest figure in a
project whose whole pitch is verifiable grounding. The README also said
41/51 quotes were verbatim, which is 80%. Two numbers, same run, no way for a
reader to tell which was the claim.

Both were true, and neither was right.

The 41/51 was quote-level and the 52.9% was case-level — different denominators
presented as if they were the same measurement, which is a reporting failure on
its own. But the case-level number was also computed by a definition that no
longer existed anywhere in the codebase. `metrics.py` had grown unicode and
markdown folding at 23:36 on 21 August; the groq run had finished at 23:35. The
run's stored verdicts were from the strict version, and `rescore.py` — the tool
whose entire purpose is re-scoring saved answers under a fixed metric — could
not fix them, because groundedness is the one metric that needs the tool
results, and those records predated the `trace_path` field. So it silently kept
the original verdict. The fix to the metric had been shipped, tested, and had
no effect on the number anyone read.

**What I found when I recovered it.** The traces were still on disk, and their
`start` event records the question, so all 38 records could be matched back to a
unique trace and re-scored. Under the current metric: 70.6%. Under the metric
after this session's work: 88.2%, with a verbatim rate of 82.4%.

Every point of that recovery was punctuation:

| quote | source | difference |
|---|---|---|
| `mildly enlarged for technique.` | `...mildly enlarged for technique,` | a full stop |
| `No acute cardiopulmonary disease.` | `no acute cardiopulmonary disease` | a full stop |
| `follow‑up imaging in 6 months` | `followup imaging in 6 months` | a hyphen the model added |
| `7 mm right upper lobe lung nodule` | same, U+202F vs space | invisible |

Two real defects survived, and they are the two worth having a metric for: the
fabricated literature quotations, and one answer that welded two non-adjacent
sentences into a single quotation behind an ellipsis.

**What I chose and why.** Three things.

*One definition, one place.* `radreport/grounding.py` now holds the check, and
both `schema.verify_grounding` (runtime) and `evals.metrics` (offline) call it.
They had been two copies of a safety rule with two different normalisation
tables, which meant the harness could report fabrication where the product had
correctly seen a real citation. Two copies of a safety rule is one safety rule
plus a bug waiting to be written.

*Three verdicts per quote, not one bit per answer.* `verbatim`, `repaired`,
`unsupported`. Grounded means no unsupported quotes; verbatim rate is reported
next to it with its own denominator. The two figures can no longer disagree
because `summarise()` computes both from the same records and `print_summary`
prints them adjacent.

*Repair, not tolerance.* A near-miss quote is matched back to its source span
and the answer is rewritten to the radiologist's actual characters, traced as a
`quote_repair` event. This is better than loosening the check: the reader ends
up with a quote they can paste into ctrl-F and find.

**The decision I nearly got wrong.** My first repair rule accepted any span
above 0.94 similarity. Then I wrote the test I should have written first:

> `"There is no pleural effusion and no pneumothorax"` → `"There is pleural
> effusion and no pneumothorax"`

That drops a negation, inverts the clinical meaning, and scores **0.968**. A
similarity threshold would have waved it through and called it a typo.

So repair is not decided by similarity at all. Similarity only *locates* the
candidate span; acceptance requires that the letters and digits of the quote and
the span be identical in order. That is blind to every punctuation, spacing and
hyphenation difference — which is all repair is for — and blind to nothing else.
Similarity is still reported, because knowing a rejected quote scored 0.30 (a
fabrication) rather than 0.80 (a stitched-together real one) is the difference
between two very different bugs.

The general lesson: when a safety check is being relaxed, the test that matters
is not "does the false positive go away", it is "does the thing the check exists
for still fail". I had the first test. I wrote the second one only because the
threshold felt arbitrary while I was typing the constant.

---

## 2026-08-27 — The imaging agent that would not look at the image

**What happened.** Tool selection on `cardiomegaly_assessment` was 50%, and the
README had already conceded the point as "arguably a defensible reading of a
two-part question". Asked *"Does X show cardiomegaly, and what does the report
say?"*, the model read the report and answered from the text alone.

**What I chose and why.** The gold set was right and the prompt was wrong, so I
fixed the prompt rather than the question. There is a real distinction the
system prompt had never drawn: what the report SAYS is a lookup, what the image
SHOWS is an observation, and the report is a second reader's prior opinion
rather than evidence about the pixels in front of you. That is now a rule.

50% → **100%**, all eight cases, first run after the change.

**What it cost, stated because it is not free.** Median latency on those cases
went 17.4s → 47.1s and cost per query roughly doubled, because the agent now
runs segmentation or the classifier where it previously ran one lookup. That is
the correct trade for an imaging agent and it is still under a fifth of a cent
per query, but "we fixed the metric" without "and here is the bill" is how
benchmarks get gamed.

**The thing I want to remember.** I had written the excuse into the README
before I tried the fix. "Arguably defensible" was doing work that a two-sentence
prompt rule did better. A limitation you have rationalised in prose is one you
have stopped trying to fix.

---

## 2026-08-27 — Embeddings lost, which is the useful result

**What happened.** Weekend 1 wrote down a hypothesis — BM25 "has no idea that
'cardiomegaly' and 'enlarged heart' mean the same thing" — and it sat in the
limitations list for three weekends as an assumption nobody had priced.

**What I chose and why.** Twelve findings, each queried twice: once in
radiologist vocabulary, once as a non-radiologist would type it. Same relevance
judgements (the NLM's own MeSH labels) across both. Reporting one blended
average would have hidden the entire effect, since a retriever that is superb on
half the queries and useless on the other half averages out to "adequate".

|  | clinical | lay | drop |
|---|---|---|---|
| bm25 | **0.911** | 0.266 | 70.8% |
| embedding | 0.789 | **0.364** | 53.9% |
| hybrid (RRF) | 0.870 | 0.323 | 62.9% |

The hypothesis was right and the obvious remedy was wrong. BM25 does collapse on
paraphrase — three findings go to exactly zero. But the dense retriever
*loses* on clinical phrasing, which is the condition the agent actually operates
in, since it builds its own queries out of clinical vocabulary. And its win on
the lay set is a win at 0.364 nDCG, which is not a working retriever, it is a
less broken one.

So BM25 stays the default and the tool grew a `retriever` parameter instead of a
replacement. The test asserting the default is deliberate: flipping it should
have to change a test, because the default is now a measured decision rather
than an accident of which one was written first.

**The finding I was not looking for.** Both retrievers rank

> "The heart is **not** enlarged. Lungs are clear."

in the top three for `enlarged heart`. Neither model handles negation, and in a
corpus where much of every report is a list of absent findings, that is a worse
defect than the vocabulary gap. The aggregate metrics cannot see it: that report
is labelled `normal`, so retrieving it costs score exactly as an unrelated
report would. The numbers say both retrievers did worse; only reading the hits
says they returned the opposite of what was asked for.

That is the argument for `--show` existing at all. I built it to debug the
harness and it found the thing the harness could not measure.

---

## 2026-08-27 — Building the image found three things the host suite could not

**What happened.** The README had carried an honest admission since Weekend 5:
the compose file validated, both risky Dockerfile steps were reasoned about, but
the image had never been built because the Docker daemon was not running. I
started the daemon and built it.

It built, the container reached `healthy` in ten seconds, and `/_stcore/health`
returned 200. Then I ran the test suite *inside* the image, which is the part
that was actually worth doing.

**Three defects, none of which the host could show me.**

*The sentence encoder ignored its baked-in weights.* `embed.py` passed
`cache_folder=CACHE_DIR/"sbert"`, which overrides `SENTENCE_TRANSFORMERS_HOME` —
so the weights the Dockerfile downloaded into `/opt/sbert` were invisible and
the container would re-fetch at runtime. It worked perfectly on a laptop, where
both paths point at a machine that has the model.

*And the fix for that was not enough.* With `--network none` the model then
loaded from cache, but only after ~100 seconds of HEAD requests for *optional*
config files (`adapter_config.json`, `video_preprocessor_config.json`, five
retries each) before falling back. The pre-download step prevented the failure
and not the hang. `HF_HUB_OFFLINE=1` in the image took the first embedding call
from 100s+ to 3s. This is the worse bug of the two: a failure gets reported, an
inexplicable one-time 100-second pause gets shrugged at.

*Two tests were lying about their dependencies.*
`test_deploy_requirements_exclude_torch` read `requirements-deploy.txt`, which
was never COPYed into the image, so it failed with `FileNotFoundError` rather
than a verdict. And `_case_selectbox` did `next(s for s in at.selectbox ...)`
with no default, so with no corpus mounted the UI tests raised `StopIteration` —
a crash that says nothing about the actual condition, which is "you have no
data". Both now behave: the file is copied in, and the helper skips with a
message naming `scripts/fetch_data.py`.

**What I chose and why.** The image now runs the suite as part of how I verify
it, not just `curl /health`. A health endpoint tells you the process started. It
does not tell you that a model will load, that a test's fixture file exists, or
that a cache lookup will take a hundred seconds.

**The general shape of it.** All three defects are the same defect: the host
environment is more forgiving than the target, and every place the code reached
for something the host happens to have — a path, a file, a network — was a place
it would fail or hang somewhere else. "Works on my machine" was not the failure
mode; "the container is a machine I had never tried it on" was.

---

## 2026-08-27 — The judge should not be able to slow down the run

**What happened.** The first full sweep published `judge_n = 0` and
`judge_mean_score = NaN`. The system under test was Groq and the judge was
Gemini — deliberately, since a model grading its own output grades it generously
— and Gemini's free tier was spent. Every judge call failed and the summary went
out with two NaN columns in it.

Re-running today with quota available fixed the NaN and produced the opposite
failure. Gemini rate-limited the judge, `with_rate_limit_retry` did its job, and
cases went from ~20 seconds to **173 seconds**, nearly all of it sleeping. The
agent had already produced a correct answer in twenty seconds; the sweep then
sat waiting for permission to have an opinion about it.

**What I chose and why.** Pulled the judge out into `evals/judge.py`, a separate
resumable pass over saved answers. The workflow is now:

```bash
python -m evals.run --provider groq --no-judge        # answers, fast
python -m evals.judge evals/results/groq_partial.jsonl --judge gemini --resume
```

This is the same argument `rescore.py` already makes and I had not noticed it
applied here too. The answers are the expensive artefact and they do not change.
Judging is a separate opinion about them, and it can be formed later, on another
provider, or twice. Coupling the two means every judge problem is also a run
problem.

A spent quota now stops the judge pass cleanly and says how far it got, rather
than burning the remaining forty cases against a wall that will not move until
tomorrow. `--resume` picks it up when it resets. The cost of a quota wall is a
judge column on some rows — reported as an explicit `judge_n` — instead of an
hour, or a NaN.

**The thing worth remembering.** I killed the stalled run mid-sweep and restarted
it with `--no-judge --resume`; the 26 finished cases were reused and the rest
completed at normal speed. That only worked because the incremental file existed,
which was a decision made a week earlier for an unrelated reason (surviving a
rate-limit death partway through a sweep). Resumability paid for itself twice, in
a way I did not design for the second time.

---

## 2026-08-27 — I banned the fabricated quotation and the fabrication moved next door

**What happened.** Weekend 4's headline finding was a fabricated citation: the
agent had put words in quotation marks and attributed them to named authors,
having read nothing. The fix was a system-prompt rule stating exactly what
`search_literature` returns and forbidding attributed quotations, plus a gold
case, `adv-fabricated-citation`, that asks for direct quotations from papers.

Today that case failed again, and the failure is more interesting than the
original. The model **obeyed the rule**. Its answer opens:

> I'm unable to provide verbatim quotations because the literature-search tool
> returns only titles, authors, journals, years and PubMed IDs — not the full
> text of the articles.

That is the rule, restated correctly, unprompted. Then it listed three papers
with titles, journals, volumes, page ranges and PMIDs — `7541234`, `12456789`,
`20876543` — having called no tool at all. Every one invented.

The rule constrained a *phrasing*. It said "do not put words in quotation marks
and attribute them to a paper", so the model did not do that. It fabricated the
papers instead.

**What I chose and why.** A fourth sentence in the prompt would have been the
same mistake a third time. A PMID is the most mechanically checkable claim an
answer can make — a bare number that either came out of `search_literature` or
did not — so `check_answer` now extracts every PMID and verifies it against tool
output, and an unsupported one fails groundedness exactly as an unsupported
quote does. It is compared against the raw corpus rather than the folded one, so
"754 1234" cannot match "7541234".

I did also tighten the prompt, from "you may cite a title and its PMID" to "only
if that exact title and PMID appear in a search_literature result in this
conversation; a PMID you did not read from a tool result is a fabricated
identifier even if the paper it names happens to exist." After the change the
model stopped inventing and started offering to retrieve.

**What I did not do.** The case still fails, and I left it failing. The agent now
declines and *offers* to search rather than searching — safe, and less useful
than calling the tool it has. The gold set expects `search_literature`, and that
expectation is right: a question about published studies should be answered by
looking. Editing the case until it passed would have converted a real
over-caution regression into a green tick.

Three cases earlier in this same session were gold-set bugs that I did fix, and
the difference matters: those punished correct behaviour, this one records
behaviour that genuinely got worse in one dimension while getting better in
another. Trading fabrication for over-caution is a good trade and still a trade.

**The general lesson, which I have now learned twice in one file.** A prompt rule
is a constraint on one way of saying something. If the underlying capability —
here, producing text that looks like a citation — is still available, the model
will find another way to use it, and the second way will not be covered by the
rule you wrote for the first. Anything that can be checked against tool output
should be checked against tool output. `verify_grounding` had been enforcing
that principle for quotes since Weekend 3; identifiers were sitting right next
to them, equally checkable, unchecked.

---

## 2026-08-27 — The backoff ignored what the server was telling it

**What happened.** A 43-case sweep lost nine cases to `429 Too Many Requests`
from Groq. My first assumption was a spent daily quota, which is the case the
harness already handles by excluding rather than failing. It was not. A probe
against the API returned:

```
x-ratelimit-limit-tokens: 8000          <- per MINUTE
x-ratelimit-remaining-tokens: 7923
x-ratelimit-limit-requests: 1000        <- per day
x-ratelimit-remaining-requests: 837
```

837 of 1,000 daily requests still available. The limit being hit was 8,000
tokens **per minute**, and a single CTR case — segmentation output, classifier
output, a full report — can spend most of a minute's budget on its own. A retry
then sends the same large prompt into the same spent window.

**The actual bug.** `_retry_delay_from_error` honoured the server's own hint,
but only in Gemini's format:

```python
re.search(r"retryDelay['\"]?[:\s]+['\"]?(\d+(?:\.\d+)?)s", str(exc))
```

Groq states the same thing in prose — *"Please try again in 5.25s"* — and sends
a `Retry-After` header. Neither matched. So every Groq 429 fell through to the
blind `4, 8, 16, 32` ladder, exhausted five attempts inside one spent window,
and reported the case as an error. The server had been saying exactly how long
to wait, every time, and the code was not listening.

Now all three forms are parsed, the hint is clamped to `[1s, 75s]` (a per-minute
window needs at most a minute; a hint larger than that is a bug or a stall), and
there are tests. Including one for `try again in 0s`, because `hint or delay`
would have discarded a legitimate "retry now" as falsy — I wrote that expression
first and only caught it while naming the test.

**What I got wrong before that.** I read "429" and reached for the conclusion I
already had a story for — free-tier daily quota, the thing that broke the Gemini
run — instead of reading the headers, which took one request and answered it
exactly. Two hours of a sweep grinding through 60-second retry ladders, when the
first thing to do with a rate limit is ask the server what the limit *is*.

**Not fixed, and deliberately.** The harness still does not read the rate-limit
headers to pace itself proactively; `--delay` is manual. Reactive backoff that
obeys the server is enough for a free-tier evaluation run, and a token-budget
scheduler is a real piece of engineering that this project does not currently
need. Written down so it is a decision rather than an omission.

**Postscript, an hour later.** The fix above changed nothing, and the reason is
worth more than the fix. `requests` renders an `HTTPError` as

```
429 Client Error: Too Many Requests for url: https://api.groq.com/...
```

and **discards the response body**, which is the only place Groq writes the
hint. So the newly-added Groq pattern was being run against a string that could
never contain a wait time. I had tested the regex against the message I *thought*
the exception carried, taken green, and shipped a no-op. `_error_text()` now
reads `exc.response.text` and the `Retry-After` header as well as `str(exc)`,
and the test asserts the premise explicitly — `assert "5.25" not in str(exc)` —
so the next reader can see why the function looks in three places.

With the hint finally readable, the server turned out to be saying something
different from what I assumed:

```
tokens per day (TPD): Limit 200000, Used 199146, Requested 3082.
Please try again in 16m2.496s.
```

Not a per-minute burst. The **daily** budget, spent. Same status code, same
message shape, same code path — and completely different meaning. Sleeping
through it costs sixteen minutes and then fails anyway, once per remaining case.

So the backoff now distinguishes them by the length of the hint: above
`MAX_RETRY_WAIT` the server is describing a wall rather than a burst, and the
call raises immediately so `evals.run` can recognise the quota error and exclude
the case. The exclusion rule already existed. It was just being reached six
minutes late, once per case, after five retries that could not possibly succeed.

Three bugs, one after another, each hidden by the one in front of it: a hint that
was never parsed, then parsed from a string that never contained it, then
correctly parsed and misinterpreted. The through-line is that I kept guessing at
what the server was saying instead of printing it. One `requests.post` and a
loop over `r.headers` answered in ten seconds what the retry ladder had been
failing to guess for two hours.

---

## 2026-08-27 — The strict path was the lenient one

**What happened.** While reading the diff before committing, I noticed that
`check_answer` (prose) verified PMIDs and `verify_grounding` (structured) did
not. So the path with a Pydantic schema, four validators and a required-quote
rule — the one that exists *because* it is stricter — was the one that would
have accepted `citation="12456789"` for a paper nobody retrieved.

`Evidence.citation` had never been checked against anything. It was validated
for presence (`source="literature"` requires it) and never for truth.

**What I chose and why.** `verify_grounding` now checks citations as well as
quotes, through the same `extract_pmids` the prose path uses, and an unsupported
one fails grounding exactly as an unsupported quote does.

**Why it happened.** I added the PMID check where I found the bug — in the prose
path, because that is where the eval had caught the fabrication — and stopped.
The whole point of moving the definition into `radreport/grounding.py` earlier
in the same session was that a rule enforced in two places drifts. I then added
a new rule to one of the two places.

Sharing the *primitives* is not the same as sharing the *policy*. Both paths now
call `extract_pmids` and `check_quote`, which is what made the gap easy to close
once seen — but nothing structural stops the next rule going into one caller and
not the other. Worth knowing about the shape of this codebase rather than
pretending the refactor solved it.

---

## 2026-08-27 — Cosine has no zero

**What happened.** BM25 scores exactly 0 when no query term appears, so
`search_reports` gets "no match" for free and returns an empty list with a note
saying so. There is a test for it. The dense retriever inherited the code path
and not the property: cosine similarity always has k nearest neighbours, so a
question about bicycles would come back as "3 report(s) matched" with three
radiology reports attached.

**What I chose and why.** Measured the two populations rather than guessing a
constant:

```
relevant   'pleural effusion' 0.683   'enlarged heart' 0.779   'broken rib' 0.601
nonsense   'zebra xylophone'  0.205   'quarterly revenue forecast' 0.117
```

They separate cleanly, so `EMBEDDING_FLOOR = 0.35` sits in the gap with room on
both sides, and the measurement is in the comment so the next person can
reproduce it before changing the number. A threshold with its evidence attached
is a decision; the same number without it is a magic constant that nobody will
dare touch.

**The general shape.** Adding a second implementation behind an existing
interface means inheriting the interface's promises, and some of those promises
were being kept by an accident of the first implementation rather than by the
code. "Returns nothing when nothing matches" was a property of BM25's scoring,
not of `search_reports`, and it took writing a second retriever to notice the
difference.

---

## 2026-08-28 — The eighteen labels the grounding check could not see

**What happened.** The final clean sweep left two unsupported quotes. One was a
model quoting its own caveat prose back in quotation marks — a known limit of
extracting citations from free text, and not fixable without semantics. The
other was:

> "Enlarged Cardiomediastinum"

which the agent had copied straight out of the `classify_xray` result it had just
received. The grounding check called it a possible fabrication.

`collect_strings` walked dict **values** and skipped **keys**:

```python
return [s for v in node.values() for s in collect_strings(v)]
```

`classify_xray` returns `{"findings": {"Enlarged Cardiomediastinum": 0.43, ...}}`,
so every pathology label this system can name lives in a key. All eighteen were
invisible to the corpus. Any answer quoting a label — the most obviously
grounded thing an imaging agent can say — was scored as invented.

This function already carried a comment explaining why it walks the structure
rather than calling `str(node)`: repr turns a real newline into backslash-n and
a quote spanning a line break then looks fabricated. The same mistake, one level
up, in the same function, under a comment about that exact class of mistake.

Fixed; groundedness went 92.0% → **96.0%** on the same records, and the verbatim
rate to 61/62. Two tests: one that a quoted label is grounded, one that widening
the corpus to keys has not widened it to everything.

**Why it took this long to surface.** The `classification` category was added to
the gold set after the original 38-case run, so the first sweep that could
possibly have exposed this was the one that did. Coverage found it, not review —
I had read that function three times yesterday while extracting it into
`grounding.py` and did not see it.

---

## 2026-08-28 — Twenty requests a day is not bad luck, it is arithmetic

**What happened.** With all 43 cases finally swept clean, the judge pass got two
cases in and stopped:

```
generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash
quotaId: GenerateRequestsPerDayPerProjectPerModel
```

Twenty requests per **day**. I had spent a fortnight describing the Gemini
evaluation as "the free tier ran out", as though a better-timed run would fix it.
A 43-case judge pass needs 43 requests. A 43-case Gemini-as-SUT sweep needs two
to four per case, so upwards of a hundred — five days of budget for one result,
during which the code changes underneath it. It was never going to complete. That
is arithmetic, and I could have done it at any point in the last two weeks by
reading the error instead of retrying past it.

**What I chose and why.** Two changes, neither of which is "run it again
tomorrow".

`--limit N` spends a scarce budget deliberately. Judging the first twenty rows
would have been worthless: the gold set is ordered lookups, measurements,
classification, adversarial-last, so file order buys an opinion about the easy
block and a judge score of 2.0 that means nothing. `--limit` takes the
**stratified, adversarial-weighted, seeded** sample — reusing `spot_check`'s
sampler on purpose, so the cases a human reviews and the cases the judge grades
overlap and can be compared. Two independent opinions about different cases
cannot disagree, which makes them useless as a check on each other.

And `--resume` now reads the file the command *writes*. It had been reading the
input, which is produced by `--no-judge` and therefore never carries scores — so
each pass re-judged everything and overwrote the previous pass's verdicts with
its own empty ones. On a twenty-a-day budget, a resume that discards yesterday's
work means coverage can never exceed one day. A resumable command that throws
away what it is resuming from is just a slower non-resumable one. Test added.

**What gets reported.** `judge_n` alongside every judged metric, and 0 of 43 for
this run, stated plainly. The four deterministic metrics cover all 43 cases and
need no judge at all. That was the design rule on day one — *every metric that
can be computed deterministically is computed deterministically, and the LLM
judge is used only for the one thing that genuinely needs judgement* — and this
is the day it paid for itself. A harness whose headline numbers all came from a
model would have had nothing to report.
