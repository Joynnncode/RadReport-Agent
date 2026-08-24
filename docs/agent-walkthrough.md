# `agent.py`, line by line

Read this with `radreport/agent.py` open beside it. The goal is that you can
draw the loop on a whiteboard and answer any question about why it is shaped
this way.

---

## The one-sentence version

> An agent is a `while` loop around a chat API. The model can only emit text;
> when it emits a *structured* request instead of prose, our code runs the tool
> and appends the result to the conversation. Everything the agent can do is
> something we explicitly gave it.

If you say only that in an interview, you are already ahead.

---

## Part 1: `TOOLS` — the model's only documentation

The model has never seen our source. It knows what the schema says and nothing
else. Each description carries four things beyond the function name:

| | Example from `compute_ctr` |
|---|---|
| **when to use / not use** | "REQUIRES a mask_handle produced by segment_lungs" |
| **where the argument comes from** | "The 'mask_handle' value from a segment_lungs result" |
| **what the output means** | "above 0.50 on an erect PA film may be consistent with…" |
| **the boundary** | "must never be reported as a diagnosis" |

The pair that needs the most care is `search_reports` vs `get_report_by_image`.
Confusing them means reporting **another patient's** report as this patient's.
So both descriptions carry an explicit *negative*: "Never use it to answer a
question about a specific named case." There's a test asserting those sentences
are still there — a test on a prompt, which feels odd until you realise the
description is load-bearing safety logic.

> **Interview line:** "When an agent misuses a tool, I suspect my schema before
> I suspect the model. Nine times out of ten I under-described something."

---

## Part 2: `SYSTEM_PROMPT` — deliberately short

Everything enforceable in Python is enforced in Python. The CTR plausibility
range, the exact-lookup/fuzzy-search split, the diagnostic-language ban — all
code. What's left in the prompt is only what can't be expressed as a check: the
evidence rule, the quoting rule, the scope boundary.

> **Interview line:** "A long system prompt usually means a rule belongs in a
> tool. A prompt is a request the model can decline; a validator is a guarantee."

---

## Part 3: `dispatch()` — the function that must never raise

Four cases, most specific first:

```
(a) unknown tool     -> "No tool named 'diagnose'. Available: classify_xray, …"
(b) ToolError        -> exc.as_tool_result()   (message already written for the model)
(c) TypeError        -> echo the real signature so it can correct itself
(d) anything else    -> generic failure, recoverable=False, NO traceback
```

Why never raise: if `dispatch` raised, one bad tool call would kill the run.
Because it doesn't, a bad call becomes a message the model reads and recovers
from. That recovery is tested end to end (`test_agent_recovers_from_a_tool_error`).

Why no traceback in (d): a traceback costs hundreds of tokens and tells the
model nothing it can act on. Compare:

- ✗ `FileNotFoundError: [Errno 2] No such file or directory: 'masks.npz'`
- ✓ `Mask file masks.npz not found. Call segment_lungs first.`

Both true. Only one lets the model fix it. **Write error messages for the model,
because it is the one reading them.**

`resolve_image_path()` sits here rather than in the tools on purpose: the tools
take real paths and stay testable as such, while the agent stays forgiving of
the bare case id a user would actually type.

---

## Part 4: `Trace` — not logging

Logging is for a human when something breaks. A trace is **structured data the
evaluation reads**. Weekend 4 computes every metric from these files, so the
schema is an interface.

| Metric | Field it reads |
|---|---|
| tool selection accuracy | the ordered `tool_call` events (`tracer.tool_sequence`) |
| groundedness | `result` on each `tool_call` |
| cost per query | `input_tokens` / `output_tokens` on `llm_call` |
| latency | `latency_s` |
| failure rate | `converged` on `final` |

JSONL not JSON: we append as we go, so a run that crashes halfway still leaves a
readable file — exactly when you most want it.

The full tool `result` goes in the trace so the file is self-contained and the
eval can score groundedness offline. This was a correction: the first version
re-ran the tools, which verified the answer against a *fresh* run rather than
against what the model actually saw. When those differ, the check is worthless.

---

## Part 5: `run()` — the loop

```
messages = [user]
for iteration in range(max_iterations):
    response = provider.chat(messages, TOOLS, SYSTEM_PROMPT)
    trace llm_call
    if not response.wants_tools:          # <- the exit
        trace final(converged=True); return
    messages.append(assistant turn WITH tool_calls)      # <- both
    for call in response.tool_calls:
        result = dispatch(call.name, call.arguments)
        trace tool_call
        messages.append(tool result)                     # <- of these
else:
    trace final(converged=False)          # <- a FAILURE, not an answer
```

**Three things to be able to defend:**

1. **Why append the assistant's tool-call turn as well as the result?** If you
   append only the result, the model sees an answer to a question it has no
   record of asking. It gets confused and calls the same tool again, forever,
   until `max_iterations` fires. This is the single most common bug when writing
   a loop from scratch. `test_model_sees_both_its_request_and_the_result` pins
   the exact message sequence `["user", "assistant", "tool"]`.

2. **What stops it looping forever?** `max_iterations`, default 8. It guards
   three real failure modes: ping-pong (call, dislike result, call again);
   quota burn (every iteration is an API call); and cost blowup (the
   conversation grows each turn, so iteration 20 costs far more than iteration
   2 — it's superlinear). Eight is chosen because the longest legitimate chain
   here is segment → ctr → lookup → literature → answer, which is five.

3. **Why is hitting the cap a failure rather than a result?** Because if we
   quietly returned the last text lying around, the eval would score garbage as
   if it were an answer. `converged=False` lets Weekend 4 count non-convergence
   honestly. The CLI exits 4 so a shell script can detect it without parsing JSON.

---

## Part 6: `run_structured()` — prose in, validated JSON out

Three phases: run the normal loop → ask for the answer as JSON against the
schema → validate, and on failure hand the **validation error** back and retry
once.

Phase 3 is the interesting one. Pydantic's error names the field and the rule
that failed, which is precisely the feedback a model needs to correct itself.
Retrying with the same prompt would just reroll the dice; retrying with the
error is a repair. Budget is one attempt, then report the failure.

Kept separate from `run()` because the loop and the output contract are separate
concerns: "did it pick the right tools" is a different question from "did it
format and ground the answer", and the eval should measure them independently.

`_extract_json` strips markdown fences because models emit ```json blocks
roughly half the time however clearly you ask them not to. Cheaper than a retry.

---

## Answer these out loud before your interview

1. Draw the loop. Where does it exit, and where can it fail to exit?
2. Why append the model's tool-call turn as well as the result?
3. A tool throws. Trace the exception's path to the model and back.
4. Why is `compute_ctr` a separate tool from `segment_lungs`?
5. Why is the CTR plausibility check in Python rather than in the prompt?
6. What's in your trace, and which metric consumes each field?
7. Why does `FakeProvider` exist, and what can you test with it that you
   couldn't test against Gemini?
8. What is the most dangerous failure mode of this system? *(Answer: fabricating
   a report quote, or returning another patient's report as this patient's. Then
   describe the three defences: the tool split, the verbatim-quote requirement,
   and `verify_grounding`.)*

Question 8 is the one that will impress a clinical AI company. Prepare it properly.
