# Weekend 2: understanding the agent loop before you write it

Read this, close it, then write `radreport/agent.py`. No copying.

---

## 1. What an "agent" actually is

Strip away the marketing and an agent is a `while` loop around a chat API.

The model cannot run code. It cannot open your files. All it can do is emit
text. Tool calling is a convention where, instead of prose, the model emits a
structured request — *"call `compute_ctr` with `mask_handle='...'`"* — and your
code decides whether and how to honour it.

**You** run the tool. **You** decide what goes back into the conversation. The
model never touches your filesystem. Everything an agent can do is something you
explicitly gave it.

That is the whole idea. LangChain is a wrapper around this loop. You are writing
the loop, so you will understand what the wrapper is hiding.

---

## 2. The conversation is a list you keep appending to

The API is stateless. Every call sends the whole conversation. "Memory" is just
a Python list you own.

After a full run the list looks like:

```
[ system  ] you are a clinical imaging assistant, always cite evidence...
[ user    ] Does 702_IM-2267-1001 show cardiomegaly?
[ model   ] (no text) tool_calls: [segment_lungs(image_path=...)]
[ tool    ] {"ok": true, "mask_handle": "artifacts/..._masks.npz", ...}
[ model   ] (no text) tool_calls: [compute_ctr(mask_handle="artifacts/...")]
[ tool    ] {"ok": true, "ctr": 0.528, "interpretation": "..."}
[ model   ] "CTR is 0.53, above the 0.50 threshold. On a frontal film..."
```

Each `[model]` turn is one API call. The loop above made three.

**The mistake almost everyone makes once:** appending the tool *result* but not
the model's *tool-call turn*. Then the model sees a result to a question it has
no record of asking, gets confused, and calls the same tool again. And again.
Your `max_iterations` guard fires and you spend an hour confused. Append both.

---

## 3. Tool schemas are the model's only documentation

The model has never seen your source. It knows exactly what your schema says.

```json
{
  "name": "compute_ctr",
  "description": "Compute the cardiothoracic ratio from segmentation masks. Requires a mask_handle produced by segment_lungs; call that first. Returns a ratio where >0.50 on a frontal film may suggest cardiomegaly. Does not diagnose.",
  "parameters": {
    "type": "object",
    "properties": {
      "mask_handle": {
        "type": "string",
        "description": "Path returned in the 'mask_handle' field of a segment_lungs result."
      }
    },
    "required": ["mask_handle"]
  }
}
```

Notice what the description does beyond naming the function: it states a
**precondition** (call segment_lungs first), tells the model **where the
argument comes from**, gives the output's **meaning and threshold**, and marks a
**boundary** (does not diagnose).

> When an agent misuses a tool, suspect your schema before you suspect the
> model. Ninety percent of the time you under-described something.

Two schemas that need special care in this project:

- `search_reports` vs `get_report_by_image`. If these are described sloppily the
  model will use fuzzy search for a specific patient and confidently report
  someone else's findings. Say plainly in each description when *not* to use it.
- `classify_xray`. Its output is 18 calibrated probabilities, not a diagnosis.
  If the description says "detects pathologies" the model will report detections.
  Say "returns per-pathology probabilities from a screening model".

---

## 4. Why `max_iterations` exists

Three real failure modes it protects against:

1. **Ping-pong.** The model calls a tool, dislikes the result, calls it again
   with near-identical arguments, forever.
2. **Quota burn.** Every iteration is an API call. An unbounded loop on a free
   tier means no more agent until tomorrow.
3. **Cost blowup.** The conversation grows every iteration, so iteration 20
   costs far more than iteration 2. It is superlinear.

Hitting the cap is a **failure**, not a result. Do not silently return the last
text you happened to have. Return something that says the run did not converge,
so your Weekend 4 eval can count these as failures rather than scoring garbage.

Eight is a reasonable default here: the longest legitimate chain in this project
is roughly segment → ctr → lookup report → literature → answer, which is five.

---

## 5. Errors must become results, not crashes

A tool raising `ToolError` is normal operation, not an emergency.

```
model calls compute_ctr(mask_handle="masks.npz")   # a path it invented
  -> ToolError("Mask file masks.npz not found. Call segment_lungs first.")
  -> becomes a tool result: {"ok": false, "error": "Mask file ... Call segment_lungs first."}
  -> model reads it, calls segment_lungs, then retries. Recovered.
```

That recovery only works if the error message says what to do next. Compare:

- ✗ `FileNotFoundError: [Errno 2] No such file or directory: 'masks.npz'`
- ✓ `Mask file masks.npz not found. Call segment_lungs first.`

Both are true. Only one lets the model fix it. **Write error messages for the
model, because it is the one reading them.** This is why the tools already carry
a `recoverable` flag: it tells the model whether retrying is even worth a turn.

Your `dispatch()` must never raise. If it does, one bad tool call kills the run.

---

## 6. The trace is not logging

Logging is for you when something breaks. A trace is **structured data your
evaluation reads**. Weekend 4 computes tool-selection accuracy, cost and latency
by parsing this file, so its schema is an interface, not a debugging convenience.

One JSON object per line, appended as you go:

```jsonl
{"ts":"...","run_id":"a3f","iteration":0,"event":"llm_call","model":"gemini-2.0-flash","input_tokens":812,"output_tokens":24,"latency_s":0.71}
{"ts":"...","run_id":"a3f","iteration":0,"event":"tool_call","tool":"segment_lungs","arguments":{"image_path":"..."},"latency_s":3.9,"ok":true}
{"ts":"...","run_id":"a3f","iteration":2,"event":"final","answer":"...","total_iterations":3}
```

JSONL, not JSON: you append line by line, and a run that crashes halfway still
leaves a readable file. That is exactly when you most want the trace.

The sequence of `tool_call` events *is* the tool-selection metric. Get the
schema right now and Weekend 4 is a `for` loop over files instead of a rewrite.

---

## 7. Build order

Do not write all 150 lines then run it. Six steps, run it after each:

1. Loop with **zero** tools. Confirm you get plain text back. This proves your
   provider wiring and message format work.
2. Add **one** schema (`search_reports` — no models, fast, no image path to get
   wrong). Confirm the model requests it.
3. Add `dispatch()` for that one tool. Confirm you get a real answer.
4. Add the trace. Read the file. Is it what Weekend 4 needs?
5. Add the remaining five schemas, one at a time, testing after each.
6. Break it on purpose: pass a bad image path, ask about a patient id that does
   not exist, ask "what drug should I prescribe". Watch what it does. Write down
   what happened in `DECISIONS.md`. Those notes are your interview answers to
   "what is the most dangerous failure mode of this system".

---

## 8. Questions to answer out loud when you finish

If you can answer these without notes, the weekend worked.

1. Draw the loop. Where does it exit, and where can it fail to exit?
2. Why append the model's tool-call turn as well as the result?
3. A tool throws. Trace the exception's path to the model and back.
4. Why is `compute_ctr` a separate tool instead of part of `segment_lungs`?
5. Why is the CTR plausibility check in Python rather than in the prompt?
6. What is in your trace, and which Weekend 4 metric consumes each field?
