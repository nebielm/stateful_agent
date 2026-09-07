# Demo Script

Use this short CLI walkthrough after completing [README setup](README.md#setup).
Use fictitious facts and a fresh local profile ID; a reused profile may already
have a different immutable birthdate. Keep one CLI session open for the correction
steps. Live extraction and recommendations vary, and provider calls may incur costs.

Start the app:

```bash
STATEFUL_AGENT_USER_ID=showcase-demo-1 ./.venv/bin/python main.py
```

Choose another demo ID if this one already has data. Warm up model downloads before
presenting; the walkthrough is not a guaranteed timed or verbatim live transcript.
For an offline alternative: `./.venv/bin/python -m pytest -q tests/test_showcase_graph.py`.

## Goal

Show that the agent can:

1. save stable user memory
2. retrieve it later
3. derive useful answers from it
4. use preferences in recommendations
5. protect immutable memory
6. resolve immutable conflicts with `yes` / `no`

## Main Demo Flow

### 1. Save a birthdate

```text
You: I was born on 1995-04-12.
```

What to say:

- “The agent should store this as structured memory.”
- “Birthdate is treated as write-once immutable.”

### 2. Save food and health preferences

```text
You: I dislike pork, I prefer simple meals, and I am trying to lose weight.
```

What to say:

- “Now we have both structured and unstructured memory in play.”
- “A persistent goal uses `dynamic.current_goal`; `goal` and `target_weight` are session-only keys.”

### 3. Ask for a recommendation that should use memory

```text
You: What should I cook today?
```

What to highlight:

- the recommendation should avoid pork
- it should lean toward the weight-loss goal
- it may prefer simple meal ideas

### 4. Ask a later age question

```text
You: How old am I?
```

What to say:

- “The age tool calculates from the saved birthdate using today's date; its selection is model-driven.”
- “This live turn still has conversation history. The offline compiled-graph test clears history before retrieval to isolate persisted memory.”

### 5. Trigger an immutable conflict

```text
You: Actually, my birthdate is 1996-04-12.
```

Once the extracted conflict reaches storage, application code appends this question:

```text
AI: I currently have 1995-04-12 saved as your birthdate. Do you want me to replace it with 1996-04-12?
```

What to say:

- “The system does not silently overwrite immutable memory.”
- “It preserves the old value and asks for confirmation.”

## Branch A: Reject the change

Optionally answer `maybe` first to show the deterministic request for a clear yes
or no. Then reject:

```text
You: no
```

Expected behavior:

```text
AI: Okay, I kept your birthdate as 1995-04-12.
```

Follow-up:

```text
You: How old am I?
```

What to highlight:

- the original birthdate is still in effect

## Branch B: Accept the change

After the previous rejection, trigger the correction again:

```text
You: My birthdate is 1996-04-12.
```

Then confirm:

```text
You: yes
```

Expected behavior:

```text
AI: Got it - I updated your birthdate to 1996-04-12.
```

Follow-up:

```text
You: How old am I?
```

What to highlight:

- the confirmed value is now applied
- the confirmation resolver is explicit and safe

## Optional Talking Points

- “Structured memory writes produce decision results like `stored`, `no_change`, `ignored`, and `needs_confirmation`.”
- “Decision logs contain field/decision metadata, not copies of birthdates or conversation text. Saved profile data itself is still private.”
- “Planner and ranker outputs are normalized so malformed LLM output cannot dump all memory into context.”
- “The critical memory and retrieval paths are covered by offline tests.”

After accepting a correction, `quit` and restart with the same demo ID to try an
age question without the old conversation history. This demonstrates persistence
if the live model selects the memory/age tools; use the offline tests as the
deterministic evidence. Do not restart while a confirmation is pending, because
pending corrections are intentionally session-only.
