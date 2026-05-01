# Prompt Engineering

**Goal:** Improve accuracy without touching weights — only the prompt, the chat template, and the output format.

**Levers:**
- System-prompt rewrites (stricter format, role framing, terse-vs-verbose).
- Few-shot exemplars inside the user turn for both MCQ and free-form.
- Forcing an early `\boxed{}` commit so the thinking model doesn't run out of tokens before answering.
- Different output schemas (e.g. `Final answer:` markers) and post-hoc extraction.

**Why first:** Cheapest lever — no GPU training, fast iteration. Establishes the upper bound on prompt-only gains so later experiments (SFT, RL) can be compared against a fair prompt baseline.

**Success looks like:** A prompt template that lifts overall accuracy on the 100-question eval slice over the starter prompt, with the gain split between MCQ and free-form. Bonus: lower average response length than the starter (fewer cut-off `\boxed{}`-less responses).
