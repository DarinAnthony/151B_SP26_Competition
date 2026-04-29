# Parameter Sampling

**Goal:** Find decoding settings that maximize accuracy on the eval slice — and exploit multi-sample strategies (self-consistency / majority voting) when the budget allows.

**Levers:**
- `temperature`, `top_p`, `top_k`, `repetition_penalty`.
- Greedy vs. sampled decoding for MCQ (often greedy wins) vs. free-form (sampling helps).
- N-sample majority voting on MCQ; N-sample answer-aggregation (mode of `\boxed{}` extractions) on free-form.
- `MAX_TOKENS` cap — pin the right value once and use it everywhere.

**Why:** Same model weights, same prompts — just better-tuned decoding. Pairs naturally with prompt-engineering wins.

**Success looks like:** A pinned `(temperature, top_p, top_k, MAX_TOKENS, n_samples)` config that beats the starter defaults on the eval slice. Document the cost-vs-accuracy curve for `n_samples ∈ {1, 4, 8}` so future experiments know what to budget.
