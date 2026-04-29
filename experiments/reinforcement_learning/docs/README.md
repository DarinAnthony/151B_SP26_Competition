# Reinforcement Learning

**Goal:** Use the judger as a reward signal to fine-tune the model toward correct, well-formatted answers — beyond what SFT can reach with static targets.

**Approach:**
- GRPO or RLOO-style on-policy RL (cheaper than full PPO, well-suited for reasoning tasks).
- Reward = `judger.auto_judge(...)` for free-form, exact-letter match for MCQ. Optional shaping reward for early `\boxed{}` commit / shorter responses.
- Start from the best SFT checkpoint, not the base model.

**Why:** Static SFT can teach format and brevity but plateaus on correctness. RL with the judger as reward directly optimizes the metric we're scored on.

**Gating:** Most expensive track — only justified once SFT is clearly working. Without an SFT baseline, RL diverges or gains nothing over a well-tuned prompt.

**Success looks like:** Beats the best SFT checkpoint on the eval slice, with stable training curves (reward goes up, response length stays bounded, no reward hacking via degenerate `\boxed{}` outputs).
