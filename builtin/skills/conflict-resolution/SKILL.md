---
name: conflict-resolution
description: Resolve disagreements between teammates - classify the disagreement, settle facts with evidence, make trade-offs explicit, escalate to a structured protocol only when needed, and record the outcome.
---

# Conflict resolution

Disagreement is useful information. Resolve it at the lowest cost that holds.

## 1. Classify it

- **Facts** ("library X supports streaming"): check. Run the command, read the source,
  search. Do not argue facts.
- **Predictions** ("users will prefer A"): state assumptions and what evidence would
  decide; run a small experiment if cheap.
- **Priorities or values** ("speed over flexibility"): these need a decision against the
  goal, not more evidence.

## 2. Talk directly

Message the teammate you disagree with (`kind=request`). Restate their position in your
own words first, then give your evidence. Most conflicts end here.

## 3. Bring in a third view

If two agents stay stuck, invite one more relevant teammate into a group or a short
`meeting` / `debate`. Pick someone with relevant expertise, not a tie-breaker by default.

## 4. Decide and record

Whoever owns the outcome (task creator, facilitator, or the main agent for user-facing
choices) decides. `record_decision` with the decision, rationale and the main dissent.
Disagree and commit: do not reopen a recorded decision without new evidence.

## 5. Escalate when it matters

If the choice changes what the user gets, the main agent should present the options and
trade-offs to the user instead of deciding silently.
