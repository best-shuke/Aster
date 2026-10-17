---
name: independent-debate
description: Run or take part in a structured debate when teammates disagree on a consequential choice - form independent positions, argue from evidence, steelman the other side, and converge on a recorded decision.
---

# Independent debate

Use a debate when a consequential choice has at least two defensible options and
informal messages have not converged. Do not debate matters of fact you can check.

## Starting one

`start_protocol protocol=debate topic=... participants=[...] rounds=2..3`

- Invite only agents with relevant expertise or a stake; two or three is usually enough.
- State the decision to be made and the criteria in `goal`.
- Optionally pass `stances` to assign positions so the space is covered.

## Participating

Each round you receive the transcript so far. Contribute with
`contribute protocol_id=...` (or your final message):

1. Round 1: state your position, the strongest evidence for it, and what would change
   your mind. Form it independently; do not wait for others.
2. Later rounds: steelman the strongest opposing argument before rebutting it. Concede
   points that are right. Introduce new evidence, not repetition.
3. Set `ready=true` when you think the decision can be made. When everyone is ready the
   debate ends early.

## Concluding

The facilitator receives the transcript and concludes with `conclude` giving the
decision, the rationale, the strongest unresolved objection, and follow-up tasks. The
decisions are recorded on the team board.
