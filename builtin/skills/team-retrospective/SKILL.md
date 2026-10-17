---
name: team-retrospective
description: After significant multi-agent work, review how the team collaborated - what worked, what wasted effort, and concrete changes - and record lessons so later sessions start smarter. Skip for small tasks.
---

# Team retrospective

Run only after substantial work (several tasks, a protocol, or a failure worth learning
from). A retrospective for a two-message exchange is waste.

## Gather facts first

Use `get_state` to see tasks, protocols, decisions and message counts. Note:

- Tasks that failed, were blocked, or were redone.
- Messages that did not move work forward (reminders, relays, repeated questions).
- Protocols that ran more rounds than needed, or decisions that were reopened.
- Budget or hop limits that were hit.

## Write the retrospective

Share it with `share_artifact` titled "Retrospective: <topic>":

1. **Outcome**: what was delivered and how it was verified.
2. **Worked**: collaboration choices to repeat, with the reason.
3. **Wasted**: where effort or tokens went without value.
4. **Change**: at most three concrete changes (e.g. "assign API research to researcher
   directly instead of via main", "use independent_research before debating vendors").

Record durable changes with `record_decision` so they appear on the board in later
sessions of this environment.
