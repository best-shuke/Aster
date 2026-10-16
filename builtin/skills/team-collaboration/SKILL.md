---
name: team-collaboration
description: Choose how (and whether) to collaborate in a multi-agent environment - when to work alone, ask a teammate, form a group, run a structured protocol, and how to keep collaboration cheap. Load before engaging teammates on non-trivial work.
---

# Team collaboration

Collaboration exists to solve the problem, not to maximize agents, messages or meetings.
Every message wakes someone and costs tokens. Pick the cheapest mode that works.

## Pick a mode

| Situation | Mode |
| --- | --- |
| You can finish it yourself with your tools | Work alone. Report the result. |
| You are blocked on a fact, permission or skill a teammate has | `send_message` with `kind=request` to that one teammate |
| Several independent pieces of work | `create_task` per piece, with `assignee` and `depends_on` |
| You need unbiased views before anyone anchors on an answer | `start_protocol` `independent_research` (or `brainstorm`) |
| A real disagreement that messages have not resolved | `start_protocol` `debate` or `meeting` with only the people involved |
| A decision others must review before it sticks | `request_review` or `start_protocol` `review_decision` |
| Ongoing work between two or three agents | `create_group` and message the group |

Structured protocols are optional. A two-message exchange often beats a meeting.

## Asking for help

- Ask the teammate whose description or permissions fit the gap. Look at `list_members`.
- Make the request self-contained: what you need, why, what you already know, the format
  you want back. Teammates cannot see your private context.
- Use `kind=request` when you need an answer. You are woken when the reply arrives; do not
  poll or send reminders.
- Answer requests addressed to you with `reply_to`. Your final message of a turn answers
  any request you did not answer explicitly.

## Sharing work

- Put long results in `share_artifact` and send the artifact id, not the whole text.
- Record agreed outcomes with `record_decision` so later work does not reopen them.
- Mark tasks `completed` only after checking the result; otherwise `blocked` or `failed`
  with a note saying what is missing.

## Cost control

- Message one teammate rather than `all`. Broadcast only facts everyone needs.
- Do not forward a message just to keep someone informed; they can read the task board.
- Stop when the goal is met. Do not start a meeting to celebrate or to restate results.
- If the runtime reports a budget or hop limit, summarize what you have and end your turn.
