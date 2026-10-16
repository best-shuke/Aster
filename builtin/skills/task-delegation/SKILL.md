---
name: task-delegation
description: Split work into tasks on the team board, assign them to suitable teammates, express dependencies, run independent tasks in parallel, and integrate and verify the results.
---

# Task delegation

## Decide whether to split

Split only when pieces are separable and someone else is better placed or can run in
parallel. Tightly coupled edits to the same file are faster done by one agent.

## Write good tasks

Each `create_task` should stand alone:

- **title**: the outcome, not the activity ("Benchmark 3 vector stores", not "research").
- **description**: inputs, constraints, acceptance criteria and the expected deliverable
  (file path, artifact, or a short report).
- **assignee**: the member whose description and tools fit. Check `list_members`; a
  read-only member cannot edit files.
- **depends_on**: task ids that must complete first. The runtime holds the task and wakes
  the assignee only when its dependencies complete, so independent tasks run in parallel.

Leave `assignee` empty only for work anyone may claim; claim it with
`update_task status=in_progress`.

## While executing an assigned task

- Set `in_progress` when you start if the work is long.
- If you are blocked, set `blocked` with a precise note and message the teammate who can
  unblock you. Do not silently stop.
- Finish with `completed` and a `result` that states what was done and how it was
  verified, or `failed` with the reason. If you end your turn without updating it, your
  final message becomes the result and is flagged as not explicitly confirmed.

## Integrating

The creator is notified when tasks finish. Read the results, check the actual files or
evidence for anything that matters, resolve conflicts between results, and only then
report completion. A task status of `completed` is a claim, not proof.

If plans change, cancel tasks you created and no longer need with
`update_task status=cancelled`. A message saying "never mind" is not enough: an open
task stays on the board and wakes its assignee again in a later turn.

Nesting is limited: a task created while working on a task is one level deeper, and the
runtime rejects tasks beyond the environment's depth limit. Prefer flat task lists.
