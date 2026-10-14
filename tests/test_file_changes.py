import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from file_changes import FileMergeConflict, FileTransaction, merge_text
from file_state import FileChangedError, atomic_write_text, file_digest, file_lock
from tool_runtime import ArtifactStore, ExecutionContext


class TextMergeTests(unittest.TestCase):
    def test_disjoint_changes_merge_including_adjacent_lines(self):
        self.assertEqual(
            merge_text("alpha\nbeta\n", "changed alpha\nbeta\n", "alpha\nchanged beta\n"),
            "changed alpha\nchanged beta\n",
        )

    def test_identical_changes_are_idempotent(self):
        self.assertEqual(merge_text("old\n", "new\n", "new\n"), "new\n")

    def test_overlapping_replacements_and_insertions_are_conflicts(self):
        for base, current, proposed in [
            ("old\n", "first\n", "second\n"),
            ("old\n", "first\nold\n", "second\nold\n"),
            ("first\nsecond\n", "second\n", "changed first\nsecond\n"),
        ]:
            with self.subTest(base=base), self.assertRaises(FileMergeConflict):
                merge_text(base, current, proposed)

    def test_insertions_at_different_positions_merge(self):
        self.assertEqual(
            merge_text("base\n", "first\nbase\n", "base\nlast\n"),
            "first\nbase\nlast\n",
        )

    def test_duplicate_hunks_are_not_applied_twice(self):
        self.assertEqual(
            merge_text(
                "first\nsecond\nthird\n",
                "shared\nsecond\nthird\n",
                "shared\nsecond\nchanged third\n",
            ),
            "shared\nsecond\nchanged third\n",
        )

    def test_identical_overlapping_lines_and_additional_changes_merge(self):
        self.assertEqual(
            merge_text(
                "first\nsecond\nthird\n",
                "shared\nsecond\nthird\n",
                "shared\nchanged second\nthird\n",
            ),
            "shared\nchanged second\nthird\n",
        )

    def test_missing_final_newline_is_preserved(self):
        self.assertEqual(
            merge_text("first\nsecond", "new first\nsecond", "first\nnew second"),
            "new first\nnew second",
        )

    def test_unicode_and_crlf_are_preserved(self):
        self.assertEqual(
            merge_text("原始\r\n末行\r\n", "更新\r\n末行\r\n", "原始\r\n结束\r\n"),
            "更新\r\n结束\r\n",
        )

    def test_large_diverged_files_require_manual_merge(self):
        with patch("file_changes.MAX_AUTO_MERGE_CHARS", 2), self.assertRaises(FileMergeConflict):
            merge_text("base\n", "current\n", "proposed\n")


class FileTransactionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.artifacts = ArtifactStore(self.root)
        self.context = ExecutionContext(time.monotonic() + 30)

    def transaction(self):
        return FileTransaction(self.artifacts)

    def test_drafts_are_private_and_reads_see_own_writes(self):
        target = self.root / "shared.txt"
        target.write_text("original")
        changes = self.transaction()
        self.assertEqual(changes.read(target, self.context.remaining), "original")
        changes.write(target, "draft", self.context.remaining)
        self.assertEqual(changes.read(target, self.context.remaining), "draft")
        self.assertEqual(target.read_text(), "original")
        manifest = json.loads(changes.artifact_path.read_text())
        self.assertEqual(manifest["files"][0]["base"], "original")
        self.assertEqual(manifest["files"][0]["proposed"], "draft")

    def test_same_file_disjoint_transactions_preserve_both_changes(self):
        target = self.root / "shared.txt"
        target.write_text("alpha\nbeta\n")
        first, second = self.transaction(), self.transaction()
        first.write(target, "new alpha\nbeta\n", self.context.remaining)
        second.write(target, "alpha\nnew beta\n", self.context.remaining)
        self.assertEqual(first.commit(self.context.remaining)["status"], "merged")
        self.assertEqual(second.commit(self.context.remaining)["status"], "merged")
        self.assertEqual(target.read_text(), "new alpha\nnew beta\n")

    def test_conflict_preserves_existing_changes_and_entire_incoming_transaction(self):
        shared, other = self.root / "shared.txt", self.root / "other.txt"
        shared.write_text("original\n")
        other.write_text("unchanged\n")
        first, second = self.transaction(), self.transaction()
        first.write(shared, "first\n", self.context.remaining)
        second.write(shared, "second\n", self.context.remaining)
        second.write(other, "incoming\n", self.context.remaining)
        first.commit(self.context.remaining)
        with self.assertRaises(FileMergeConflict):
            second.commit(self.context.remaining)
        self.assertEqual(shared.read_text(), "first\n")
        self.assertEqual(other.read_text(), "unchanged\n")
        self.assertTrue(second.artifact_path.is_file())

    def test_creating_the_same_path_with_different_content_conflicts(self):
        target = self.root / "new.txt"
        first, second = self.transaction(), self.transaction()
        first.write(target, "first", self.context.remaining)
        second.write(target, "second", self.context.remaining)
        self.assertFalse(target.exists())
        first.commit(self.context.remaining)
        with self.assertRaises(FileMergeConflict):
            second.commit(self.context.remaining)
        self.assertEqual(target.read_text(), "first")

    def test_external_disjoint_change_is_preserved_and_overlapping_change_is_rejected(self):
        target = self.root / "shared.txt"
        target.write_text("alpha\nbeta\n")
        changes = self.transaction()
        changes.write(target, "worker\nbeta\n", self.context.remaining)
        target.write_text("alpha\nuser\n")
        changes.commit(self.context.remaining)
        self.assertEqual(target.read_text(), "worker\nuser\n")
        overlapping = self.transaction()
        overlapping.write(target, "other worker\nuser\n", self.context.remaining)
        target.write_text("user wins\nuser\n")
        with self.assertRaises(FileMergeConflict):
            overlapping.commit(self.context.remaining)
        self.assertEqual(target.read_text(), "user wins\nuser\n")

    def test_cancelled_commit_leaves_workspace_unchanged(self):
        target = self.root / "shared.txt"
        target.write_text("original")
        changes = self.transaction()
        changes.write(target, "draft", self.context.remaining)
        self.context.cancelled.set()
        with self.assertRaises(TimeoutError):
            changes.commit(self.context.remaining)
        self.assertEqual(target.read_text(), "original")

    def test_io_failure_rolls_back_already_written_files(self):
        first, second = self.root / "a.txt", self.root / "b.txt"
        first.write_text("original first")
        second.write_text("original second")
        changes = self.transaction()
        changes.write(first, "new first", self.context.remaining)
        changes.write(second, "new second", self.context.remaining)

        def fail_second(path, content, **kwargs):
            if path == second:
                raise OSError("simulated write failure")
            return atomic_write_text(path, content, **kwargs)

        with (
            patch("file_changes.atomic_write_text", side_effect=fail_second),
            self.assertRaises(OSError),
        ):
            changes.commit(self.context.remaining)
        self.assertEqual(first.read_text(), "original first")
        self.assertEqual(second.read_text(), "original second")

    def test_failed_draft_persistence_does_not_stage_a_failed_write(self):
        target = self.root / "shared.txt"
        target.write_text("original")
        changes = self.transaction()
        with (
            patch.object(changes, "persist", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            changes.write(target, "draft", self.context.remaining)
        self.assertEqual(changes.changes, [])
        self.assertEqual(target.read_text(), "original")


class FileLockTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.target = Path(temporary.name) / "shared.txt"
        self.target.write_text("original")

    def test_concurrent_writes_with_same_base_cannot_both_succeed(self):
        version = file_digest(self.target)
        started = threading.Barrier(2)

        def writer(content):
            started.wait(timeout=5)
            try:
                atomic_write_text(self.target, content, expected=version)
                return "ok"
            except FileChangedError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(writer, "first")
            second = workers.submit(writer, "second")
            self.assertEqual(sorted([first.result(), second.result()]), ["conflict", "ok"])

    def test_file_lock_is_reentrant_and_waiting_respects_deadlines(self):
        result = []

        def wait_for_lock():
            context = ExecutionContext(time.monotonic() + 0.02)
            try:
                with file_lock(self.target, context.remaining):
                    result.append("acquired")
            except TimeoutError:
                result.append("timeout")

        with file_lock(self.target), file_lock(self.target):
            worker = threading.Thread(target=wait_for_lock)
            worker.start()
            worker.join(timeout=5)
        self.assertEqual(result, ["timeout"])

    def test_another_process_cannot_write_while_the_file_is_locked(self):
        program = "\n".join(
            [
                "import sys, time",
                "from pathlib import Path",
                "from file_state import atomic_write_text",
                "from tool_runtime import ExecutionContext",
                "context = ExecutionContext(time.monotonic() + 0.05)",
                "try:",
                "    atomic_write_text(Path(sys.argv[1]), 'child write', check=context.remaining)",
                "except TimeoutError:",
                "    print('timeout')",
            ]
        )
        with file_lock(self.target):
            process = subprocess.run(
                [sys.executable, "-c", program, str(self.target)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        self.assertEqual(process.stdout.strip(), "timeout")
        self.assertEqual(self.target.read_text(), "original")
