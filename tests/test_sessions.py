import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sessions import SessionConflictError, SessionStore


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = SessionStore(self.root / ".deepseek-session.json")

    def tearDown(self):
        self.temporary.cleanup()

    def data(self, session_id="a" * 32, updated_at="2026-09-30T08:00:00+00:00"):
        return {
            "version": 1,
            "session_id": session_id,
            "updated_at": updated_at,
            "messages": [
                {"role": "user", "content": "之前的\n 对话"},
                {"role": "assistant", "content": "之前的回答", "reasoning": "思考内容"},
            ],
            "tool_runs": [],
        }

    def test_saved_sessions_and_tool_output_ignore_themselves_in_git(self):
        from tool_runtime import ArtifactStore

        self.store.save(self.data())
        self.assertEqual((self.store.directory / ".gitignore").read_text().splitlines()[-1], "*")
        self.assertEqual(len(self.store.list()), 1)
        artifacts = ArtifactStore(self.root)
        artifacts.new_path(".json")
        self.assertEqual((artifacts.directory / ".gitignore").read_text().splitlines()[-1], "*")

    def test_save_preserves_unicode_in_active_file_and_snapshot(self):
        data = self.data()
        self.store.save(data)

        self.assertEqual(self.store.read(), data)
        self.assertEqual(self.store.read(self.store.snapshot_path(data["session_id"])), data)
        session = self.store.list()[0]
        self.assertEqual(session.title, "之前的 对话")
        self.assertEqual(session.message_count, 2)
        self.assertEqual(session.session_id, data["session_id"])
        self.assertIn("思考内容", self.store.active_file.read_text())
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_saving_again_updates_one_snapshot_instead_of_adding_another(self):
        data = self.data()
        self.store.save(data)
        data["messages"].append({"role": "user", "content": "接着提问"})
        self.store.save(data)

        self.assertEqual(len(list(self.store.directory.glob("*.json"))), 1)
        self.assertEqual(self.store.list()[0].message_count, 3)
        self.assertEqual(self.store.read(self.store.snapshot_path(data["session_id"])), data)

    def test_list_orders_sessions_by_last_saved_time(self):
        self.store.save(self.data("a" * 32, "2026-09-30T08:00:00+00:00"))
        self.store.save(self.data("b" * 32, "2026-09-30T09:00:00+00:00"))

        self.assertEqual(
            [session.session_id for session in self.store.list()], ["b" * 32, "a" * 32]
        )

    def test_empty_conversations_are_not_archived(self):
        data = self.data()
        data["messages"] = []
        self.store.save(data)

        self.assertEqual(self.store.list(), [])
        self.assertFalse(self.store.directory.exists())
        self.assertEqual(self.store.read(), data)

    def test_corrupt_snapshots_are_reported_without_hiding_valid_sessions(self):
        self.store.save(self.data())
        broken = self.store.directory / "broken.json"
        broken.write_text("{bad", encoding="utf-8")

        self.assertEqual(len(self.store.list()), 1)
        self.assertEqual(len(self.store.diagnostics), 1)
        self.assertIn("broken.json", self.store.diagnostics[0])
        self.assertEqual(broken.read_text(), "{bad")
        self.store.list()
        self.assertEqual(len(self.store.diagnostics), 1)

    def test_invalid_session_ids_cannot_escape_the_archive_directory(self):
        for session_id in ("../session", "/tmp/session", "", "x" * 32, None):
            with self.subTest(session_id=session_id), self.assertRaises(ValueError):
                self.store.snapshot_path(session_id)
        self.assertFalse(self.store.directory.exists())

    def test_failed_snapshot_write_preserves_both_previous_files(self):
        data = self.data()
        self.store.save(data)
        data["messages"].append({"role": "user", "content": "未保存的新内容"})
        with (
            patch.object(Path, "replace", side_effect=OSError("disk full")),
            self.assertRaises(OSError),
        ):
            self.store.save(data)

        self.assertEqual(len(self.store.read()["messages"]), 2)
        self.assertEqual(
            len(self.store.read(self.store.snapshot_path(data["session_id"]))["messages"]), 2
        )
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_custom_active_file_has_its_own_archive_directory(self):
        custom = SessionStore(self.root / "custom.json")
        self.store.save(self.data("a" * 32))
        custom.save(self.data("b" * 32))

        self.assertEqual(custom.directory, self.root / "custom.sessions")
        self.assertEqual([session.session_id for session in custom.list()], ["b" * 32])
        self.assertEqual([session.session_id for session in self.store.list()], ["a" * 32])

    def test_invalid_session_structure_is_rejected(self):
        changes = [
            {"version": 2},
            {"messages": "bad"},
            {"messages": [{"content": "missing role"}]},
            {"messages": [{"role": "assistant", "tool_calls": "bad"}]},
            {"messages": [{"role": "assistant", "tool_calls": ["bad"]}]},
            {"messages": [{"role": "assistant", "tool_calls": [{"id": "call"}]}]},
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": "call", "type": "function", "function": "bad"}],
                    }
                ]
            },
            {"messages": [{"role": "tool", "tool_call_id": []}]},
            {"tool_runs": "bad"},
            {"task_state": "bad"},
            {"last_outcome": []},
            {"session_id": "../session"},
            {"environment_id": ""},
            {"environment_id": 1},
            {"environment_id": "x" * 101},
            {"environment_ref": {"path": "/example", "fingerprint": "a" * 64}},
            {"environment_id": "example", "environment_ref": []},
            {"environment_id": "example", "environment_ref": {"fingerprint": "a" * 64}},
            {"environment_id": "example", "environment_ref": {"path": "/example"}},
            {
                "environment_id": "example",
                "environment_ref": {"path": "/example", "fingerprint": "A" * 64},
            },
            {
                "environment_id": "example",
                "environment_ref": {"path": "/example", "fingerprint": "a" * 63},
            },
        ]
        for change in changes:
            with self.subTest(change=change):
                self.store.active_file.write_text(json.dumps({**self.data(), **change}))
                with self.assertRaisesRegex(RuntimeError, "Cannot load session file"):
                    self.store.read()

    def test_read_keeps_legacy_optional_fields_and_unknown_roles(self):
        data = {"version": 1, "messages": [{"role": "legacy"}, {"role": "tool"}]}
        self.store.active_file.write_text(json.dumps(data))
        self.assertEqual(self.store.read(), data)

    def test_invalid_read_does_not_accept_a_new_active_revision(self):
        data = self.data()
        self.store.save(data)
        invalid = json.dumps({**data, "environment_id": ""})
        self.store.active_file.write_text(invalid)

        with self.assertRaisesRegex(RuntimeError, "invalid environment ID"):
            self.store.read()
        with self.assertRaises(SessionConflictError):
            self.store.save(data)
        self.assertEqual(self.store.active_file.read_text(), invalid)
        self.assertEqual(self.store.read(self.store.snapshot_path(data["session_id"])), data)
