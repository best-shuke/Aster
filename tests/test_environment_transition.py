import json
from unittest.mock import Mock, patch

from client_runtime import ClientState
from environment_transition import ConversationState, EnvironmentTransition
from tests.test_environments import EnvironmentTestCase
from tests.test_tools import call, python_command


class EnvironmentTransitionTests(EnvironmentTestCase):
    def start_background(self, conversation):
        started = json.loads(
            conversation.tools.execute(
                call(
                    "exec",
                    {
                        "command": python_command("print('READY'); input()"),
                        "background": True,
                    },
                )
            )
        )
        session_id = started["data"]["session_id"]
        conversation.tools.execute(
            call(
                "monitor",
                {
                    "action": "wait",
                    "session_id": session_id,
                    "wait_for": "READY",
                    "wait_seconds": 2,
                },
            )
        )
        return conversation.tools.processes.lookup(session_id)

    def test_committed_environment_switch_stops_existing_background_processes(self):
        environment = self.environment(tools={"deny": ["exec"]})
        conversation = self.conversation()
        session = self.start_background(conversation)
        warnings = conversation.apply_environment(environment, keep=True)
        self.assertTrue(session.done.is_set())
        self.assertEqual(session.state, "stopped")
        self.assertTrue(any("后台命令" in warning for warning in warnings))
        self.assertFalse(conversation.tools.tool_available("monitor"))

    def test_failed_environment_switch_keeps_existing_background_processes(self):
        environment = self.environment(tools={"deny": ["exec"]})
        conversation = self.conversation()
        previous = conversation.tools
        session = self.start_background(conversation)
        with (
            patch.object(
                conversation.sessions, "transition", side_effect=OSError("storage unavailable")
            ),
            self.assertRaises(OSError),
        ):
            conversation.apply_environment(environment, keep=True)
        self.assertIs(conversation.tools, previous)
        self.assertEqual(session.state, "running")
        self.assertIsNone(session.process.poll())

    def test_prepare_isolated_candidates_without_changing_live_state(self):
        environment = self.environment(reasoning_effort="low", tools={"deny": ["exec"]})
        conversation = self.conversation()
        conversation.messages.append({"role": "user", "content": "preserve"})
        conversation.tools.task_title = "preserve task"
        conversation.save()
        previous = ConversationState.capture(conversation)
        disk = self.settings.session_file.read_bytes()
        transition = EnvironmentTransition(conversation, keep=True)
        prepared = transition.prepare(environment, ClientState.capture(conversation.client))
        self.assertEqual(ConversationState.capture(conversation), previous)
        self.assertEqual(self.settings.session_file.read_bytes(), disk)
        self.assertIsNot(prepared.tools, conversation.tools)
        self.assertIsNot(prepared.cache_stats, conversation.client.cache_stats)
        self.assertEqual(prepared.settings.reasoning_effort, "low")
        self.assertFalse(prepared.tools.tool_available("exec"))
        self.assertIs(prepared.tools.registry["read"].handler.__self__, prepared.tools)

    def test_resources_are_revalidated_after_preparation_before_persistence(self):
        environment = self.environment()
        conversation = self.conversation()
        conversation.save()
        previous = ConversationState.capture(conversation)
        disk = self.settings.session_file.read_bytes()
        original_prepare = EnvironmentTransition.prepare

        def prepare_then_revoke(transition, environment, client_state):
            prepared = original_prepare(transition, environment, client_state)
            conversation.environments.deny(environment.environment_id)
            return prepared

        with (
            patch.object(EnvironmentTransition, "prepare", prepare_then_revoke),
            self.assertRaises(ValueError),
        ):
            conversation.apply_environment(environment)
        self.assertEqual(ConversationState.capture(conversation), previous)
        self.assertEqual(self.settings.session_file.read_bytes(), disk)

    def test_base_exception_during_commit_restores_state_and_closes_only_candidate(self):
        environment = self.environment()
        old_manager, candidate = Mock(), Mock()
        old_manager.server_count = 0
        old_manager.servers = {}
        conversation = self.conversation(manager=old_manager)
        previous = ConversationState.capture(conversation)
        old_settings = conversation.client.settings
        transition = EnvironmentTransition(conversation)
        original_prepare = transition.prepare

        def prepare_with_candidate(environment, client_state):
            prepared = original_prepare(environment, client_state)
            transition.manager = candidate
            return prepared

        with (
            patch.object(transition, "prepare", side_effect=prepare_with_candidate),
            patch.object(transition, "commit", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            transition.run(environment)
        self.assertEqual(ConversationState.capture(conversation), previous)
        self.assertIs(conversation.client.settings, old_settings)
        candidate.close.assert_called_once_with()
        old_manager.close.assert_not_called()
