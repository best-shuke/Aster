import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from context_management import summarize
from deepseek_client import CompletionError, DeepSeekClient
from task_runtime import TaskProgress, TaskRuntime
from tests.support import make_settings


def event(delta, finish=None):
    return (
        "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n"
    ).encode()


class RecoveryRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = make_settings(
            api_key="test",
            base_url="https://api.cline.bot/api/v1",
            model="cline-pass/deepseek-v4.1-flash",
            session_file=root / "session.json",
            workspace=root,
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=3600,
            max_subagents=1,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_default_live_stream_can_cross_the_old_reasoning_cutoff(self):
        work = "".join(f"Unique equation {i}: a{i} = {i * 17}.\n" for i in range(4000))
        raw = event({"reasoning": work}) + event({"content": "finished"}, "stop")
        with patch("http_transport.open_url", return_value=io.BytesIO(raw)):
            answer = DeepSeekClient(self.settings).complete([], on_content=lambda _: None)
        self.assertEqual(answer["content"], "finished")
        self.assertEqual(answer["reasoning"], work)
        self.assertIsNone(self.settings.max_reasoning_chars)
        self.assertIsNone(self.settings.generation_timeout_seconds)

    def test_summary_and_final_requests_keep_default_max_and_final_tools(self):
        client = DeepSeekClient(self.settings)
        definitions = [
            {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
        ]
        with patch(
            "http_transport.open_url",
            side_effect=lambda *a, **k: io.BytesIO(event({"content": "done"}, "stop")),
        ) as opened:
            client.complete([], tools=definitions, on_content=lambda _: None)
            client.complete(
                [], purpose="summary", max_output_tokens=4096, on_content=lambda _: None
            )
            client.complete(
                [],
                tools=definitions,
                purpose="final",
                max_output_tokens=8192,
                on_content=lambda _: None,
            )
        solver, summary, final = [json.loads(c.args[0].data) for c in opened.call_args_list]
        for body in (solver, summary, final):
            self.assertEqual(body["thinking"], {"type": "enabled"})
            self.assertEqual(body["reasoning"], {"enabled": True, "effort": "max"})
        self.assertEqual(final["tools"], definitions)
        self.assertEqual(final["tool_choice"], "auto")
        self.assertNotIn("tools", summary)

    def test_finalization_is_a_real_mode_transition_and_keeps_unfinished_work(self):
        clock = [100.0]
        requests = []

        class Client:
            def complete(inner, messages, **kwargs):
                requests.append({"messages": json.loads(json.dumps(messages)), **kwargs})
                if len(requests) == 1:
                    clock[0] = 990.0
                    raise CompletionError(
                        "generation_budget",
                        "reserve boundary",
                        partial_message={
                            "reasoning": "Computed value 323; final answer still needed."
                        },
                    )
                return {"role": "assistant", "content": "323"}

        definitions = [{"type": "function", "function": {"name": "read"}}]
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            runtime = TaskRuntime(Client(), self.settings, 1100)
            answer = runtime.generate(
                [{"role": "user", "content": "Compute the value"}],
                {"role": "system", "content": "Solve the task"},
                definitions,
            )
        self.assertEqual(answer["content"], "323")
        self.assertEqual([r["purpose"] for r in requests], ["solve", "final"])
        self.assertEqual(requests[-1]["tools"], definitions)
        self.assertIn("Computed value 323", str(requests[-1]["messages"]))
        self.assertEqual(runtime.recovery.summary_requests, 0)

    def test_failed_later_summary_chunk_keeps_earlier_accumulated_state(self):
        requests = []

        class Client:
            def complete(inner, messages, **kwargs):
                requests.append(messages)
                if len(requests) == 1:
                    return {
                        "content": "Verified earlier finding: calibration=17. Failed approach: missing sage."
                    }
                raise CompletionError(
                    "output_limit",
                    "cut off",
                    partial_message={"reasoning": "invented calibration=99"},
                )

        source = [
            {
                "role": "assistant",
                "reasoning": "x" * 65000 + "MIDDLE MARKER" + "y" * 65000 + "LAST FINDING",
            }
        ]
        result = summarize(Client(), self.settings, source)
        self.assertEqual(len(requests), 2)
        self.assertIn("calibration=17", result)
        self.assertIn("missing sage", result)
        self.assertIn("LAST FINDING", result)
        self.assertNotIn("calibration=99", result)

    def test_large_summary_has_per_chunk_time_without_consuming_the_task_budget(self):
        clock, deadlines = [100.0], []

        class Client:
            def complete(inner, messages, **kwargs):
                deadlines.append(kwargs["deadline"])
                clock[0] += 20  # More than 45 seconds overall, but each piece is prompt.
                return {"content": "Accumulated result with previous constraints."}

        with patch("time.monotonic", side_effect=lambda: clock[0]):
            result = summarize(
                Client(),
                self.settings,
                [{"role": "assistant", "reasoning": "x" * 150000}],
                deadline=1100,
            )
        self.assertEqual(len(deadlines), 3)
        self.assertEqual(deadlines, [145, 165, 185])
        self.assertIn("Accumulated result", result)

    def test_missing_command_detection_handles_stderr_redirection_but_not_hidden_stdout(self):
        progress = TaskProgress(self.settings.workspace)

        def observe(command, output):
            progress.observe(
                {"function": {"name": "exec", "arguments": json.dumps({"command": command})}},
                {"status": "ok", "data": {"exit_code": 0, "output": output}},
            )

        observe(
            "which sage gp python3 2>/dev/null; python3 --version",
            "/usr/bin/python3\nPython 3.13\n",
        )
        self.assertEqual(progress.missing_commands, {"sage", "gp"})
        observe("command -v sage 2> /dev/null", "/opt/bin/sage\n")
        self.assertEqual(progress.missing_commands, {"gp"})
        observe("which magma >/dev/null", "")
        observe("false && which other", "")
        self.assertNotIn("magma", progress.missing_commands)
        self.assertNotIn("other", progress.missing_commands)

    def test_different_history_tools_prompt_a_progress_review(self):
        progress = TaskProgress(self.settings.workspace)
        for name, arguments in [
            ("read", {"path": ".agent-artifacts/a.unfinished.json"}),
            ("exec", {"command": "head scratch/a.unfinished.json.txt"}),
            ("grep", {"path": ".agent-artifacts", "pattern": "answer"}),
        ]:
            progress.observe(
                {"function": {"name": name, "arguments": json.dumps(arguments)}},
                {"status": "ok", "data": {"output": "earlier speculation"}},
            )
        self.assertIn("progress_review", progress.guidance())
        self.assertIn("Necessary polling and reads remain allowed", progress.guidance())


if __name__ == "__main__":
    unittest.main()
