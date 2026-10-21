import io
import os
import select
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from rich.markdown import Markdown
from rich.segment import Segment

from ui import (
    Console,
    _CommandCompleter,
    _display_width,
    _edit_stats,
    _ellipsize,
    _git_branch,
    _reasoning_text,
    _short_path,
    _TaskInputMonitor,
    _wrap_words,
)


class CommandCompletionTests(unittest.TestCase):
    def completions(self, text, cursor_position=None, skill_provider=None):
        document = Document(text, cursor_position=cursor_position)
        return list(_CommandCompleter(skill_provider).get_completions(document, CompleteEvent()))

    def test_slash_lists_commands_with_descriptions_and_aliases(self):
        candidates = self.completions("/")
        commands = [candidate.text for candidate in candidates]

        self.assertEqual(commands, sorted(commands))
        for command in ("/effort", "/help", "/new", "/clear", "/tasks", "/todo", "/quit"):
            self.assertIn(command, commands)
        self.assertTrue(all(candidate.display_meta_text for candidate in candidates))

    def test_command_prefix_is_replaced_without_removing_leading_whitespace(self):
        for text in ("/eff", "  /eff"):
            with self.subTest(text=text):
                candidates = self.completions(text)
                self.assertEqual([candidate.text for candidate in candidates], ["/effort"])
                self.assertEqual(candidates[0].start_position, -4)

    def test_ambiguous_prefix_lists_only_matching_commands(self):
        self.assertEqual(
            [candidate.text for candidate in self.completions("/e")],
            ["/effort", "/env", "/environment", "/exit"],
        )

    def test_workflow_command_supports_prefix_completion(self):
        self.assertEqual([candidate.text for candidate in self.completions("/work")], ["/workflow"])
        self.assertEqual(self.completions("/workflow new"), [])

    def test_completed_commands_arguments_prose_and_multiline_pastes_do_not_complete(self):
        for text in (
            "",
            "ordinary text",
            "请介绍 /eff",
            "/unknown",
            "/effort",
            "/effort ",
            "/effort h",
            "/template add /eff",
            "/skill:example",
            "first line\n/eff",
            "/eff\nsecond line",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.completions(text), [])

    def test_cursor_inside_a_command_does_not_duplicate_its_suffix(self):
        self.assertEqual(self.completions("/effort", cursor_position=4), [])

    def test_command_prefix_before_existing_arguments_can_complete(self):
        candidates = self.completions("/eff max", cursor_position=4)
        self.assertEqual([candidate.text for candidate in candidates], ["/effort"])
        self.assertEqual(candidates[0].start_position, -4)

    def test_skill_commands_complete_with_descriptions(self):
        skills = {"pdf-tools": "Extract PDF tables.", "release": "Manual release."}
        candidates = self.completions("/skill:", skill_provider=lambda: skills)

        self.assertEqual(
            [candidate.text for candidate in candidates], ["/skill:pdf-tools", "/skill:release"]
        )
        self.assertEqual(candidates[0].display_meta_text, "Extract PDF tables.")
        self.assertEqual(candidates[0].start_position, -7)
        commands = [
            candidate.text for candidate in self.completions("/", skill_provider=lambda: skills)
        ]
        self.assertEqual(commands, sorted(commands))
        self.assertIn("/skills", commands)

    def test_skills_arguments_complete_names_and_all_without_replacing_command(self):
        skills = {"alpha": "Alpha skill.", "beta": "Beta skill."}
        candidates = self.completions("/skills ", skill_provider=lambda: skills)
        self.assertEqual([candidate.text for candidate in candidates], ["all", "alpha", "beta"])
        self.assertTrue(all(candidate.start_position == 0 for candidate in candidates))
        candidates = self.completions("  /skills be", skill_provider=lambda: skills)
        self.assertEqual([candidate.text for candidate in candidates], ["beta"])
        self.assertEqual(candidates[0].start_position, -2)

    def test_skill_completions_follow_provider_changes(self):
        skills = {"alpha": "Alpha."}
        completer = _CommandCompleter(lambda: skills)
        document = Document("/skill:")
        self.assertEqual(
            [item.text for item in completer.get_completions(document, CompleteEvent())],
            ["/skill:alpha"],
        )
        skills.clear()
        skills["beta"] = "Beta."
        self.assertEqual(
            [item.text for item in completer.get_completions(document, CompleteEvent())],
            ["/skill:beta"],
        )
        skills.clear()
        self.assertEqual(list(completer.get_completions(document, CompleteEvent())), [])

    def test_skill_arguments_multiline_and_cursor_inside_suffix_do_not_complete(self):
        skills = {"alpha": "Alpha."}
        for text in (
            "/skill:alpha",
            "/skill:alpha task",
            "/skills alpha",
            "/skills alpha extra",
            "/skills \nalpha",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.completions(text, skill_provider=lambda: skills), [])
        self.assertEqual(
            self.completions("/skill:alpha", cursor_position=9, skill_provider=lambda: skills), []
        )


class ConsoleLayoutTests(unittest.TestCase):
    def test_no_color_does_not_disable_terminal_live_rendering(self):
        class TerminalOutput(io.StringIO):
            def isatty(self):
                return True

        with (
            patch("sys.stdout", TerminalOutput()),
            patch("sys.stdin.isatty", return_value=True),
            patch.dict(os.environ, {"NO_COLOR": "1"}),
        ):
            console = Console()
            self.assertTrue(console.interactive)
            self.assertFalse(console.colors)
            self.assertTrue(console._rich().is_terminal)
            self.assertTrue(console._rich().no_color)

    def test_ellipsize_respects_wide_chinese_characters(self):
        shortened = _ellipsize("正在接收一段很长的回复", 11)

        self.assertLessEqual(_display_width(shortened), 11)
        self.assertTrue(shortened.endswith("…"))

    def test_banner_prioritizes_runtime_state(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console(
                "deepseek-v4",
                workspace=Path("/tmp/workspace"),
                reasoning_effort="high",
                websearch_status="可用",
                mcp_status="2 个服务",
            )
            console.banner(Path("/tmp/session.json"), history_count=4)

        rendered = output.getvalue()
        self.assertIn("ASTER", rendered)
        self.assertNotIn("CODING COMPANION", rendered)
        self.assertNotIn("DEEPSEEK", rendered)
        self.assertIn("deepseek-v4", rendered)
        self.assertIn("/tmp/workspace", rendered)
        self.assertIn("推理 high", rendered)
        self.assertIn("联网 可用", rendered)
        self.assertIn("MCP 2 个服务", rendered)
        self.assertIn("已恢复", rendered)
        self.assertNotIn("\033[", rendered)

    def test_narrow_banner_keeps_aster_brand(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console("test-model")
            with patch.object(console, "terminal_width", return_value=30):
                console.banner(Path("/tmp/session.json"))

        self.assertIn("ASTER", output.getvalue())
        self.assertNotIn("CODING COMPANION", output.getvalue())

    def test_banner_shows_runtime_metrics_without_explanatory_copy(self):
        for width in (24, 80, 200):
            with self.subTest(width=width):
                output = io.StringIO()
                with (
                    patch("sys.stdout", output),
                    patch.object(Console, "terminal_width", return_value=width),
                ):
                    console = Console("cline-pass/deepseek-v4.1-flash")
                    console.update_context(
                        {
                            "tokens": 5800,
                            "window_tokens": 1000000,
                            "budget_tokens": 983616,
                            "source": "estimate",
                        }
                    )
                    console.banner(Path("/tmp/session.json"))
                rendered = output.getvalue()
                self.assertIn("5.8k", rendered)
                self.assertIn("0.6%", rendered)
                self.assertIn("新会话", rendered)
                for label in ("本地估算", "系统提示", "尚无对话", "≈"):
                    self.assertNotIn(label, rendered)
                self.assertIn("工作区", rendered)
                self.assertIn("上下文", rendered)
                self.assertTrue(
                    all(_display_width(line) <= width for line in rendered.splitlines())
                )
                if width >= 80:
                    self.assertNotIn("TPS —", rendered)
                    self.assertLessEqual(len(rendered.splitlines()), 6)
                    self.assertTrue(all(line.strip() for line in rendered.splitlines()))
                    self.assertIn("╭", rendered)
                    self.assertIn("╰", rendered)
                    self.assertTrue(
                        all(_display_width(line) <= 100 for line in rendered.splitlines())
                    )

    def test_banner_keeps_capabilities_and_metrics_in_a_compact_panel(self):
        output = io.StringIO()
        with patch("sys.stdout", output), patch.object(Console, "terminal_width", return_value=80):
            console = Console(
                "cline-pass/deepseek-v4.1-flash",
                workspace=Path("/tmp/用户项目/很长的工作区名称/src/project"),
                websearch_status="可用",
                mcp_status="已连接 2/2",
            )
            console.skill_status = {"enabled": True, "loaded": 17}
            console.update_context(
                {"tokens": 5800, "window_tokens": 1000000, "budget_tokens": 983616}
            )
            console.banner(Path("session.json"), history_count=4)
        rendered = output.getvalue()
        for value in ("ASTER", "project", "已恢复 · 4 条记录", "Skills 17", "MCP 已连接 2/2"):
            self.assertIn(value, rendered)
        self.assertEqual(len(rendered.splitlines()), 6)

    def test_session_screen_restores_terminal_after_an_exception(self):
        output = io.StringIO()
        console = Console()
        console.interactive = True
        with (
            patch("sys.stdout", output),
            patch.dict(os.environ, {"TERM": "xterm-256color"}),
            self.assertRaisesRegex(ValueError, "interrupted"),
            console.session_screen(),
        ):
            self.assertTrue(console._screen_active)
            console._live = Mock()
            live = console._live
            output.write("SESSION CONTENT")
            raise ValueError("interrupted")
        live.stop.assert_called_once()
        self.assertIsNone(console._live)
        self.assertFalse(console._screen_active)
        self.assertTrue(output.getvalue().startswith("\033[?1049h\033[2J\033[H"))
        self.assertTrue(output.getvalue().endswith("\033[0m\033[?25h\033[?1049l"))

    def test_errors_nobody_saw_are_repeated_after_leaving_the_session_screen(self):
        for prompted in (False, True):
            console = Console()
            console.interactive = True
            console.colors = False
            errors = io.StringIO()
            with (
                self.subTest(prompted=prompted),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", errors),
                patch.dict(os.environ, {"TERM": "xterm-256color"}),
                patch.object(console, "_read_input", return_value=""),
                console.session_screen(),
            ):
                console.error(ValueError("环境 'x' 未找到"))
                if prompted:
                    console.prompt()
            # A startup failure ends the session before any prompt: show it again.
            self.assertEqual(errors.getvalue().count("环境 'x' 未找到"), 1 if prompted else 2)

    def test_closing_the_inline_trace_redraws_the_latest_exchanges(self):
        from execution_history import ExecutionEntry

        console = Console()
        console.interactive = True
        console.colors = False
        console._screen_active = True
        console.execution_provider = lambda: [ExecutionEntry(1, {"tool": "read"})]
        messages = [
            {"role": "user", "content": "OLD_QUESTION"},
            {"role": "assistant", "content": "OLD_ANSWER"},
            {"role": "user", "content": "LAST_QUESTION"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "read"}}]},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "LAST_ANSWER"},
        ]
        console.history_provider = lambda: messages
        output = io.StringIO()
        with (
            patch("sys.stdout", output),
            patch("trace_ui.TraceInspector") as inspector,
            patch("ui.patch_stdout", return_value=nullcontext()),
        ):
            console.trace()
        inspector.return_value.run.assert_called_once()
        self.assertFalse(inspector.call_args.kwargs["full_screen"])
        rendered = output.getvalue()
        self.assertIn("\033[2J", rendered)
        # Both exchanges are short, so the redraw starts at the earlier question.
        for text in ("OLD_QUESTION", "OLD_ANSWER", "LAST_QUESTION", "LAST_ANSWER"):
            self.assertIn(text, rendered)

    def test_nested_session_screens_enter_and_leave_once(self):
        output = io.StringIO()
        console = Console()
        console.interactive = True
        with (
            patch("sys.stdout", output),
            patch.dict(os.environ, {"TERM": "xterm-256color"}),
            console.session_screen(),
        ):
            with console.session_screen():
                self.assertTrue(console._screen_active)
            self.assertTrue(console._screen_active)
        self.assertEqual(output.getvalue().count("\033[?1049h"), 1)
        self.assertEqual(output.getvalue().count("\033[?1049l"), 1)

    def test_session_screen_skips_redirected_disabled_and_dumb_terminals(self):
        for interactive, enabled, term in (
            (False, True, "xterm-256color"),
            (True, False, "xterm-256color"),
            (True, True, "dumb"),
        ):
            with self.subTest(interactive=interactive, enabled=enabled, term=term):
                output = io.StringIO()
                console = Console()
                console.interactive = interactive
                with (
                    patch("sys.stdout", output),
                    patch.dict(os.environ, {"TERM": term}),
                    console.session_screen(enabled=enabled),
                ):
                    self.assertFalse(console._screen_active)
                self.assertEqual(output.getvalue(), "")

    def test_new_session_preserves_shell_scrollback_in_session_screen(self):
        output = io.StringIO()
        console = Console()
        console.interactive = True
        with (
            patch("sys.stdout", output),
            patch.dict(os.environ, {"TERM": "xterm-256color"}),
            console.session_screen(),
        ):
            console.reset_session(Path("session.json"))
        self.assertIn("\033[2J\033[H", output.getvalue())
        self.assertNotIn("\033[3J", output.getvalue())

    def test_restored_session_banner_does_not_claim_the_context_is_empty(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console()
            console.update_context(
                {
                    "tokens": 12000,
                    "window_tokens": 1000000,
                    "budget_tokens": 983616,
                    "source": "usage",
                }
            )
            console.banner(Path("/tmp/session.json"), history_count=4)
        rendered = output.getvalue()
        self.assertIn("已恢复 · 4 条记录", rendered)
        self.assertNotIn("尚无对话", rendered)
        self.assertNotIn("本地估算", rendered)

    def test_noninteractive_progress_does_not_flood_logs(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console("test-model")
            console.assistant_start()
            for elapsed in range(20):
                console.progress(
                    {
                        "phase": "receiving",
                        "elapsed_seconds": elapsed,
                        "received_bytes": elapsed * 1024,
                    }
                )
            console.progress({"phase": "retry", "attempt": 2})
            console.progress({"phase": "retry", "attempt": 2})
            console.progress({"phase": "retry", "attempt": 3})

        rendered = output.getvalue()
        self.assertNotIn("\033[", rendered)
        self.assertNotIn("已接收", rendered)
        self.assertEqual(rendered.count("第 2 次请求"), 1)
        self.assertEqual(rendered.count("第 3 次请求"), 1)

    def test_rate_limit_wait_is_visible_without_flooding_noninteractive_logs(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console("test-model")
            console.assistant_start()
            for delay in range(20, 0, -1):
                console.progress({"phase": "rate_limit", "delay_seconds": delay})
            console.progress({"phase": "connecting"})
            console.progress({"phase": "rate_limit", "delay_seconds": 5})
        rendered = output.getvalue()
        self.assertEqual(rendered.count("模型服务限流"), 2)
        self.assertNotIn("连接暂时失败", rendered)

    def test_rate_limit_wait_has_distinct_main_and_subagent_status(self):
        console = Console("test-model")
        console.interactive = True
        with patch.object(console, "_set_status") as status:
            console.progress({"phase": "rate_limit", "delay_seconds": 12})
        self.assertIn("模型服务限流", status.call_args.args[0])
        self.assertIn("12.0 秒", status.call_args.args[0])
        activity = console._agent_activity(
            {"activity": "model", "model_phase": "rate_limit", "delay_seconds": 12}
        )
        self.assertIn("模型服务限流", activity)
        self.assertIn("12.0 秒", activity)
        self.assertEqual(
            console._agent_activity(
                {"activity": "model", "model_phase": "rate_limit", "delay_seconds": 0.1}
            ),
            "等待模型服务恢复",
        )

    def test_progress_warning_is_visible_without_stopping_the_task(self):
        for interactive in (False, True):
            with self.subTest(interactive=interactive):
                output = io.StringIO()
                with patch("sys.stdout", output):
                    console = Console()
                    console.task_start("检查目录")
                    console.interactive = interactive
                    console.progress({"phase": "warning", "message": "日志不可写，任务继续"})

                self.assertIn("日志不可写，任务继续", output.getvalue())
                self.assertEqual(console._task_state, "running")

    def test_subagent_warning_is_visible_without_marking_it_failed(self):
        for interactive in (False, True):
            with self.subTest(interactive=interactive):
                output = io.StringIO()
                with patch("sys.stdout", output):
                    console = Console()
                    console.interactive = interactive
                    console.task_update(
                        {
                            "phase": "subagent",
                            "agent_id": "A1",
                            "title": "检查目录",
                            "state": "running",
                            "activity": "warning",
                            "warning": "日志不可写，任务继续",
                        }
                    )

                self.assertEqual(output.getvalue().count("日志不可写，任务继续"), 1)
                self.assertIn("子代理 A1", output.getvalue())
                self.assertEqual(console._agents["A1"]["state"], "running")

    def test_interactive_status_is_throttled_and_width_bounded(self):
        output = io.StringIO()
        console = Console("very-long-model-name")
        console.interactive = True
        console.colors = False
        event = {
            "phase": "receiving",
            "elapsed_seconds": 12,
            "received_bytes": 4096,
        }

        with (
            patch("sys.stdout", output),
            patch.object(console, "terminal_width", return_value=30),
            patch("ui.time.monotonic", side_effect=[10.0, 10.2, 11.2]),
        ):
            console.assistant_start()
            console.progress(event)
            console.progress(event)
            console.progress(event)

        rendered = output.getvalue()
        updates = rendered.split("\r\033[2K")[1:]
        self.assertEqual(len(updates), 3)  # Initial state plus two timed progress updates.
        self.assertTrue(all(_display_width(line) <= 29 for line in updates))

    def test_history_defaults_can_limit_large_sessions(self):
        messages = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"message-{index}"}
            for index in range(12)
        ]
        output = io.StringIO()
        with patch("sys.stdout", output):
            Console().history(messages, limit=3)

        rendered = output.getvalue()
        self.assertNotIn("message-8", rendered)
        self.assertIn("message-9", rendered)
        self.assertIn("message-11", rendered)
        self.assertIn("/history all", rendered)

    def test_limited_history_truncates_an_individual_large_message(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            Console().history([{"role": "assistant", "content": "x" * 5000}], limit=10)

        rendered = output.getvalue()
        self.assertLess(len(rendered), 4500)
        self.assertIn("内容过长", rendered)

    def test_reasoning_and_answer_are_rendered_as_separate_markdown_sections(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console("test-model")
            console.assistant_start()
            console.reasoning("先检查 **输入**。")
            console.stream("## 结果\n\n- 第一项\n- 第二项\n\n```python\nprint('ok')\n```")
            console.assistant_end()

        rendered = output.getvalue()
        self.assertIn("思考过程", rendered)
        self.assertIn("先检查 输入。", rendered)
        self.assertIn("test-model · 回答", rendered)
        self.assertIn("结果", rendered)
        self.assertIn("第一项", rendered)
        self.assertIn("print('ok')", rendered)
        self.assertNotIn("**输入**", rendered)
        self.assertNotIn("## 结果", rendered)

    def test_live_preview_grows_without_publishing_an_accepted_answer(self):
        output = io.StringIO()
        console = Console()
        console.interactive = True
        console.progress({"phase": "response_delta", "reasoning": "先检查输入", "content": ""})
        with patch("sys.stdout", output):
            console._rich().print(console._live_renderable())
        self.assertIn("先检查输入", output.getvalue())
        self.assertIn("思考过程", output.getvalue())
        self.assertEqual(console._response_content, [])

        output = io.StringIO()
        console.progress({"phase": "response_delta", "content": "第一段", "reasoning": ""})
        console.progress({"phase": "response_delta", "content": "第二段", "reasoning": ""})
        with patch("sys.stdout", output):
            console._rich().print(console._live_renderable())
        self.assertIn("第一段第二段", output.getvalue())
        self.assertEqual(console._response_content, [])

    def test_live_preview_renders_markdown_for_both_reasoning_and_answer(self):
        console = Console()
        console.interactive = True
        console.progress({"phase": "response_delta", "reasoning": "先检查 **输入**。"})
        console.progress(
            {
                "phase": "response_delta",
                "content": "## 结果\n\n- **重点**\n- `sample`\n\n```python\nprint('ok')\n",
            }
        )
        output = io.StringIO()
        with patch("sys.stdout", output):
            console._rich().print(console._live_renderable())

        rendered = output.getvalue()
        for text in ("输入", "结果", "重点", "sample", "print('ok')", "回答"):
            self.assertIn(text, rendered)
        for markup in ("**输入**", "## 结果", "**重点**", "`sample`", "```python"):
            self.assertNotIn(markup, rendered)
        self.assertEqual(console._response_content, [])
        self.assertEqual(console._response_reasoning, [])

    def test_stream_callbacks_update_the_live_response_without_duplicating_delta_text(self):
        console = Console()
        console.interactive = True
        console._live = Mock()
        console.reasoning("思考第一段")
        self.assertEqual(console._preview_reasoning, ["思考第一段"])
        console.stream("回答第一段")
        console.stream("回答第二段")
        self.assertEqual("".join(console._preview_content), "回答第一段回答第二段")
        console.assistant_start()
        console.progress({"phase": "response_delta", "content": "真实流式回答"})
        console.stream("真实流式回答")
        self.assertEqual("".join(console._preview_content), "真实流式回答")

    def test_reset_starts_a_new_scroll_area_for_the_next_stream(self):
        console = Console()
        console.interactive = True
        console.progress({"phase": "response_delta", "content": "旧回答\n" * 100})
        old = console._live_renderable()
        console.progress({"phase": "response_reset"})
        console.progress({"phase": "response_delta", "content": "新回答"})
        new = console._live_renderable()
        self.assertNotEqual(old.key, new.key)
        self.assertIn("新回答", "".join(segment.text for line in new.body for segment in line))
        self.assertNotIn("旧回答", "".join(segment.text for line in new.body for segment in line))

    def test_inline_code_uses_light_color_in_final_markdown_and_live_preview(self):
        for preview in (False, True):
            for dim in (False, True):
                with self.subTest(preview=preview, dim=dim):
                    console = Console()
                    rich_console = console._rich()
                    source = "正文 `sample` 和 ``other``"
                    if preview:
                        renderable = console._preview_markdown(source, 20, dim=dim)
                        segments = [segment for line in renderable.lines for segment in line]
                    else:
                        with (
                            patch.object(console, "_rich", return_value=rich_console),
                            patch.object(rich_console, "print") as print_markdown,
                        ):
                            console._markdown(source, dim=dim)
                        segments = list(rich_console.render(print_markdown.call_args.args[0]))

                    code_segments = [
                        segment for segment in segments if segment.text in {"sample", "other"}
                    ]
                    self.assertEqual(len(code_segments), 2)
                    for segment in code_segments:
                        self.assertEqual(segment.style.color.name, "bright_cyan")
                        self.assertTrue(segment.style.bgcolor.is_default)
                        self.assertTrue(segment.style.bold)
                        self.assertEqual(bool(segment.style.dim), dim)
                    body = next(segment for segment in segments if "正文" in segment.text)
                    self.assertIsNone(body.style.color)

    def test_inline_code_theme_does_not_change_code_block_style(self):
        style = Console()._rich().get_style("markdown.code_block")

        self.assertEqual(style.color.name, "cyan")
        self.assertEqual(style.bgcolor.name, "black")

    def test_split_markdown_delimiters_are_reparsed_as_more_text_arrives(self):
        console = Console()
        console.interactive = True
        fragments = ["准备 **加", "粗** 内容\n\n```py", "thon\nprint('ok')\n", "```\n\n- 完成"]
        for fragment in fragments:
            console.progress({"phase": "response_delta", "content": fragment})
            with patch("sys.stdout", io.StringIO()):
                console._rich().print(console._live_renderable())

        preview = console._preview_markdown("".join(fragments), 20)
        segments = [segment for line in preview.lines for segment in line]
        self.assertTrue(
            any(
                "加粗" in segment.text and segment.style.bold
                for segment in segments
                if segment.style
            )
        )
        text = "".join(segment.text for segment in segments)
        self.assertIn("print('ok')", text)
        self.assertIn("完成", text)
        self.assertNotIn("**加粗**", text)
        self.assertNotIn("```", text)

    def test_partial_headings_lists_tables_links_and_code_blocks_can_render(self):
        console = Console()
        source = (
            "# 标题\n\n**加粗** 和 *斜体*\n\n1. 第一项\n2. 第二项\n\n"
            "| 名称 | 状态 |\n| --- | --- |\n| 测试 | 通过 |\n\n"
            "[说明](https://example.invalid/docs)\n\n```python\nprint('ok')\n```"
        )
        for length in range(1, len(source) + 1):
            with self.subTest(length=length):
                console._preview_markdown(source[:length], 8)

        rendered = console._preview_markdown(source, 100)
        text = "".join(segment.text for line in rendered.lines for segment in line)
        self.assertIn("测试", text)
        self.assertIn("通过", text)
        self.assertNotIn("| --- |", text)

    def test_long_open_code_block_keeps_syntax_context_when_clipped(self):
        console = Console()
        source = "```python\n" + "value = 1\n" * 1000 + "latest = 42\n"
        self.assertGreater(len(source), 8000)
        preview = console._preview_markdown(source, 4)
        self.assertLessEqual(len(preview.lines), 4)
        segments = [segment for line in preview.lines for segment in line]
        self.assertIn("latest = 42", "".join(segment.text for segment in segments))
        self.assertTrue(
            any(
                "42" in segment.text and segment.style.color
                for segment in segments
                if segment.style
            )
        )
        self.assertIn("前文已折叠", "".join(segment.text for segment in preview.lines[0]))

    def test_growing_code_preserves_wrapping_and_multiline_syntax_styles(self):
        sources = (
            ("python", '\tvalue = 1\ntext = """中文\ninside string\n"""\nprint(value)\n'),
            ("javascript", '/* comment\ninside comment\n*/\nconst value = "中文";\n'),
            ("python", 'value = "' + "long words 中文 " * 12 + '"\nprint(value)\n'),
        )
        for width in (24, 80):
            with (
                patch.object(Console, "terminal_width", return_value=width),
                patch("sys.stdout", io.StringIO()),
            ):
                console = Console()
                rich = console._rich()
                for language, code in sources:
                    for length in range(1, len(code) + 1, 3):
                        source = f"```{language}\n" + code[:length]
                        actual = console._preview_markdown(source, None).lines
                        expected = rich.render_lines(
                            console._markdown_renderable(source),
                            rich.options.update(width=width - 1),
                            pad=False,
                        )
                        while expected and not "".join(s.text for s in expected[-1]).strip():
                            expected.pop()
                        self.assertEqual(
                            [list(Segment.simplify(line)) for line in actual],
                            [list(Segment.simplify(line)) for line in expected],
                            (width, language, length),
                        )

    def test_markdown_preview_caches_unchanged_text_and_reflows_on_resize(self):
        console = Console()
        with patch("ui.Markdown", wraps=Markdown) as markdown:
            console._preview_markdown("**内容**", 10)
            console._preview_markdown("**内容**", 3)
            self.assertEqual(markdown.call_count, 1)
            with patch.object(console, "terminal_width", return_value=30):
                console._preview_markdown("**内容**", 3)
            self.assertEqual(markdown.call_count, 2)
            console._preview_markdown("**内容**", 3, dim=True)
            self.assertEqual(markdown.call_count, 3)
            console.progress({"phase": "response_reset"})
            self.assertEqual(console._preview_cache, {})
            console._preview_markdown("**内容**", 3)
            self.assertEqual(markdown.call_count, 4)

    def test_markdown_preview_frees_cached_lines_after_the_response_and_new_session(self):
        console = Console()
        with patch("sys.stdout", io.StringIO()):
            console._preview_markdown("**旧预览**", 3)
            console.assistant_end()
            self.assertEqual(console._preview_cache, {})
            console._preview_markdown("**旧预览**", 3)
            console.reset_session(Path("session.json"))
            self.assertEqual(console._preview_cache, {})

    def test_a_render_finishing_after_reset_cannot_repopulate_the_new_preview_cache(self):
        console = Console()

        def reset_during_render(*args, **kwargs):
            console.progress({"phase": "response_reset"})
            return Markdown(*args, **kwargs)

        with patch("ui.Markdown", side_effect=reset_during_render):
            console._preview_markdown("**old preview**", 3)

        self.assertEqual(console._preview_cache, {})

    def test_render_finishing_after_retry_keeps_the_old_response_identity(self):
        console = Console()
        console.interactive = True
        console.progress({"phase": "response_delta", "content": "旧回答\n" * 30})

        def retry_during_render(*args, **kwargs):
            console.progress({"phase": "response_reset"})
            console.progress({"phase": "response_delta", "content": "新回答"})
            return Markdown(*args, **kwargs)

        with patch("ui.Markdown", side_effect=retry_during_render):
            old = console._live_renderable()
        self.assertEqual(console._stream_body_cache, {})
        current = console._live_renderable()
        self.assertNotEqual(old.key, current.key)
        self.assertIn("新回答", "".join(segment.text for line in current.body for segment in line))

    def test_stream_reset_and_rejected_response_do_not_publish_partial_text(self):
        console = Console()
        console.interactive = True
        output = io.StringIO()
        with patch("sys.stdout", output):
            console.progress({"phase": "response_delta", "content": "old preview"})
            console.progress({"phase": "response_reset"})
            self.assertEqual(console._preview_content, [])
            console.progress({"phase": "response_delta", "content": "rejected preview"})
            console.assistant_end()

        self.assertNotIn("old preview", output.getvalue())
        self.assertNotIn("rejected preview", output.getvalue())
        self.assertEqual(console._preview_content, [])

    def test_accepted_answer_is_rendered_once_after_the_preview(self):
        console = Console()
        console.interactive = True
        output = io.StringIO()
        with patch("sys.stdout", output):
            console.progress({"phase": "response_delta", "content": "final answer"})
            console.stream("final answer")
            console.assistant_end()

        self.assertEqual(output.getvalue().count("final answer"), 1)
        self.assertEqual(console._preview_content, [])
        self.assertEqual(console._response_content, [])

    def test_terminal_receives_each_part_before_the_response_finishes(self):
        first, second = threading.Event(), threading.Event()

        class Output(io.StringIO):
            def isatty(self):
                return True

            def write(self, value):
                result = super().write(value)
                if "第一段实时回答" in value:
                    first.set()
                if "第二段实时回答" in value:
                    second.set()
                return result

        output = Output()
        with patch("sys.stdout", output):
            console = Console()
            console.interactive = True
            console.task_start("实时输出")
            live = console._live
            worker = live._scheduler
            try:
                console.assistant_start()
                console.progress({"phase": "response_delta", "content": "第一段实时回答"})
                self.assertTrue(first.wait(1), "The first part must appear before assistant_end")
                self.assertEqual(console._response_content, [])
                console.progress({"phase": "response_delta", "content": "\n\n第二段实时回答"})
                self.assertTrue(second.wait(1), "The next part must appear during generation")
                console.stream("第一段实时回答\n\n第二段实时回答")
                output.seek(0)
                output.truncate()
                console.assistant_end()
                final_update = output.getvalue()
                self.assertNotIn("第一段实时回答", final_update)
                self.assertNotIn("第二段实时回答", final_update)
                self.assertNotIn("\x1b[2K", final_update)
            finally:
                console.task_end(completed=True)
                worker.join(timeout=1)
                self.assertFalse(worker.is_alive())

    def test_long_live_preview_keeps_the_latest_text_visible(self):
        console = Console()
        console.interactive = True
        console.progress(
            {"phase": "response_delta", "content": "earlier line\n" * 1000 + "最新文字"}
        )
        output = io.StringIO()
        with (
            patch("sys.stdout", output),
            patch.object(console, "terminal_width", return_value=32),
            patch("ui.shutil.get_terminal_size", return_value=os.terminal_size((32, 12))),
        ):
            console._rich().print(console._live_renderable())

        self.assertIn("最新文字", output.getvalue())
        self.assertLessEqual(len(output.getvalue().splitlines()), 12)

    def test_dense_live_dashboard_leaves_room_for_the_latest_answer_and_metrics(self):
        for width, height in ((80, 24), (32, 12), (24, 8)):
            with self.subTest(width=width, height=height):
                console = Console()
                console.interactive = True
                console._task_title = "界面优化"
                console._task_activity = "正在接收 · Esc 暂停"
                console._todos = [
                    {
                        "step": f"步骤 {index}",
                        "status": "in_progress" if index == 19 else "completed",
                    }
                    for index in range(20)
                ]
                console._agents = {
                    f"A{index}": {
                        "agent_id": f"A{index}",
                        "title": f"代理 {index}",
                        "state": "running" if index == 19 else "completed",
                        "activity": "model",
                    }
                    for index in range(20)
                }
                console._context = {
                    "tokens": 12000,
                    "window_tokens": 131072,
                    "budget_tokens": 114688,
                    "source": "estimate",
                }
                console.progress({"phase": "response_delta", "reasoning": "思考\n" * 100})
                console.progress(
                    {"phase": "response_delta", "content": "前文\n" * 100 + "最新结果"}
                )
                output = io.StringIO()
                with (
                    patch("sys.stdout", output),
                    patch.object(console, "terminal_width", return_value=width),
                    patch(
                        "ui.shutil.get_terminal_size",
                        return_value=os.terminal_size((width, height)),
                    ),
                ):
                    console._rich().print(console._live_renderable())
                text = output.getvalue()
                self.assertIn("最新结果", text)
                self.assertIn("TPS", text)
                self.assertIn("CTX" if width < 60 else "上下文", text)
                self.assertLessEqual(len(text.splitlines()), height - 1)
                self.assertTrue(all(_display_width(line) <= width for line in text.splitlines()))
                if width == 80:
                    self.assertIn("步骤 19", text)
                    self.assertIn("A19", text)

    def test_compact_layout_does_not_remove_steps_from_full_task_details(self):
        console = Console()
        console._todos = [{"step": f"步骤 {index}", "status": "pending"} for index in range(20)]
        console._dashboard(max_height=3)
        output = io.StringIO()
        with patch("sys.stdout", output):
            console.tasks()
        self.assertIn("步骤 0", output.getvalue())
        self.assertIn("步骤 19", output.getvalue())
        self.assertEqual(len(console._todos), 20)

    def test_accepted_reasoning_and_answer_are_published_in_one_render(self):
        console = Console()
        console.reasoning("思考结果")
        console.stream("最终回答")
        rich_console = console._rich()
        with (
            patch.object(console, "_rich", return_value=rich_console),
            patch.object(rich_console, "print") as printed,
        ):
            console.assistant_end()
        printed.assert_called_once()
        text = "".join(segment.text for segment in rich_console.render(printed.call_args.args[0]))
        self.assertIn("思考结果", text)
        self.assertIn("最终回答", text)

    def test_history_includes_reasoning_and_tool_names(self):
        output = io.StringIO()
        messages = [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "需要先读取配置。",
                "tool_calls": [{"function": {"name": "read"}}],
            }
        ]
        with patch("sys.stdout", output):
            Console().history(messages)

        rendered = output.getvalue()
        self.assertIn("思考过程", rendered)
        self.assertIn("需要先读取配置", rendered)
        self.assertIn("工具调用  read", rendered)

    def test_reasoning_compatibility_fields_are_not_duplicated(self):
        self.assertEqual(
            _reasoning_text({"reasoning_content": "same", "reasoning": "same"}), "same"
        )

    def test_delegate_tool_detail_shows_task_count_not_payload(self):
        self.assertEqual(
            Console._tool_detail("delegate", {"tasks": [{"task": "one"}, {"task": "two"}]}),
            "2 个任务",
        )

    def test_task_dashboard_shows_title_todos_and_current_subagent_action(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console()
            console.task_start("修复前端状态")
            console.task_update(
                {
                    "phase": "plan",
                    "title": "前端状态展示",
                    "steps": [
                        {"step": "检查现有界面", "status": "completed"},
                        {"step": "实现展示", "status": "in_progress"},
                        {"step": "运行测试", "status": "pending"},
                    ],
                }
            )
            console.task_update(
                {
                    "phase": "subagent",
                    "agent_id": "A1",
                    "title": "检查界面",
                    "state": "running",
                    "mode": "inspect",
                    "activity": "tool",
                    "tool": "read",
                    "arguments": {"path": "ui.py"},
                }
            )
            console.tasks()

        rendered = output.getvalue()
        for expected in (
            "前端状态展示",
            "Todo · 1/3",
            "◉ 实现展示",
            "✓ 检查现有界面",
            "○ 运行测试",
            "A1",
            "运行中",
            "检查界面",
            "读取 · ui.py",
        ):
            self.assertIn(expected, rendered)
        self.assertNotIn("\033[", rendered)

    def test_subagent_heartbeat_is_not_repeated_in_redirected_output(self):
        output = io.StringIO()
        event = {
            "phase": "subagent",
            "agent_id": "A1",
            "title": "调查接口",
            "state": "running",
            "mode": "inspect",
            "activity": "model",
            "model_phase": "connecting",
        }
        with patch("sys.stdout", output):
            console = Console()
            for index in range(25):
                console.task_update({**event, "model_requests": index})
        self.assertEqual(output.getvalue().count("子代理 A1"), 1)

    def test_subagent_failure_displays_the_reason(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console()
            console.task_update(
                {
                    "phase": "subagent",
                    "agent_id": "A2",
                    "title": "检查测试",
                    "state": "error",
                    "activity": "finished",
                    "error": {"message": "model unavailable"},
                }
            )
            console.tasks()
        self.assertIn("失败", output.getvalue())
        self.assertIn("model unavailable", output.getvalue())

    def test_pause_cancels_live_agents_without_completing_todos(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console()
            console.task_start("检查任务")
            console.task_update(
                {
                    "phase": "plan",
                    "title": "检查任务",
                    "steps": [{"step": "检查文件", "status": "in_progress"}],
                }
            )
            console.task_update(
                {
                    "phase": "subagent",
                    "agent_id": "A1",
                    "title": "读取文件",
                    "state": "running",
                    "activity": "model",
                }
            )
            console.task_end(False)
        self.assertEqual(console._task_state, "paused")
        self.assertEqual(console._agents["A1"]["state"], "cancelled")
        self.assertEqual(console._todos[0]["status"], "in_progress")
        self.assertIn("已暂停", output.getvalue())

    def test_failed_task_is_not_presented_as_paused(self):
        with patch("sys.stdout", io.StringIO()):
            console = Console()
            console.task_start("任务")
            console.task_end(False, failed=True)
        self.assertEqual(console._task_state, "error")

    def test_interactive_dashboard_replaces_the_single_line_status(self):
        output = io.StringIO()
        with patch("sys.stdout", output), patch("ui.TaskLive") as live:
            console = Console()
            console.interactive = True
            console.task_start("交互任务")
            console.assistant_start()
            console.progress({"phase": "receiving", "received_bytes": 1024})
            console.task_end(True)
        live.return_value.start.assert_called_once_with(refresh=True)
        live.return_value.stop.assert_called_once()
        self.assertTrue(live.return_value.refresh.called)
        self.assertNotIn("\r\033[2K", output.getvalue())

    def test_live_dashboard_is_suspended_during_user_input(self):
        console = Console()
        live = Mock()
        console._live = live
        with patch.object(console, "_read_prompt", return_value="2"):
            self.assertEqual(console._read_input("选择"), "2")
        live.stop.assert_called_once()
        live.start.assert_called_once_with(refresh=True)
        self.assertIs(console._live, live)

    def test_selection_suspends_live_and_restores_it_when_cancelled(self):
        console = Console()
        console.interactive = True
        live = Mock()
        console._live = live

        def cancel_menu(*args):
            self.assertIsNone(console._live)
            return None

        with patch.object(console, "_select_prompt", side_effect=cancel_menu):
            self.assertIsNone(console.select_option("请选择", [{"label": "A"}]))

        live.stop.assert_called_once()
        live.start.assert_called_once_with(refresh=True)
        self.assertIs(console._live, live)

    def test_selection_interruption_restores_the_live_dashboard(self):
        console = Console()
        console.interactive = True
        live = Mock()
        console._live = live
        with patch.object(console, "_select_prompt", side_effect=KeyboardInterrupt):
            self.assertIsNone(console.select_option("请选择", [{"label": "A"}]))

        live.stop.assert_called_once()
        live.start.assert_called_once_with(refresh=True)
        self.assertIs(console._live, live)

    def test_selection_rejects_empty_options_and_invalid_defaults(self):
        console = Console()
        for options, default in [([], 0), ([{"label": "A"}], -1), ([{"label": "A"}], 1)]:
            with self.subTest(options=options, default=default), self.assertRaises(ValueError):
                console.select_option("请选择", options, default_index=default)

    def test_custom_answer_editor_does_not_replace_the_chat_input_history(self):
        console = Console()
        console.interactive = True
        original_session = Mock()
        console._prompt_session = original_session
        with patch("ui.PromptSession") as session:
            session.return_value.prompt.return_value = "我的答案"
            self.assertEqual(console._read_prompt("回答", isolated=True), "我的答案")

        self.assertIs(console._prompt_session, original_session)
        original_session.prompt.assert_not_called()
        session.return_value.prompt.assert_called_once_with([("class:prompt.marker", "回答")])
        self.assertIsNone(session.call_args.kwargs["completer"])
        self.assertFalse(session.call_args.kwargs["complete_while_typing"]())

    def test_chat_editor_enables_command_completion_while_typing(self):
        console = Console()
        console.interactive = True
        with patch("ui.PromptSession") as session:
            session.return_value.prompt.return_value = "/effort"
            self.assertEqual(console._read_prompt("问题"), "/effort")

        self.assertIsInstance(session.call_args.kwargs["completer"], _CommandCompleter)
        with patch("ui.get_app") as app:
            for text, expected in (("/eff", True), ("plain text", False), ("/effort max", False)):
                with self.subTest(text=text):
                    app.return_value.current_buffer.document = Document(text)
                    self.assertEqual(session.call_args.kwargs["complete_while_typing"](), expected)

    def test_existing_chat_editor_uses_the_current_skill_provider(self):
        console = Console()
        console.interactive = True
        console.skill_provider = lambda: {"alpha": "Alpha."}
        with patch("ui.PromptSession") as session:
            session.return_value.prompt.return_value = "/skill:"
            console._read_prompt("问题")
            completer = session.call_args.kwargs["completer"]
            document = Document("/skill:")
            self.assertEqual(
                [item.text for item in completer.get_completions(document, CompleteEvent())],
                ["/skill:alpha"],
            )

            console.skill_provider = lambda: {"beta": "Beta."}
            console._read_prompt("问题")
            self.assertEqual(session.call_count, 1)
            self.assertEqual(
                [item.text for item in completer.get_completions(document, CompleteEvent())],
                ["/skill:beta"],
            )

    def test_noninteractive_command_input_keeps_the_literal_prefix(self):
        console = Console()
        console.interactive = False
        with patch("builtins.input", return_value="/eff"), patch("ui.PromptSession") as session:
            self.assertEqual(console.prompt(), "/eff")

        session.assert_not_called()

    def test_read_value_uses_an_isolated_editor_and_restores_live_on_cancel(self):
        console = Console()
        live = Mock()
        console._live = live
        with patch.object(console, "_read_prompt", side_effect=KeyboardInterrupt) as reader:
            self.assertIsNone(console.read_value("数量"))

        reader.assert_called_once_with("数量", isolated=True)
        live.stop.assert_called_once()
        live.start.assert_called_once_with(refresh=True)
        self.assertIs(console._live, live)

    def test_read_value_accepts_text_and_treats_eof_as_cancellation(self):
        console = Console()
        with patch.object(console, "_read_prompt", side_effect=[" 37 ", EOFError]):
            self.assertEqual(console.read_value("数量"), " 37 ")
            self.assertIsNone(console.read_value("数量"))

    def test_reset_session_clears_screen_scrollback_and_all_ui_state(self):
        output = io.StringIO()
        console = Console()
        console.interactive = True
        console._task_title = "旧任务"
        console._todos = [{"step": "旧步骤", "status": "pending"}]
        console._agents = {"A1": {"title": "旧子代理"}}
        console._response_content = ["旧回复"]
        console._response_reasoning = ["旧思考"]
        console._prompt_session = Mock()
        console._last_phase = "receiving"
        console._last_retry_attempt = 3
        with patch("sys.stdout", output):
            console.reset_session(Path("session.json"))
        self.assertIn("\033[2J\033[H\033[3J", output.getvalue())
        self.assertIn("新会话", output.getvalue())
        self.assertNotIn("旧任务", output.getvalue())
        self.assertEqual(console._task_title, "")
        self.assertEqual(console._todos, [])
        self.assertEqual(console._agents, {})
        self.assertEqual(console._response_content, [])
        self.assertEqual(console._response_reasoning, [])
        self.assertIsNone(console._prompt_session)
        self.assertEqual(console._last_phase, "")
        self.assertIsNone(console._last_retry_attempt)

    def test_redirected_reset_does_not_emit_terminal_control_sequences(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            Console().reset_session(Path("session.json"))
        self.assertIn("新会话", output.getvalue())
        self.assertNotIn("\033[", output.getvalue())

    def test_narrow_dashboard_respects_terminal_width(self):
        output = io.StringIO()
        console = Console()
        console._task_title = "很长的中文任务标题" * 5
        console._todos = [{"step": "很长的中文步骤" * 5, "status": "in_progress"}]
        console._agents = {
            "A1": {
                "agent_id": "A1",
                "title": "很长的子代理标题" * 5,
                "state": "running",
                "activity": "tool",
                "tool": "read",
                "arguments": {"path": "/very/long/path/to/source.py"},
            }
        }
        with patch("sys.stdout", output), patch.object(console, "terminal_width", return_value=24):
            console.tasks()
        self.assertTrue(all(_display_width(line) <= 24 for line in output.getvalue().splitlines()))

    def test_live_dashboard_tracks_terminal_resizing(self):
        console = Console()
        console._live = Mock()
        with patch.object(console, "terminal_width", return_value=30):
            console._dashboard()
        self.assertEqual(console._live.console.width, 30)

    def test_compact_dashboard_keeps_current_step_and_active_agents_visible(self):
        output = io.StringIO()
        console = Console()
        console._todos = [
            {"step": f"步骤 {index}", "status": "in_progress" if index == 9 else "completed"}
            for index in range(10)
        ]
        console._agents = {
            f"A{index}": {
                "agent_id": f"A{index}",
                "title": f"代理 {index}",
                "activity": "finished",
                "state": "running" if index == 9 else "completed",
            }
            for index in range(10)
        }
        with patch("sys.stdout", output):
            console._rich().print(console._dashboard())
        rendered = output.getvalue()
        self.assertIn("◉ 步骤 9", rendered)
        self.assertIn("代理 9", rendered)
        self.assertIn("/tasks", rendered)

    def test_compact_dashboard_shows_the_next_step_and_prioritizes_failed_agents(self):
        output = io.StringIO()
        console = Console()
        console._todos = [
            {
                "step": f"步骤 {index}",
                "status": "in_progress" if index == 5 else "pending",
            }
            for index in range(10)
        ]
        console._agents = {
            f"A{index}": {
                "agent_id": f"A{index}",
                "title": f"代理 {index}",
                "state": "error" if index == 9 else "completed",
                "error": {"message": "需要检查连接"} if index == 9 else None,
            }
            for index in range(10)
        }
        with (
            patch("sys.stdout", output),
            patch("ui.shutil.get_terminal_size", return_value=os.terminal_size((80, 24))),
        ):
            console._rich().print(console._dashboard())
        rendered = output.getvalue()
        self.assertIn("◉ 步骤 5", rendered)
        self.assertIn("○ 步骤 6", rendered)
        self.assertIn("A9", rendered)
        self.assertIn("需要检查连接", rendered)
        self.assertLess(rendered.index("A9"), rendered.index("A0"))

    def test_streaming_activity_tracks_thinking_and_answering_despite_network_heartbeats(self):
        console = Console()
        console.interactive = True
        console._live = Mock()
        console.progress({"phase": "response_delta", "reasoning": "检查界面"})
        self.assertIn("正在思考", console._task_activity)
        console.progress({"phase": "receiving", "elapsed_seconds": 3, "received_bytes": 1024})
        self.assertIn("正在思考", console._task_activity)
        console.progress({"phase": "response_delta", "content": "已经修复"})
        self.assertIn("正在回答", console._task_activity)
        console.progress({"phase": "receiving", "elapsed_seconds": 5, "received_bytes": 2048})
        self.assertIn("正在回答", console._task_activity)
        console.progress({"phase": "response_reset"})
        self.assertIn("正在重新生成", console._task_activity)
        self.assertFalse(console._preview_content)

    def test_context_warning_does_not_color_the_speed_as_a_warning(self):
        console = Console()
        console._context = {
            "tokens": 1000000,
            "window_tokens": 1000000,
            "budget_tokens": 983616,
            "source": "usage",
        }
        rich = console._rich()
        lines = rich.render_lines(console._dashboard(), rich.options, pad=False)
        context_segments = [segment for segment in lines[-1] if "上下文" in segment.text]
        speed_segments = [segment for segment in lines[-1] if "TPS" in segment.text]
        self.assertTrue(any(segment.style.color.name == "red" for segment in context_segments))
        self.assertTrue(speed_segments)
        self.assertTrue(all(segment.style.color is None for segment in speed_segments))

    def test_restored_plan_does_not_appear_to_be_running(self):
        output = io.StringIO()
        plan = {"steps": [{"step": "尚未完成", "status": "in_progress"}]}
        with patch("sys.stdout", output):
            console = Console()
            console.restore_task("恢复任务", plan, completed=False)
        self.assertEqual(console._task_state, "restored")
        self.assertIsNone(console._task_started)
        self.assertIn("已恢复", output.getvalue())
        self.assertIn("尚未完成", output.getvalue())

    def test_simple_task_is_a_small_strip_without_empty_sections(self):
        output = io.StringIO()
        console = Console()
        console._task_title = "很长的任务标题" * 30
        with patch("sys.stdout", output):
            console._rich().print(console._dashboard())
        rendered = output.getvalue()
        self.assertNotIn("子代理", rendered)
        self.assertNotIn("Todo", rendered)
        self.assertNotIn("暂无", rendered)
        self.assertNotIn("╭", rendered)
        self.assertLessEqual(len(rendered.splitlines()), 2)
        self.assertIn("…", rendered)

    def test_context_and_tps_are_shown_without_counting_first_packet_wait(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console()
            console._task_title = "任务"
            console.update_context(
                {
                    "tokens": 12000,
                    "window_tokens": 1000000,
                    "budget_tokens": 983616,
                    "source": "usage",
                }
            )
            console.progress(
                {
                    "phase": "generation",
                    "request_id": "one",
                    "tokens": 200,
                    "generation_seconds": 4,
                    "excluded_wait_seconds": 30,
                    "tps": 50,
                    "source": "usage",
                    "finished": True,
                }
            )
            console.tasks()
            console.status(Path("session.json"), 0)
        rendered = output.getvalue()
        self.assertIn("12k/1M tokens", rendered)
        self.assertIn("TPS 50.0", rendered)
        self.assertNotIn("首包等待", rendered)
        self.assertNotIn("本地估算", rendered)

    def test_task_tps_is_weighted_by_output_time_not_averaged_per_request(self):
        console = Console()
        for request_id, tokens, duration in (("one", 100, 1), ("two", 100, 9)):
            console.progress(
                {
                    "phase": "generation",
                    "request_id": request_id,
                    "tokens": tokens,
                    "generation_seconds": duration,
                    "tps": tokens / duration,
                    "source": "usage",
                    "finished": True,
                }
            )
        self.assertEqual(console._tps_label(), "TPS 20.0")

    def test_speed_displays_the_value_without_explanatory_prefixes(self):
        console = Console()
        self.assertEqual(console._tps_label(), "TPS —")
        console.progress(
            {
                "phase": "generation",
                "request_id": "one",
                "tokens": 100,
                "generation_seconds": 2,
                "tps": 50,
                "source": "estimate",
            }
        )
        self.assertEqual(console._tps_label(), "TPS 50.0")

    def test_retry_generations_are_not_overwritten_by_the_shared_request_id(self):
        console = Console()
        for generation_id, tokens, duration in (("first", 100, 1), ("second", 100, 9)):
            console.progress(
                {
                    "phase": "generation",
                    "request_id": "same-request",
                    "generation_id": generation_id,
                    "tokens": tokens,
                    "generation_seconds": duration,
                    "tps": tokens / duration,
                    "source": "usage",
                    "finished": True,
                }
            )
        self.assertEqual(len(console._performance), 2)
        self.assertEqual(console._tps_label(), "TPS 20.0")

    def test_mcp_summary_uses_live_connections_and_excludes_disabled_services(self):
        console = Console()
        servers = [
            {"name": "playwright", "status": "connected", "tool_count": 25},
            {"name": "js-reverse", "status": "failed", "tool_count": 0},
            {"name": "disabled", "status": "disabled", "tool_count": 0},
        ]
        console.mcp_provider = lambda: servers
        self.assertIn("MCP 已连接 1/2 · 1 失败", console._toolbar())
        servers[1]["status"] = "connected"
        self.assertIn("MCP 已连接 2/2", console._toolbar())

    def test_narrow_live_strip_keeps_context_and_tps_visible(self):
        output = io.StringIO()
        console = Console()
        console._task_title = "窄屏中文任务" * 10
        console.update_context(
            {
                "tokens": 12000,
                "window_tokens": 131072,
                "budget_tokens": 114688,
                "source": "estimate",
            }
        )
        with patch("sys.stdout", output), patch.object(console, "terminal_width", return_value=24):
            console._rich().print(console._dashboard())
            toolbar = console._toolbar()
        self.assertIn("CTX", output.getvalue())
        self.assertIn("TPS", output.getvalue())
        self.assertTrue(all(_display_width(line) <= 24 for line in output.getvalue().splitlines()))
        self.assertIn("CTX", toolbar)
        self.assertIn("TPS", toolbar)
        self.assertIn("环境", toolbar)

    def test_reset_discards_previous_performance_and_context(self):
        console = Console()
        console._context = {"tokens": 999999}
        console._performance = {"one": {"tps": 100}}
        with patch("sys.stdout", io.StringIO()):
            console.reset_session(Path("session.json"))
        self.assertIsNone(console._context)
        self.assertEqual(console._tps_label(), "TPS —")

    def test_mcp_tool_detail_shows_action_and_target(self):
        self.assertEqual(
            Console._tool_detail(
                "mcp", {"action": "call", "tool": "github/search", "arguments": {"q": "x"}}
            ),
            "call · github/search",
        )

    def test_tool_result_closes_the_call_without_dumping_payload(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console()
            console.tool("read", {"path": "ui.py"})
            console.tool_result(
                "read",
                {"status": "ok", "data": {"content": "private and very long payload"}},
            )

        rendered = output.getvalue()
        self.assertIn("✓ 读取  ui.py", rendered)
        self.assertIn("#1", rendered)
        self.assertEqual(len(rendered.splitlines()), 1)
        self.assertNotIn("private and very long payload", rendered)

    def test_tool_result_surfaces_failure_reason(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            Console().tool_result(
                "exec",
                {
                    "status": "error",
                    "error": {"code": "nonzero_exit", "message": "command exited with 2"},
                },
            )

        rendered = output.getvalue()
        self.assertIn("失败", rendered)
        self.assertIn("command exited with 2", rendered)

    def test_user_can_select_a_described_option(self):
        output = io.StringIO()
        options = [
            {"label": "快速模式", "description": "减少检查步骤，优先尽快交付。"},
            {"label": "稳妥模式", "description": "执行完整检查，耗时会更长。"},
        ]
        with patch("sys.stdout", output), patch("builtins.input", return_value="2"):
            result = Console().ask_user("你希望采用哪种执行方式？", options)

        rendered = output.getvalue()
        self.assertIn("你希望采用哪种执行方式", rendered)
        self.assertIn("减少检查步骤", rendered)
        self.assertIn("执行完整检查", rendered)
        self.assertIn("其他（自行输入）", rendered)
        self.assertEqual(
            result,
            {
                "answer": "稳妥模式",
                "selected_index": 2,
                "selected_label": "稳妥模式",
                "custom": False,
            },
        )

    def test_interactive_question_uses_the_menu_without_requesting_a_number(self):
        console = Console()
        console.interactive = True
        options = [{"label": "A", "description": "方案 A"}, {"label": "B", "description": "方案 B"}]
        with (
            patch.object(console, "_select_prompt", return_value=1) as menu,
            patch("builtins.input") as numeric_input,
            patch("sys.stdout", io.StringIO()),
        ):
            result = console.ask_user("请选择方案", options)

        numeric_input.assert_not_called()
        self.assertEqual(menu.call_args.args[1][-1]["label"], "其他（自行输入）")
        self.assertEqual(result["selected_index"], 2)
        self.assertEqual(result["answer"], "B")
        self.assertFalse(result["custom"])

    def test_cancelled_question_pauses_the_task_and_restores_live_once(self):
        console = Console()
        console.interactive = True
        live = Mock()
        console._live = live
        with (
            patch.object(console, "_select_prompt", return_value=None),
            self.assertRaises(KeyboardInterrupt),
        ):
            console.ask_user("请选择方案", [{"label": "A"}, {"label": "B"}])

        live.stop.assert_called_once()
        live.start.assert_called_once_with(refresh=True)
        self.assertIs(console._live, live)

    def test_redirected_selection_rejects_invalid_numbers_then_accepts_one(self):
        output = io.StringIO()
        with (
            patch("sys.stdout", output),
            patch("builtins.input", side_effect=["0", "3", "text", "2"]),
        ):
            result = Console().select_option("请选择方案", [{"label": "A"}, {"label": "B"}])

        self.assertEqual(result, 1)
        self.assertEqual(output.getvalue().count("请输入 1 到 2"), 3)

    def test_last_user_option_accepts_custom_input(self):
        output = io.StringIO()
        options = [
            {"label": "A", "description": "选择方案 A。"},
            {"label": "B", "description": "选择方案 B。"},
        ]
        with patch("sys.stdout", output), patch("builtins.input", side_effect=["3", "我的方案"]):
            result = Console().ask_user("请选择方案", options)

        self.assertEqual(result["answer"], "我的方案")
        self.assertEqual(result["selected_index"], 3)
        self.assertTrue(result["custom"])

    def test_end_of_input_cancels_an_active_question(self):
        options = [
            {"label": "A", "description": "选择方案 A。"},
            {"label": "B", "description": "选择方案 B。"},
        ]
        with (
            patch("sys.stdout", io.StringIO()),
            patch("builtins.input", side_effect=EOFError),
            self.assertRaises(KeyboardInterrupt),
        ):
            Console().ask_user("请选择方案", options)


@unittest.skipUnless(os.name == "posix", "requires select on pipe descriptors")
class TaskInputMonitorTests(unittest.TestCase):
    def setUp(self):
        self.read_fd, self.write_fd = os.pipe()
        self.addCleanup(os.close, self.read_fd)
        self.addCleanup(os.close, self.write_fd)
        self.input = Mock()
        self.input.fileno.return_value = self.read_fd
        self.input.closed = False
        self.input.raw_mode.return_value = nullcontext()
        self.input.flush_keys.return_value = []
        self.monitor = _TaskInputMonitor()
        input_patch = patch("ui.create_input", return_value=self.input)
        input_patch.start()
        self.addCleanup(input_patch.stop)
        self.addCleanup(self.monitor.close)

    def test_idle_keyboard_wait_blocks_until_input_or_stop(self):
        entered = threading.Event()
        select_keys = select.select

        def watch(*args):
            entered.set()
            return select_keys(*args)

        with patch("ui.select.select", side_effect=watch) as selected:
            self.monitor.start()
            thread = self.monitor._thread
            self.assertTrue(entered.wait(1))
            self.assertIsNone(selected.call_args.args[3])
            self.monitor.stop()
        self.assertFalse(thread.is_alive())
        selected.assert_called_once()
        self.input.flush_keys.assert_not_called()

    def test_standalone_escape_is_flushed_once_and_interrupts(self):
        interrupted = threading.Event()
        self.input.read_keys.side_effect = lambda: (os.read(self.read_fd, 1024), [])[1]
        self.input.flush_keys.return_value = [KeyPress(Keys.Escape, "\x1b")]
        with patch("ui.os.kill", side_effect=lambda *_: interrupted.set()) as interrupt:
            self.monitor.start()
            os.write(self.write_fd, b"\x1b")
            self.assertTrue(interrupted.wait(1))
            self.monitor.stop()
        interrupt.assert_called_once_with(os.getpid(), signal.SIGINT)
        self.input.flush_keys.assert_called_once()

    def test_navigation_key_does_not_interrupt_and_returns_to_blocking_wait(self):
        flushed = threading.Event()
        self.input.read_keys.side_effect = lambda: (
            os.read(self.read_fd, 1024),
            [KeyPress(Keys.Up, "\x1b[A")],
        )[1]
        self.input.flush_keys.side_effect = lambda: (flushed.set(), [])[1]
        with patch("ui.os.kill") as interrupt:
            self.monitor.start()
            os.write(self.write_fd, b"\x1b[A")
            self.assertTrue(flushed.wait(1))
            self.monitor.stop()
        interrupt.assert_not_called()
        self.input.flush_keys.assert_called_once()

    def test_failed_input_cleanup_still_closes_wakeup_descriptors(self):
        self.monitor.start()
        descriptors = self.monitor._wake_pipe
        self.input.close.side_effect = OSError("input cleanup failed")
        with self.assertRaisesRegex(OSError, "input cleanup failed"):
            self.monitor.stop()
        for descriptor in descriptors:
            with self.assertRaises(OSError):
                os.fstat(descriptor)


if __name__ == "__main__":
    unittest.main()


class TeamBackgroundDisplayTests(unittest.TestCase):
    def test_toolbar_shows_background_work_then_unread_results(self):
        console = Console()
        console.update_team(
            {
                "phase": "team",
                "background": "running",
                "working": ["engineer"],
                "queued": ["critic"],
            }
        )
        self.assertIn("团队后台 2 人工作中", console._toolbar())
        console.update_team({"phase": "team", "background": "finished", "unread": 2})
        self.assertIn("团队 2 条新结果 /env results", console._toolbar())
        console.update_team(None)
        self.assertNotIn("团队", console._toolbar())

    def test_notices_print_from_any_thread(self):
        output = io.StringIO()
        console = Console()
        console.colors = False
        with patch("sys.stdout", output):
            worker = threading.Thread(
                target=console.task_update,
                args=({"phase": "team", "background": "finished", "notice": "团队后台工作已完成"},),
            )
            worker.start()
            worker.join()
        self.assertIn("◆ 团队后台工作已完成", output.getvalue())

    def test_task_end_keeps_background_members_running(self):
        output = io.StringIO()
        console = Console()
        console.interactive = False
        console.colors = False
        with patch("sys.stdout", output):
            console.task_start("build")
            console.task_update(
                {
                    "phase": "subagent",
                    "agent_id": "engineer",
                    "title": "t1 Build",
                    "mode": "team",
                    "state": "running",
                    "activity": "model",
                }
            )
            console.task_update(
                {"phase": "team", "background": "running", "working": ["engineer"], "queued": []}
            )
            console.task_end(True)
        self.assertEqual(console._agents["engineer"]["state"], "background")
        self.assertIn("后台继续", output.getvalue())
        self.assertIn("团队成员在后台继续工作（engineer）", output.getvalue())
        self.assertNotIn("已取消", output.getvalue())


class ReadableOutputTests(unittest.TestCase):
    def console(self, *args, **kwargs):
        console = Console(*args, **kwargs)
        # The same output whether or not the test runner is attached to a terminal.
        console.interactive = False
        console.colors = False
        return console

    def render(self, action, width=100):
        output = io.StringIO()
        with (
            patch("sys.stdout", output),
            patch.object(Console, "terminal_width", return_value=width),
        ):
            action()
        return output.getvalue()

    def test_multiline_messages_keep_their_indentation(self):
        console = self.console()
        rendered = self.render(lambda: console.info("第一行\n第二行"))
        self.assertEqual(rendered.splitlines(), ["  • 第一行", "    第二行"])
        errors = io.StringIO()
        with patch("sys.stderr", errors):
            self.render(lambda: console.error(ValueError("用法：a\nb")))
        self.assertEqual(errors.getvalue().splitlines(), ["", "  ✗ 用法：a", "    b"])

    def test_interactive_messages_wrap_under_their_text(self):
        console = self.console()
        console.interactive = True
        rendered = self.render(lambda: console.info("word " * 30), width=40)
        lines = rendered.splitlines()
        self.assertGreater(len(lines), 1)
        self.assertTrue(lines[0].startswith("  • word"))
        self.assertTrue(all(line.startswith("    word") for line in lines[1:]))
        self.assertTrue(all(_display_width(line) <= 39 for line in lines))

    def test_word_wrapping_breaks_at_spaces_and_handles_wide_text(self):
        self.assertEqual(_wrap_words("alpha beta gamma", 11), ["alpha beta", "gamma"])
        self.assertEqual(_wrap_words("中文中文中文", 4), ["中文", "中文", "中文"])
        self.assertEqual(_wrap_words("中", 1), ["中"])
        self.assertEqual(_wrap_words("abcd ", 4), ["abcd"])
        self.assertEqual(_wrap_words("", 10), [""])

    def test_input_history_survives_a_new_session(self):
        console = self.console()
        console.interactive = True
        with (
            patch("ui.PromptSession") as session,
            patch("ui.patch_stdout", return_value=nullcontext()),
        ):
            session.return_value.prompt.return_value = "hello"
            console._read_prompt("  You  › ")
            self.render(lambda: console.reset_session(Path("/tmp/session.json")))
            console._read_prompt("  You  › ")
            console._read_prompt("  API Key › ", isolated=True, secret=True)
        histories = [call.kwargs["history"] for call in session.call_args_list]
        self.assertEqual(len(histories), 3)
        self.assertIs(histories[0], histories[1])
        # Answers such as keys never join the message history.
        self.assertIsNot(histories[2], histories[0])

    def test_tasks_without_a_task_say_so_instead_of_an_empty_panel(self):
        console = self.console()
        rendered = self.render(console.tasks)
        self.assertIn("当前没有任务", rendered)
        self.assertNotIn("0.0 秒", rendered)

    def test_restored_tasks_show_no_made_up_duration(self):
        console = self.console()
        plan = {"steps": [{"step": "写测试", "status": "in_progress"}]}
        rendered = self.render(lambda: console.restore_task("恢复任务", plan, completed=False))
        self.assertIn("已恢复", rendered)
        self.assertNotIn("0.0 秒", rendered)
        # A finished question without a plan is already visible in the restored history.
        console = self.console()
        rendered = self.render(lambda: console.restore_task("问答", None, completed=True))
        self.assertEqual(rendered, "")
        self.assertEqual(console._task_state, "completed")

    def test_status_counts_records_without_claiming_a_restore(self):
        console = self.console()
        rendered = self.render(lambda: console.status(Path("/tmp/session.json"), 2))
        self.assertIn("记录      2 条", rendered)
        self.assertNotIn("已恢复", rendered)
        self.assertIn("空", self.render(lambda: console.status(Path("/tmp/session.json"), 0)))

    def test_long_paths_keep_their_root_and_final_directories(self):
        path = "/tmp/claude-1000/-home-opc-project/0123456789abcdef/scratchpad/workspace"
        shortened = _short_path(path, 40)
        self.assertTrue(shortened.startswith("/tmp/…/"))
        self.assertTrue(shortened.endswith("/scratchpad/workspace"))
        self.assertLessEqual(_display_width(shortened), 40)
        self.assertEqual(_short_path("/tmp/a", 40), "/tmp/a")

    def test_each_call_is_one_line_with_its_outcome_and_number(self):
        console = self.console(workspace=Path("/work"))

        def run():
            console.tool("read", {"path": "/work/src/app.py", "offset": 40})
            console.tool_result("read", {"status": "ok", "data": {"content": "a\nb\n"}})

        rendered = self.render(run)
        self.assertEqual(rendered.splitlines(), ["  ✓ 读取  src/app.py:40  2 行  #1"])

    def test_background_session_line_names_the_state_once_with_a_short_id(self):
        console = self.console()
        session_id = "cd3215374dd04cce86a2119ccc4df6d5"

        def run(state, status="ok"):
            console.tool("exec", {"command": "sleep 30", "background": True})
            console.tool_result(
                "exec", {"status": status, "data": {"session_id": session_id, "state": state}}
            )

        line = self.render(lambda: run("running"))
        self.assertIn("运行中 · 进程 cd321537", line)
        self.assertNotIn(session_id, line)
        self.assertNotIn("running", line)
        line = self.render(lambda: run("timed_out", "timeout"))
        self.assertIn("进程 cd321537 · 超时终止", line)

    def test_narrow_lines_keep_the_outcome_and_call_number(self):
        console = self.console()
        console._active_tool_number = 57
        result = {
            "status": "timeout",
            "data": {"session_id": "3f2a9c1b", "state": "running", "output": ""},
        }
        for width in (50, 80):
            with self.subTest(width=width):
                console._active_tool = (
                    "monitor",
                    {"action": "wait", "session_id": "3f2a9c1b", "wait_for": "Compiled ok"},
                )
                line = self.render(lambda: console.tool_result("monitor", result), width).split(
                    "\n"
                )[0]
                self.assertIn("超时", line)
                self.assertTrue(line.endswith("#57"), line)
                self.assertLessEqual(_display_width(line), width - 1)

    def test_truncated_results_point_to_the_retained_full_output(self):
        console = self.console()

        def run(truncated):
            console.tool("grep", {"pattern": "TODO", "path": "."})
            console.tool_result(
                "grep",
                {
                    "status": "ok",
                    "data": {"matches": [1, 2]},
                    "artifact_path": "/tmp/out.txt",
                    "truncated": truncated,
                },
            )

        self.assertIn("当前为部分结果  /tool 1 查看全文", self.render(lambda: run(True)))
        self.assertIn("2 条匹配  #2", self.render(lambda: run(False)))

    def test_failed_call_shows_the_reason_without_the_workspace_prefix(self):
        console = self.console(workspace=Path("/work"))

        def run():
            console.tool("read", {"path": "missing.toml"})
            console.tool_result(
                "read",
                {
                    "status": "error",
                    "error": {
                        "code": "FileNotFoundError",
                        "message": "No file: /work/missing.toml",
                    },
                },
            )

        lines = self.render(run).splitlines()
        self.assertIn("✗ 读取  missing.toml  失败", lines[0])
        self.assertEqual(lines[1].strip(), "╰ No file: missing.toml")
        self.assertEqual(console._tool_errors, 1)

    def test_edits_report_changed_lines_and_writes_report_their_size(self):
        console = self.console(workspace=Path("/work"))
        edit = {
            "path": "/work/ui.py",
            "edits": [{"oldText": "a\nb\nc\n", "newText": "a\nB\nc\nd\n"}],
        }

        def run():
            console.task_start("修改界面")
            console.tool("edit", edit)
            console.tool_result("edit", {"status": "ok", "data": {"path": "/work/ui.py"}})
            console.tool("write", {"path": "/work/new.py", "content": "x = 1\ny = 2\n"})
            console.tool_result("write", {"status": "ok", "data": {"path": "/work/new.py"}})
            console.task_end(True)

        rendered = self.render(run)
        self.assertIn("✓ 修改  ui.py  +2 -1  #1", rendered)
        self.assertIn("✓ 写入  new.py  2 行  #2", rendered)
        self.assertIn("✎ 修改 2 个文件  ui.py +2 -1 · new.py 写入 2 行", rendered)
        self.assertNotIn("new.py 写入 2 行 +", rendered)

    def test_large_repetitive_edits_are_counted_quickly(self):
        old = "\n".join("}" if index % 2 else "" for index in range(6000))
        new = "\n".join("{" if index % 3 else "" for index in range(6000))
        started = time.monotonic()
        added, removed = _edit_stats([{"oldText": old, "newText": new}])
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertGreater(added, 0)
        self.assertGreater(removed, 0)
        self.assertEqual(
            _edit_stats([{"oldText": "keep\nold\nkeep", "newText": "keep\nnew\nkeep"}]),
            (1, 1),
        )

    def test_plan_updates_show_progress_and_the_current_step(self):
        console = self.console()
        plan = {
            "title": "整理界面",
            "steps": [
                {"step": "检查现状", "status": "completed"},
                {"step": "改进输出", "status": "in_progress"},
                {"step": "验证", "status": "pending"},
            ],
        }

        def run():
            console.tool("update_plan", plan)
            console.tool_result("update_plan", {"status": "ok", "data": plan})

        self.assertEqual(self.render(run).splitlines(), ["  ☰ 计划  改进输出  1/3  #1"])

    def test_long_command_output_is_folded_with_a_pointer_to_the_full_result(self):
        console = self.console()

        def run():
            console.tool("exec", {"command": "make test"})
            console.tool_result(
                "exec",
                {
                    "status": "ok",
                    "data": {
                        "output": "\n".join(f"line {n}" for n in range(10)),
                        "exit_code": 0,
                    },
                    "artifact_path": "/tmp/out.txt",
                },
            )

        lines = self.render(run).splitlines()
        self.assertIn("✓ 运行  make test  exit 0 · 10 行输出  #1", lines[0])
        self.assertEqual(
            [line.strip() for line in lines[1:]],
            ["│ line 0", "│ … 省略 7 行 · /tool 1 查看全部", "│ line 8", "│ line 9"],
        )

    def cache_task(self, console, *, plan=False):
        totals = {"calls": 2, "prompt_tokens": 1000, "hit_tokens": 100, "hit_rate": 0.1}
        console.cache_provider = lambda: totals

        def run():
            console.task_start("问题")
            totals.update(prompt_tokens=2000, hit_tokens=975, hit_rate=0.4875)
            if plan:
                console.task_update(
                    {
                        "phase": "plan",
                        "title": "问题",
                        "steps": [{"step": "做", "status": "completed"}],
                    }
                )
            console.update_context(
                {"tokens": 12000, "window_tokens": 1000000, "budget_tokens": 983616}
            )
            console.progress(
                {
                    "phase": "generation",
                    "request_id": "one",
                    "tokens": 200,
                    "generation_seconds": 4,
                    "tps": 50,
                }
            )
            console.tool("ls", {"path": "."})
            console.tool_result("ls", {"status": "error", "error": {"message": "denied"}})
            console.task_end(True)

        return run

    def test_task_summary_reports_time_tools_output_speed_context_and_task_cache(self):
        console = self.console()
        summary = self.render(self.cache_task(console)).splitlines()[-1]
        for value in ("✓ 已完成", "工具 1", "异常 1", "输出 200", "TPS 50.0", "1.2%", "缓存 88%"):
            self.assertIn(value, summary)

    def test_planned_task_keeps_full_statistics_on_narrow_terminals(self):
        console = self.console()
        lines = self.render(self.cache_task(console, plan=True), width=60).splitlines()
        heading = lines.index(next(line for line in lines if "已完成" in line))
        self.assertIn("Todo · 1/1", lines[heading + 1])
        self.assertIn("缓存 88%", lines[-1])
        self.assertNotIn("已完成", lines[-1])
        self.assertTrue(all(_display_width(line) <= 59 for line in lines[heading:]))

    def test_toolbar_shows_model_effort_and_cache_then_drops_hints_when_narrow(self):
        console = self.console("provider/model-x", reasoning_effort="high")
        console.cache_provider = lambda: {"hit_rate": 0.5}
        console.update_context({"tokens": 900000, "window_tokens": 1000000, "budget_tokens": 1})
        with patch.object(Console, "terminal_width", return_value=160):
            wide = console._toolbar()
            styles = {style for style, _ in console._toolbar_fragments()}
        for value in ("model-x", "推理 high", "缓存 50%", "90.0%", "Ctrl-T"):
            self.assertIn(value, wide)
        self.assertNotIn("provider/", wide)
        self.assertIn("class:toolbar.error", styles)
        with patch.object(Console, "terminal_width", return_value=70):
            narrow = console._toolbar()
        self.assertNotIn("Ctrl-T", narrow)
        self.assertIn("环境", narrow)
        self.assertIn("TPS", narrow)
        self.assertLessEqual(_display_width(narrow), 69)

    def test_toolbar_shortens_long_labels_before_hiding_context(self):
        console = self.console()
        console.environment_name = "multi-agent-research-environment"
        console.update_team(
            {"phase": "team", "background": "running", "working": ["a", "b", "c", "d"]}
        )
        console.update_context({"tokens": 950000, "window_tokens": 1000000, "budget_tokens": 1})
        for width in (60, 70, 100):
            with (
                self.subTest(width=width),
                patch.object(Console, "terminal_width", return_value=width),
            ):
                toolbar = console._toolbar()
                self.assertIn("95", toolbar)
                self.assertIn("TPS", toolbar)
                self.assertFalse(toolbar.endswith(" · "))
                self.assertLessEqual(_display_width(toolbar), width - 1)

    def test_help_groups_commands_and_keeps_bracketed_arguments(self):
        console = self.console()
        sections = (
            ("环境", (("/env 名称 [keep] [--yes]", "切换环境"),)),
            ("快捷键", (("Ctrl-T", "轨迹"),)),
        )
        for width in (100, 40):
            with self.subTest(width=width):
                rendered = self.render(lambda: console.help(sections), width=width)
                self.assertIn("/env 名称 [keep] [--yes]", rendered)
                self.assertIn("快捷键", rendered)
                self.assertTrue(
                    all(_display_width(line) <= width for line in rendered.splitlines())
                )
        self.assertEqual(self.render(lambda: console.help("plain help")), "plain help\n")

    def test_status_is_grouped_and_includes_usage_and_cache(self):
        console = self.console("model-x")
        console.cache_provider = lambda: {
            "calls": 3,
            "prompt_tokens": 100,
            "hit_tokens": 50,
            "hit_rate": 0.5,
            "latest_hit_rate": 0.9,
        }
        console.update_context({"tokens": 12000, "window_tokens": 1000000, "budget_tokens": 983616})
        rendered = self.render(lambda: console.status(Path("/tmp/session.json"), 4))
        for value in ("运行", "能力", "用量", "会话", "12k / 1M tokens · 1.2%", "会话 50.0%"):
            self.assertIn(value, rendered)
        self.assertIn("最近一次 90.0%", rendered)
        self.assertIn("记录      4 条", rendered)

    def test_git_branch_is_read_from_head_including_worktree_pointers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "repo" / ".git").mkdir(parents=True)
            (root / "repo" / ".git" / "HEAD").write_text("ref: refs/heads/feature/ui\n")
            (root / "repo" / "src").mkdir()
            self.assertEqual(_git_branch(root / "repo" / "src"), "feature/ui")
            (root / "linked").mkdir()
            (root / "gitdir").mkdir()
            (root / "gitdir" / "HEAD").write_text("0123456789abcdef\n")
            (root / "linked" / ".git").write_text(f"gitdir: {root / 'gitdir'}\n")
            self.assertEqual(_git_branch(root / "linked"), "01234567")
            # A pruned worktree must not report the enclosing repository's branch.
            (root / "repo" / "pruned").mkdir()
            (root / "repo" / "pruned" / ".git").write_text("gitdir: ../missing/worktrees/x\n")
            self.assertIsNone(_git_branch(root / "repo" / "pruned"))

    def test_git_branch_of_a_reftable_repository_comes_from_git(self):
        with tempfile.TemporaryDirectory() as directory:
            created = subprocess.run(
                ["git", "init", "-q", "--ref-format=reftable", "-b", "trunk", directory],
                capture_output=True,
                check=False,
            )
            if created.returncode:
                self.skipTest("git without reftable support")
            self.assertEqual(_git_branch(Path(directory)), "trunk")
