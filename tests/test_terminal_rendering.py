import io
import threading
import unittest
from unittest.mock import Mock, patch

from rich.console import Console, Group
from rich.segment import Segment
from rich.text import Text

from terminal_rendering import StreamingDisplay, TaskLive


class TerminalRenderingTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.console = Console(
            file=self.output,
            force_terminal=True,
            color_system="standard",
            no_color=True,
            width=40,
            height=12,
        )
        self.live = TaskLive(
            Group(Text("回答标题"), Text("正在生成"), Text("任务状态")),
            console=self.console,
            refresh_per_second=12,
            auto_refresh=False,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        clock_patch = patch("terminal_rendering.time.monotonic", return_value=10.0)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.live.start(refresh=True)
        self.addCleanup(self.live.stop)
        self.clear_output()

    def clear_output(self):
        self.output.seek(0)
        self.output.truncate()

    def refresh(self, renderable=None, now=11.0):
        self.clock.return_value = now
        if renderable is not None:
            self.live.update(renderable)
        self.live.refresh()
        return self.output.getvalue()

    def test_unchanged_frame_writes_nothing(self):
        self.assertEqual(self.refresh(), "")

    def test_only_changed_row_is_repainted_without_clearing_the_area(self):
        output = self.refresh(Group(Text("回答标题"), Text("更多文字"), Text("任务状态")))
        self.assertIn("更多文字", output)
        self.assertNotIn("回答标题", output)
        self.assertNotIn("任务状态", output)
        self.assertNotIn("\x1b[2K", output)
        self.assertEqual(output.count("\x1b[0K"), 1)
        self.assertLess(output.index("更多文字"), output.index("\x1b[0K"))

    def test_all_refresh_sources_share_one_rate_limit_and_flush_latest_text(self):
        self.refresh(Text("第一段"), now=10.01)
        self.refresh(Text("第二段"), now=10.04)
        self.assertEqual(self.output.getvalue(), "")
        output = self.refresh(now=10.09)
        self.assertIn("第二段", output)
        self.assertNotIn("第一段", output)

    def test_shorter_row_erases_only_the_old_suffix(self):
        output = self.refresh(Group(Text("回答标题"), Text("短"), Text("任务状态")))
        self.assertIn("短\x1b[0K", output)
        self.assertNotIn("正在生成", output)

    def test_growing_frame_retains_existing_rows(self):
        output = self.refresh(
            Group(Text("回答标题"), Text("正在生成"), Text("任务状态"), Text("新增行"))
        )
        self.assertIn("新增行", output)
        self.assertNotIn("回答标题", output)
        self.assertEqual(self.live._live_render.last_render_height, 4)

    def test_shrinking_frame_erases_removed_rows_and_restores_cursor(self):
        output = self.refresh(Text("回答标题"))
        self.assertEqual(output.count("\x1b[0K"), 2)
        self.assertIn("\x1b[2A", output)
        self.assertEqual(self.live._live_render.last_render_height, 1)

    def test_style_changes_are_visible_even_when_text_is_unchanged(self):
        output = self.refresh(
            Group(Text("回答标题", style="bold"), Text("正在生成"), Text("任务状态"))
        )
        self.assertIn("\x1b[1m回答标题", output)
        self.assertNotIn("正在生成", output)

    def test_resize_reflows_unchanged_text(self):
        self.console.width = 30
        output = self.refresh()
        self.assertIn("回答标题", output)
        self.assertIn("正在生成", output)
        self.assertIn("任务状态", output)

    def test_resize_with_a_shorter_frame_clears_previous_rows(self):
        self.console.width = 30
        output = self.refresh(Text("短回答"))
        self.assertEqual(output.count("\x1b[2K"), 3)
        self.assertIn("短回答", output)
        self.assertEqual(self.live._live_render.last_render_height, 1)

    def test_last_column_is_reserved_to_prevent_terminal_auto_wrap(self):
        self.refresh(Text("长" * 40))
        lines = self.live._live_render.previous_lines
        self.assertTrue(all(Segment.get_line_length(line) < self.console.width for line in lines))

    def test_log_output_resets_the_area_and_restores_all_rows(self):
        self.console.print(Text("工具完成"))
        output = self.output.getvalue()
        self.assertIn("工具完成", output)
        self.assertIn("回答标题", output)
        self.assertIn("任务状态", output)
        self.assertIn("\x1b[2K", output)
        self.clear_output()
        self.assertEqual(self.refresh(), "")

    def test_publish_replaces_preview_and_writes_one_batch(self):
        self.live.update(Text("任务状态"))
        with patch.object(self.output, "write", wraps=self.output.write) as write:
            self.live.publish(Group(Text("思考结果"), Text("最终回答")))
        output = self.output.getvalue()
        self.assertEqual(write.call_count, 1)
        self.assertIn("思考结果", output)
        self.assertIn("最终回答", output)
        self.assertIn("任务状态", output)
        self.assertNotIn("正在生成", output)

    def test_stop_and_restart_redraws_after_input(self):
        self.live.stop()
        self.clear_output()
        self.live.start(refresh=True)
        self.assertIn("回答标题", self.output.getvalue())
        self.assertIn("任务状态", self.output.getvalue())

    def test_long_response_scrolls_into_history_while_it_is_still_streaming(self):
        body = [[Segment(f"回答第 {index:02d} 行")] for index in range(30)]
        display = StreamingDisplay(body, Text("任务状态"), key=1, height=12)
        first = self.refresh(display)
        for index in range(30):
            self.assertIn(f"回答第 {index:02d} 行", first)
        self.assertNotIn("\x1b[2K", first)
        self.clear_output()
        body = [*body, [Segment("回答第 30 行")]]
        updated = self.refresh(StreamingDisplay(body, Text("任务状态"), key=1, height=12), now=12)
        self.assertIn("回答第 30 行", updated)
        self.assertNotIn("回答第 00 行", updated)
        self.assertNotIn("\x1b[2K", updated)

    def test_finishing_keeps_streamed_text_in_place_without_reprinting_it(self):
        display = StreamingDisplay(
            [[Segment("第一段回答")], [Segment("第二段回答")]],
            Text("任务状态"),
            key=1,
            height=12,
        )
        self.refresh(display)
        self.clear_output()
        self.live.update(Text("任务状态"))
        self.live.finish_stream(display)
        output = self.output.getvalue()
        self.assertNotIn("第一段回答", output)
        self.assertNotIn("第二段回答", output)
        self.assertNotIn("\x1b[2K", output)
        self.assertEqual(self.live._live_render.last_render_height, 1)

    def test_resize_reflows_the_live_tail_without_repeating_older_history(self):
        source = Group(*(Text(f"row-{index:02d} " + "x" * 72) for index in range(40)))

        def display():
            body = self.console.render_lines(
                source, self.console.options.update(width=self.console.width - 1), pad=False
            )
            return StreamingDisplay(body, Text("任务状态"), key=1, height=12)

        self.refresh(display())
        self.clear_output()
        self.console.width = 24
        output = self.refresh(display(), now=12)
        self.assertNotIn("row-28", output)
        self.assertNotIn("row-29", output)
        self.assertIn("row-39", output)

    def test_idle_scheduler_checks_only_once_per_second(self):
        self.live._last_refresh = 10.0
        self.assertAlmostEqual(self.live._refresh_timeout(10.1), 0.9)
        self.live._refresh_pending = True
        self.assertEqual(self.live._refresh_timeout(10.1), 0)

    def test_expensive_frames_slow_down_instead_of_running_continuously(self):
        self.live.refresh_per_second = 20
        self.assertEqual(self.live._frame_interval(), 0.05)
        self.live._render_seconds = 0.25
        self.assertEqual(self.live._frame_interval(), 1.0)
        self.live._render_seconds = 10
        self.assertEqual(self.live._frame_interval(), 1.0)

    def test_burst_of_updates_wakes_scheduler_once_without_rendering(self):
        self.live._scheduled_refresh = True
        with (
            patch.object(self.live, "get_renderable") as build,
            patch.object(self.live._refresh_condition, "notify") as wake,
        ):
            for _ in range(100):
                self.live.refresh()
        wake.assert_called_once()
        build.assert_not_called()


class ScheduledRenderingTests(unittest.TestCase):
    def create_live(self, get_renderable, *, refresh=True, output=None):
        self.output = output if output is not None else io.StringIO()
        live = TaskLive(
            get_renderable=get_renderable,
            console=Console(file=self.output, force_terminal=True, width=40, height=12),
            refresh_per_second=20,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        live.start(refresh=refresh)
        worker = live._scheduler

        def cleanup():
            live.stop()
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())

        self.addCleanup(cleanup)
        return live

    def test_idle_display_does_not_rebuild_frames_at_streaming_frequency(self):
        called = threading.Event()
        build = Mock(side_effect=lambda: (called.set(), Text("waiting"))[1])
        self.create_live(build)
        called.clear()
        build.reset_mock()
        self.assertFalse(called.wait(0.2))
        build.assert_not_called()

    def test_start_without_initial_refresh_still_displays_a_frame(self):
        ready = threading.Event()

        class Output(io.StringIO):
            def write(self, value):
                result = super().write(value)
                if "initial" in value:
                    ready.set()
                return result

        self.create_live(lambda: Text("initial"), refresh=False, output=Output())
        self.assertTrue(ready.wait(1))
        self.assertIn("initial", self.output.getvalue())

    def test_updates_do_not_block_the_producer_during_an_expensive_frame(self):
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        text = ["initial"]

        def build():
            if text[0] == "expensive":
                entered.set()
                release.wait(1)
            return Text(text[0])

        live = self.create_live(build)
        text[0] = "expensive"
        live.refresh()
        self.assertTrue(entered.wait(1))

        def produce():
            text[0] = "latest"
            live.refresh()
            returned.set()

        producer = threading.Thread(target=produce)
        try:
            producer.start()
            self.assertTrue(returned.wait(0.1))
        finally:
            release.set()
            producer.join(timeout=1)

    def test_stop_wakes_and_terminates_the_sleeping_scheduler(self):
        live = self.create_live(lambda: Text("waiting"))
        worker = live._scheduler
        live.stop()
        worker.join(timeout=0.2)
        self.assertFalse(worker.is_alive())

    def test_transient_layout_failure_does_not_kill_refresh_thread(self):
        fail_next = threading.Event()
        failed, recovered = threading.Event(), threading.Event()

        def build():
            if fail_next.is_set():
                fail_next.clear()
                failed.set()
                raise RuntimeError("transient layout failure")
            if failed.is_set():
                recovered.set()
            return Text("recovered" if failed.is_set() else "initial")

        live = self.create_live(build)
        fail_next.set()
        live.refresh()
        self.assertTrue(failed.wait(1))
        self.assertTrue(recovered.wait(2))
        self.assertTrue(live._scheduler.is_alive())

    def test_failed_scheduler_start_restores_the_live_lifecycle(self):
        output = io.StringIO()
        console = Console(file=output, force_terminal=True)
        live = TaskLive(
            Text("waiting"), console=console, redirect_stdout=False, redirect_stderr=False
        )
        with (
            patch(
                "terminal_rendering.threading.Thread.start", side_effect=RuntimeError("no threads")
            ),
            self.assertRaisesRegex(RuntimeError, "no threads"),
        ):
            live.start(refresh=True)
        self.assertFalse(live.is_started)
        self.assertEqual(console._live_stack, [])


if __name__ == "__main__":
    unittest.main()
