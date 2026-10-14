"""A terminal inspector with a call timeline and a searchable detail pane."""

from pathlib import Path

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import has_focus
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, ScrollOffsets, VSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer, DynamicContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import SearchToolbar, TextArea

from execution_history import ArtifactPages, ExecutionEntry, plain, result_summary

STATUS_LABELS = {
    "ok": "完成",
    "completed": "完成",
    "running": "运行中",
    "pending": "待执行",
    "partial": "部分完成",
    "timeout": "超时",
    "unknown": "结果未知",
    "error": "失败",
    "not_executed": "未执行",
}


class TraceInspector:
    def __init__(
        self,
        entries: list[ExecutionEntry],
        *,
        selected: int | None = None,
        query: str = "",
        errors: bool = False,
        colors: bool = True,
        full_screen: bool = True,
    ):
        self.entries = entries
        self.filtered = entries
        self.selected = selected if selected is not None else len(entries) - 1
        self.errors_only = errors
        self.latest_task = False
        self.artifact: ArtifactPages | None = None
        self.notice = ""
        self.search = SearchToolbar(ignore_case=True)
        self.details = TextArea(
            read_only=True, scrollbar=True, wrap_lines=True, search_field=self.search
        )
        self.filter_buffer = Buffer()
        self.filter_buffer.text = query
        self.filter_control = BufferControl(buffer=self.filter_buffer)
        self.list_control = FormattedTextControl(self._rows, focusable=True, show_cursor=False)
        self.filter_buffer.on_text_changed += lambda _: self._filter()
        self._filter()
        self.application = self._application(colors, full_screen)

    @property
    def entry(self) -> ExecutionEntry | None:
        return self.filtered[self.selected] if self.filtered else None

    def _filter(self) -> None:
        current = self.entry if 0 <= self.selected < len(self.filtered) else None
        query = self.filter_buffer.text.casefold().strip()
        task_id = (self.entries[-1].run.get("trace") or {}).get("task_id") if self.entries else None
        self.filtered = [
            entry
            for entry in self.entries
            if all(word in entry.searchable for word in query.split())
            and (
                not self.errors_only
                or entry.status not in {"ok", "completed", "pending", "running"}
            )
            and (not self.latest_task or (entry.run.get("trace") or {}).get("task_id") == task_id)
        ]
        self.filtered.sort(key=lambda entry: entry.run.get("started_at", float("inf")))
        self.selected = next(
            (
                index
                for index, entry in enumerate(self.filtered)
                if current and entry.number == current.number
            ),
            max(0, len(self.filtered) - 1),
        )
        self._show_detail()

    def _show_detail(self) -> None:
        self.artifact = None
        self.notice = ""
        entry = self.entry
        self.details.text = (
            entry.detail(self.entries) if entry else "没有匹配的执行记录。按 c 清除筛选。"
        )
        self.details.buffer.cursor_position = 0

    def move(self, distance: int) -> None:
        if self.filtered:
            self.selected = max(0, min(len(self.filtered) - 1, self.selected + distance))
            self._show_detail()

    def open_artifact(self) -> None:
        path = self.entry.result.get("artifact_path") if self.entry else None
        if not path:
            self.notice = "本次调用没有保留文件。"
            return
        try:
            self.artifact = ArtifactPages(Path(path))
            self._show_page()
        except (OSError, ValueError) as exc:
            self.artifact = None
            self.notice = plain(f"无法打开保留文件：{exc}")

    def _show_page(self) -> None:
        artifact = self.artifact
        self.details.text = (
            f"保留文件：{plain(artifact.path)}\n"
            f"第 {artifact.index + 1} 页 · {'后面还有内容' if artifact.has_next else '文件末尾'}\n"
            "[ / ] 切换文件页 · / 搜索当前页 · s 返回调用详情\n\n" + artifact.text
        )
        self.details.buffer.cursor_position = 0
        self.notice = ""

    def page(self, direction: int) -> None:
        if self.artifact is not None:
            try:
                if self.artifact.move(direction):
                    self._show_page()
            except (OSError, ValueError) as exc:
                self.notice = plain(f"读取失败：{exc}")

    def _rows(self):
        fragments = []
        previous_task = object()
        for index, entry in enumerate(self.filtered):
            task = (entry.run.get("trace") or {}).get("task_id")
            if task != previous_task:
                title = plain(entry.run.get("task_title") or "旧会话记录")
                fragments.append(("class:muted", f" ── {title[:80]} ──\n"))
                previous_task = task
            selected = index == self.selected
            style = "class:selected" if selected else ""
            if selected:
                fragments.append(("[SetCursorPosition]", ""))
            marker = "›" if selected else " "
            label = STATUS_LABELS.get(entry.status, entry.status)
            duration = entry.run.get("duration_seconds")
            timing = f" · {duration:.1f}s" if isinstance(duration, (int, float)) else ""
            fragments.append(
                (style, f" {marker} #{entry.number}  {entry.run.get('tool')} · {entry.agent}\n")
            )
            fragments.append(
                (style or "class:muted", f"    {label}{timing} · {entry.target[:256]}\n")
            )
        return fragments or [("class:muted", "  没有匹配的调用\n")]

    def _heading(self):
        scopes = ["当前任务" if self.latest_task else "整个会话"]
        if self.errors_only:
            scopes.append("仅异常")
        return [
            (
                "class:heading",
                f"  执行轨迹 · {' / '.join(scopes)} · {len(self.filtered)}/{len(self.entries)} 次调用  ",
            )
        ]

    def _footer(self):
        summary = self.notice or (result_summary(self.entry.result) if self.entry else "")
        if get_app().output.get_size().columns < 100:
            return [
                (
                    "class:muted",
                    f"  {summary}\n"
                    "  ↑↓ 选择 · Enter/Tab 详情\n"
                    "  / 搜索 · e 异常 · t 当前任务\n"
                    "  o 文件 · [ ] 翻页 · s 详情\n"
                    "  c 清筛选 · Esc 返回",
                )
            ]
        return [
            (
                "class:muted",
                f"  {summary}\n  ↑↓ 选择 · Enter 详情 · Tab 切换 · / 筛选或页内搜索 · e 异常 · t 当前任务\n"
                "  o 保留文件 · [ ] 文件翻页 · s 调用详情 · c 清除筛选 · Esc 返回",
            )
        ]

    def _application(self, colors: bool, full_screen: bool) -> Application:
        bindings = KeyBindings()
        browsing = has_focus(self.list_control) | has_focus(self.details)

        @bindings.add("up", filter=has_focus(self.list_control))
        def previous(event):
            self.move(-1)

        @bindings.add("down", filter=has_focus(self.list_control))
        def next_call(event):
            self.move(1)

        @bindings.add("pageup", filter=has_focus(self.list_control))
        def previous_page(event):
            self.move(-10)

        @bindings.add("pagedown", filter=has_focus(self.list_control))
        def next_page(event):
            self.move(10)

        @bindings.add("home", filter=has_focus(self.list_control))
        def first(event):
            self.move(-len(self.filtered))

        @bindings.add("end", filter=has_focus(self.list_control))
        def last(event):
            self.move(len(self.filtered))

        @bindings.add("enter", filter=has_focus(self.list_control) | has_focus(self.filter_control))
        def inspect(event):
            event.app.layout.focus(
                self.details if event.app.layout.has_focus(self.list_control) else self.list_control
            )

        @bindings.add("tab", filter=browsing)
        @bindings.add("s-tab", filter=browsing)
        def switch(event):
            event.app.layout.focus(
                self.details if event.app.layout.has_focus(self.list_control) else self.list_control
            )

        @bindings.add("/", filter=has_focus(self.list_control))
        def filter_calls(event):
            event.app.layout.focus(self.filter_control)

        @bindings.add("/", filter=has_focus(self.details))
        def search_output(event):
            from prompt_toolkit.search import start_search

            start_search(self.details.control)

        @bindings.add("e", filter=browsing)
        def errors(event):
            self.errors_only = not self.errors_only
            self._filter()

        @bindings.add("t", filter=browsing)
        def task(event):
            self.latest_task = not self.latest_task
            self._filter()

        @bindings.add("c", filter=browsing)
        def clear(event):
            self.errors_only = self.latest_task = False
            self.filter_buffer.text = ""
            self._filter()

        @bindings.add("o", filter=browsing)
        def output(event):
            self.open_artifact()
            event.app.layout.focus(self.details)

        @bindings.add("]", filter=browsing)
        def more_output(event):
            self.page(1)

        @bindings.add("[", filter=browsing)
        def less_output(event):
            self.page(-1)

        @bindings.add("s", filter=browsing)
        def summary(event):
            self._show_detail()

        @bindings.add("escape", eager=True)
        @bindings.add("c-c", eager=True)
        @bindings.add("c-d", eager=True)
        def close(event):
            if event.app.layout.has_focus(self.search):
                from prompt_toolkit.search import stop_search

                stop_search(self.details.control)
            elif event.app.layout.has_focus(self.filter_control):
                event.app.layout.focus(self.list_control)
            else:
                event.app.exit()

        list_window = Window(
            self.list_control,
            wrap_lines=False,
            always_hide_cursor=True,
            scroll_offsets=ScrollOffsets(top=1, bottom=1),
        )
        wide = VSplit([HSplit([list_window], width=42), Window(width=1, char="│"), self.details])
        narrow = HSplit(
            [
                HSplit([list_window], height=lambda: min(6, max(3, len(self.filtered) * 2 + 1))),
                Window(height=1, char="─"),
                self.details,
            ]
        )
        layout = HSplit(
            [
                Window(FormattedTextControl(self._heading), height=1),
                VSplit(
                    [
                        Window(FormattedTextControl("  筛选 / "), width=10),
                        Window(self.filter_control, height=1),
                    ],
                    height=1,
                ),
                Window(height=1, char="─"),
                DynamicContainer(
                    lambda: wide if get_app().output.get_size().columns >= 100 else narrow
                ),
                ConditionalContainer(self.search, filter=has_focus(self.search)),
                Window(
                    FormattedTextControl(self._footer),
                    height=lambda: 5 if get_app().output.get_size().columns < 100 else 3,
                    wrap_lines=False,
                ),
            ]
        )
        app = Application(
            layout=Layout(layout, focused_element=self.list_control),
            key_bindings=bindings,
            full_screen=full_screen,
            erase_when_done=True,
            include_default_pygments_style=False,
            style=Style.from_dict(
                {
                    "heading": "bold ansicyan",
                    "selected": "bold ansicyan reverse",
                    "muted": "ansibrightblack",
                }
                if colors
                else {}
            ),
        )
        app.ttimeoutlen = 0.1
        return app

    def run(self) -> None:
        self.application.run()
