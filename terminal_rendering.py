"""Incremental task display, retaining Rich's scrollback and input lifecycle."""

import threading
import time
from contextlib import contextmanager

from rich.control import Control
from rich.live import Live
from rich.live_render import LiveRender
from rich.segment import ControlType, Segment


class _Frame:
    def __init__(self, lines, prefix=()):
        self.lines = lines
        self.prefix = prefix

    def __rich_console__(self, console, options):
        for index, line in enumerate(self.lines):
            yield from line
            if index < len(self.lines) - 1:
                yield Segment.line()


class StreamingDisplay:
    """A scrolling response above a task footer that remains editable."""

    def __init__(self, body, footer, *, key, height, row_lengths=None):
        self.body = body
        self.footer = footer
        self.key = key
        self.height = height
        self.row_lengths = row_lengths if row_lengths is not None else self.measure_rows(body)

    @staticmethod
    def measure_rows(lines):
        # Padding and table borders change on resize; content characters don't.
        return [
            sum(
                not char.isspace() and not "\u2500" <= char <= "\u257f"
                for segment in line
                if not segment.control
                for char in segment.text
            )
            for line in lines
        ]

    def frame(self, console, options, offset=0):
        footer = console.render_lines(self.footer, options, pad=False)
        room = max(1, min(self.height, options.size.height) - len(footer) - 1)
        offset = min(offset, len(self.body))
        visible_start = max(offset, len(self.body) - room)
        return (
            _Frame(
                [*self.body[visible_start:], [Segment("─" * options.max_width)], *footer],
                prefix=self.body[offset:visible_start],
            ),
            visible_start,
        )

    def __rich_console__(self, console, options):
        frame, _ = self.frame(console, options)
        yield from frame.__rich_console__(console, options)


class _LineDiffRender(LiveRender):
    """Overwrite changed rows before erasing their leftover suffixes."""

    def __init__(self, renderable, **kwargs):
        super().__init__(renderable, **kwargs)
        self.previous_lines = []
        self.dimensions = None
        self.full_reset = False
        self.pending_prefix = ()

    def position_cursor(self):
        if self.full_reset:
            return super().position_cursor()
        height = self.last_render_height
        return Control(
            ControlType.CARRIAGE_RETURN,
            *(((ControlType.CURSOR_UP, height - 1),) if height > 1 else ()),
        )

    def __rich_console__(self, console, options):
        # Keep the final column free: reaching it can arm terminal auto-wrap,
        # and an erase-to-end command can then erase the last printed cell.
        lines = console.render_lines(
            self.renderable, options.update(width=max(1, options.max_width - 1)), pad=False
        )
        dimensions = (options.max_width, options.size.height)
        if len(lines) > options.size.height:
            # Preserve Rich's overflow behavior for displays outside the task UI.
            self.previous_lines = []
            self.full_reset = True
            yield from super().__rich_console__(console, options)
            return
        previous = self.previous_lines
        if self.full_reset or dimensions != self.dimensions:
            previous = []
        self.full_reset = False
        self.dimensions = dimensions
        self.previous_lines = lines
        self._shape = Segment.get_shape(lines)
        painted = [*self.pending_prefix, *lines]
        self.pending_prefix = ()
        height = max(len(previous), len(painted))
        for index in range(height):
            line = painted[index] if index < len(painted) else []
            if index >= len(previous) or previous[index] != line:
                yield from line
                yield Control((ControlType.ERASE_IN_LINE, 0)).segment
            if index < height - 1:
                yield Control(ControlType.CARRIAGE_RETURN).segment
                yield Segment.line()
        if height > len(painted):
            yield Control.move(y=-(height - len(painted))).segment
            yield Control(ControlType.CARRIAGE_RETURN).segment


class TaskLive(Live):
    """Coalesce all update sources and skip frames with no visible changes.

    Rich 14's LiveRender hook is isolated here so UI state stays independent of
    cursor control. Normal log output still uses Rich's complete area reset.
    """

    def __init__(self, *args, **kwargs):
        self._frame = None
        self._last_refresh = None
        self._last_dimensions = None
        self._scheduled_refresh = kwargs.pop("auto_refresh", True)
        self._refresh_stop = threading.Event()
        self._refresh_condition = threading.Condition()
        self._refresh_pending = False
        self._render_seconds = 0.0
        self.last_render_error = None
        self._scheduler = None
        self._stream_key = None
        self._stream_offset = 0
        self._stream_chars = 0
        self._stream_width = None
        # Rich's fixed-frequency timer renders even when nothing has changed.
        kwargs["auto_refresh"] = False
        super().__init__(*args, **kwargs)
        self._live_render = _LineDiffRender(
            self.get_renderable(), vertical_overflow=self.vertical_overflow
        )

    def get_renderable(self):
        return self._frame if self._frame is not None else super().get_renderable()

    def start(self, refresh=False):
        with self._lock:
            if self.is_started:
                return
            self._last_refresh = None
            self._render_seconds = 0.0
            with self._refresh_condition:
                self._refresh_pending = False
                self._refresh_stop = threading.Event()
            self._live_render.previous_lines = []
            self._live_render._shape = None
            try:
                super().start(refresh=False)
                if refresh:
                    self._render_frame(force=True)
                if self._scheduled_refresh and not self._nested:
                    self._scheduler = threading.Thread(
                        target=self._run_refresh,
                        args=(self._refresh_stop,),
                        name="task-ui-refresh",
                        daemon=True,
                    )
                    self._scheduler.start()
            except BaseException:
                self.stop()
                raise

    def stop(self):
        with self._lock:
            with self._refresh_condition:
                self._refresh_stop.set()
                self._refresh_condition.notify_all()
            self._scheduler = None
            super().stop()

    def _frame_interval(self):
        # Target a 2% rendering duty cycle for normal frames, but never let a
        # slow Markdown frame postpone visible progress for many seconds.
        return min(1.0, max(1 / self.refresh_per_second, self._render_seconds * 50))

    def _refresh_timeout(self, now):
        if self._last_refresh is None:
            return 0.0
        interval = self._frame_interval() if self._refresh_pending else 1.0
        return max(0.0, self._last_refresh + interval - now)

    def _run_refresh(self, stopped):
        while not stopped.is_set():
            with self._refresh_condition:
                if stopped.is_set():
                    return
                self._refresh_condition.wait(self._refresh_timeout(time.monotonic()))
            with self._lock:
                if stopped.is_set():
                    return
                if self._refresh_timeout(time.monotonic()) <= 0:
                    try:
                        self._render_frame(force=True)
                    except Exception as exc:
                        # A transient UI-state/layout error must not permanently
                        # kill the sole refresh thread while tools keep running.
                        self.last_render_error = f"{type(exc).__name__}: {exc}"
                        self._last_refresh = time.monotonic()
                        self._render_seconds = 0.0
                        with self._refresh_condition:
                            self._refresh_pending = False
                    else:
                        self.last_render_error = None

    def refresh(self):
        if self._scheduled_refresh and self.is_started:
            # Network callbacks only enqueue an update; rendering never blocks
            # the producer, and a burst wakes the scheduler just once.
            with self._refresh_condition:
                if not self._refresh_pending:
                    self._refresh_pending = True
                    self._refresh_condition.notify()
            return
        self._render_frame(force=not self.is_started)

    def _render_frame(self, *, force=False, prefix=()):
        with self._lock:
            now = time.monotonic()
            if (
                not force
                and self._last_refresh is not None
                and now - self._last_refresh < self._frame_interval()
            ):
                return
            with self._refresh_condition:
                self._refresh_pending = False
            self._last_refresh = now
            renderable = self.get_renderable()
            dimensions = (self.console.width, self.console.height)
            options = self.console.options.update(width=max(1, self.console.width - 1))
            first_stream_frame = False
            if isinstance(renderable, StreamingDisplay):
                first_stream_frame = (
                    renderable.key != self._stream_key or self._stream_width != options.max_width
                )
                offset = self._stream_offset_for(renderable, options.max_width)
                source = renderable
                renderable, self._stream_offset = renderable.frame(self.console, options, offset)
                self._stream_chars += sum(source.row_lengths[offset : self._stream_offset])
                prefix = renderable.prefix
            lines = self.console.render_lines(
                renderable,
                options,
                pad=False,
            )
            if (
                self.is_started
                and not prefix
                and lines == self._live_render.previous_lines
                and dimensions == self._last_dimensions
            ):
                self._render_seconds = time.monotonic() - now
                return
            if dimensions != self._last_dimensions:
                self._live_render.full_reset = True
            self._last_dimensions = dimensions
            self._frame = _Frame(lines, prefix=prefix)
            self._live_render.pending_prefix = prefix
            try:
                super().refresh()
            finally:
                self._frame = None
                self._live_render.pending_prefix = ()
                self._render_seconds = time.monotonic() - now
                if first_stream_frame:
                    # The first layout may include a large code block; later
                    # updates reuse it and must not inherit a long startup wait.
                    self._render_seconds = min(self._render_seconds, 0.002)

    def _stream_offset_for(self, display, width):
        if display.key != self._stream_key:
            self._stream_key = display.key
            self._stream_offset = 0
            self._stream_chars = 0
        elif width != self._stream_width and self._stream_offset:
            remaining = self._stream_chars
            offset = 0
            for length in display.row_lengths:
                if length > remaining:
                    break
                remaining -= length
                offset += 1
            self._stream_offset = offset
            self._stream_chars -= remaining
        self._stream_width = width
        return self._stream_offset

    def finish_stream(self, display):
        """Retain the response in place and leave only the footer live."""
        with self._lock:
            offset = self._stream_offset_for(display, max(1, self.console.width - 1))
            self._stream_key = None
            self._stream_offset = 0
            self._render_frame(force=True, prefix=[*display.body[offset:], []])

    def process_renderables(self, renderables):
        with self._lock:
            refreshing = (
                len(renderables) == 1
                and isinstance(renderables[0], Control)
                and not renderables[0].segment.text
            )
            if not refreshing:
                self._live_render.full_reset = True
            return super().process_renderables(renderables)

    @contextmanager
    def batch(self):
        """Keep response publication and its state update in one frame."""
        with self._lock:
            yield

    def publish(self, renderable):
        with self._lock:
            with self._refresh_condition:
                self._refresh_pending = False
            self._live_render.set_renderable(self.get_renderable())
            self.console.print(renderable)
            self._last_refresh = time.monotonic()
            with self._refresh_condition:
                self._refresh_condition.notify()
