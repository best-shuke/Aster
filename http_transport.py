"""Cancellable HTTP reads with no credential redirects.

DNS resolution cannot be interrupted portably. Its daemon thread cannot send
a request after cancellation or hold up process exit. Established sockets
are shut down immediately on cancellation.
"""

import http.client
import queue
import socket
import threading
import time
import urllib.request
from contextlib import suppress

class TransportCancelled(Exception):
    pass


class TransportDeadline(TimeoutError):
    pass


_EOF = object()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def build_no_redirect_opener(*handlers):
    """Build an opener that never forwards credentials through redirects."""
    return urllib.request.build_opener(_NoRedirect(), *handlers)


class _ConnectionMixin:
    def __init__(self, *args, guard, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard = guard

        def tracked_connect(address, timeout, source_address=None):
            error = None
            for family, kind, protocol, _, endpoint in socket.getaddrinfo(
                address[0], address[1], 0, socket.SOCK_STREAM
            ):
                guard.check()
                sock = socket.socket(family, kind, protocol)
                # Track before connect(), so cancellation can also interrupt a
                # pending TCP connection rather than waiting for its timeout.
                guard.track(sock)
                try:
                    sock.settimeout(timeout)
                    if source_address:
                        sock.bind(source_address)
                    sock.connect(endpoint)
                    guard.check()
                    return sock
                except OSError as exc:
                    error = exc
                    sock.close()
                    guard.check()
            raise error or OSError("No addresses found for HTTP endpoint")

        self._create_connection = tracked_connect

    def send(self, data):
        self.guard.check()
        return super().send(data)


class _HTTPConnection(_ConnectionMixin, http.client.HTTPConnection):
    pass


class _HTTPSConnection(_ConnectionMixin, http.client.HTTPSConnection):
    def connect(self):
        http.client.HTTPConnection.connect(self)
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname=self._tunnel_host or self.host,
            do_handshake_on_connect=False,
        )
        self.guard.track(self.sock)
        self.sock.do_handshake()


class _HTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, guard):
        super().__init__()
        self.guard = guard

    def http_open(self, req):
        return self.do_open(lambda *a, **kw: _HTTPConnection(*a, guard=self.guard, **kw), req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, guard):
        super().__init__()
        self.guard = guard

    def https_open(self, req):
        return self.do_open(lambda *a, **kw: _HTTPSConnection(*a, guard=self.guard, **kw), req)


class _Response:
    def __init__(self, request, timeout, deadline, cancelled, on_progress):
        self.request, self.timeout, self.deadline = request, timeout, deadline
        self.cancelled, self.on_progress = cancelled, on_progress
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.sockets = []
        self.chunks = queue.Queue(maxsize=8)
        self.pending = bytearray()
        self.offset = 0
        self.eof = False
        self.received = 0
        self.started = time.monotonic()
        self.thread = None

    def check(self):
        if self.stopped.is_set() or (self.cancelled is not None and self.cancelled.is_set()):
            raise TransportCancelled("Request cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TransportDeadline("Request deadline reached")

    @staticmethod
    def _close_socket(sock):
        with suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        with suppress(OSError):
            sock.close()

    def track(self, sock):
        with self.lock:
            if self.stopped.is_set():
                self._close_socket(sock)
                raise TransportCancelled("Request cancelled before connection was ready")
            self.sockets.append(sock)
        self.check()

    def progress(self):
        if self.on_progress:
            self.on_progress(
                {
                    "phase": "receiving" if self.received else "connecting",
                    "elapsed_seconds": time.monotonic() - self.started,
                    "received_bytes": self.received,
                }
            )

    def put(self, item):
        while not self.stopped.is_set():
            try:
                self.chunks.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def worker(self):
        try:
            opener = build_no_redirect_opener(_HTTPHandler(self), _HTTPSHandler(self))
            self.check()
            with opener.open(self.request, timeout=self.timeout) as response:
                self.put(("headers", dict(response.headers)))
                while not self.stopped.is_set():
                    self.check()
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.put(chunk)
            self.put(_EOF)
        except Exception as exc:
            self.put(exc)

    def next_chunk(self):
        idle_deadline = time.monotonic() + self.timeout
        while True:
            self.check()
            if time.monotonic() >= idle_deadline:
                raise TimeoutError("HTTP response idle timeout")
            self.progress()
            try:
                item = self.chunks.get(timeout=0.05)
            except queue.Empty:
                continue
            self.check()
            if isinstance(item, Exception):
                raise item
            if isinstance(item, bytes):
                self.received += len(item)
            return item

    def start(self):
        try:
            self.check()
            self.thread = threading.Thread(target=self.worker, name="model-http", daemon=True)
            self.thread.start()
            headers = self.next_chunk()
            if not isinstance(headers, tuple) or headers[0] != "headers":
                raise OSError("HTTP response has no headers")
            self.headers = headers[1]
            return self
        except BaseException:
            self.close()
            raise

    def _refill(self):
        if self.offset:
            del self.pending[:self.offset]
            self.offset = 0
        chunk = self.next_chunk()
        if chunk is _EOF:
            self.eof = True
        else:
            self.pending.extend(chunk)

    def _consume(self, end):
        data = bytes(memoryview(self.pending)[self.offset:end])
        if end == len(self.pending):
            self.pending.clear()
            self.offset = 0
        else:
            self.offset = end
        return data

    def read(self, size=-1):
        try:
            while not self.eof and (size < 0 or len(self.pending) - self.offset < size):
                self._refill()
            end = len(self.pending) if size < 0 else min(len(self.pending), self.offset + size)
            return self._consume(end)
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        try:
            search = self.offset
            while True:
                newline = self.pending.find(b"\n", search)
                if newline >= 0:
                    yield self._consume(newline + 1)
                    search = self.offset
                elif self.eof:
                    if self.offset < len(self.pending):
                        yield self._consume(len(self.pending))
                    return
                else:
                    search = len(self.pending) - self.offset
                    self._refill()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stopped.set()
        with self.lock:
            sockets, self.sockets = self.sockets, []
        for sock in sockets:
            self._close_socket(sock)
        if (
            self.thread is not None
            and self.thread.ident is not None
            and self.thread is not threading.current_thread()
        ):
            self.thread.join(timeout=0.1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open_url(request, timeout=180, *, deadline=None, cancelled=None, on_progress=None):
    return _Response(request, timeout, deadline, cancelled, on_progress).start()
