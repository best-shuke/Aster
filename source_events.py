"""Path-level filesystem evidence used to distinguish source changes from build output."""

import ctypes
import itertools
import os
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path
from weakref import ref

_MAX_EVENTS = 32_768
_INOTIFY_EVENT = struct.Struct("=iIII")
_INOTIFY_MUTATION = 0x00000FCE
_INOTIFY_DIRECTORY = 0x40000000
_INOTIFY_LOSS = 0x00006C00
_INOTIFY_ENTRY = 0x000003C0
_FSEVENT_LOSS = 0x000000EF
_FSEVENT_MUTATION = 0x0000DF00
_FSEVENT_DIRECTORY = 0x00020000
_FSEVENT_ENTRY = 0x00000B00
_MAC_OBSERVERS = {}
_MAC_CONTEXT_IDS = itertools.count(1)
_POINTER = ctypes.c_void_p
_MAC_CALLBACK = ctypes.CFUNCTYPE(
    None, _POINTER, _POINTER, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint64),
)
_DISPATCH_CALLBACK = ctypes.CFUNCTYPE(None, _POINTER)


class _MacContext(ctypes.Structure):
    _fields_ = [("version", ctypes.c_long), ("info", _POINTER), ("retain", _POINTER),
                ("release", _POINTER), ("copy_description", _POINTER)]


@_MAC_CALLBACK
def _receive_macos(_stream, context, count, paths, flags, _ids):
    observer = _MAC_OBSERVERS.get(context)
    if observer is not None:
        observer(count, paths, flags)


@_DISPATCH_CALLBACK
def _dispatch_barrier(_context):
    pass


class SourceEvents:
    """Bounded event capture. Missing or lost coverage is visible to the caller."""

    def __init__(self, directories, deadline=None):
        self.paths = set()
        self.complete = True
        self._handle = None
        self._read = None
        self._stop = None
        self._temporary = None
        self.deadline = deadline
        try:
            if sys.platform == "darwin":
                self._start_macos(directories)
            elif sys.platform.startswith("linux"):
                self._start_linux(directories)
            else:
                self.complete = False
        except (AttributeError, OSError, ValueError):
            self.complete = False
            self.close()

    @staticmethod
    def _function(library, name, result, *arguments):
        function = getattr(library, name)
        function.restype = result
        function.argtypes = arguments
        return function

    def _start_linux(self, directories):
        libc = ctypes.CDLL(None, use_errno=True)
        init = self._function(libc, "inotify_init1", ctypes.c_int, ctypes.c_int)
        add = self._function(
            libc, "inotify_add_watch", ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32,
        )
        descriptor = init(os.O_NONBLOCK | os.O_CLOEXEC)
        if descriptor < 0:
            raise OSError(ctypes.get_errno(), "Cannot start source event observation")
        self._handle = descriptor
        self._stop = os.close
        watches = {}
        for directory in directories:
            watch = add(descriptor, os.fsencode(directory), _INOTIFY_MUTATION | 0x01000000)
            if watch < 0:
                raise OSError(ctypes.get_errno(), f"Cannot observe source directory: {directory}")
            watches[watch] = directory
        owner_ref = ref(self)
        self._read = lambda descriptor: owner_ref()._read_linux(descriptor, watches)

    def _read_linux(self, descriptor, watches):
        while self.complete:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self.complete = False
                return
            try:
                data = os.read(descriptor, 64 * 1024)
            except BlockingIOError:
                break
            cursor = 0
            while cursor < len(data):
                watch, mask, _cookie, size = _INOTIFY_EVENT.unpack_from(data, cursor)
                cursor += _INOTIFY_EVENT.size
                name = data[cursor:cursor + size].split(b"\0", 1)[0]
                cursor += size
                if mask & _INOTIFY_LOSS or watch not in watches:
                    self.complete = False
                    return
                is_directory = bool(mask & _INOTIFY_DIRECTORY) or not name
                entry = bool(mask & _INOTIFY_ENTRY)
                self._record(watches[watch] / os.fsdecode(name), is_directory, entry)

    def _start_macos(self, directories):
        core = ctypes.CDLL("/System/Library/Frameworks/CoreServices.framework/CoreServices")
        foundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        dispatch = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        pointer = _POINTER
        create_string = self._function(
            foundation, "CFStringCreateWithCString", pointer, pointer, ctypes.c_char_p, ctypes.c_uint32,
        )
        create_array = self._function(
            foundation, "CFArrayCreate", pointer, pointer, ctypes.POINTER(pointer), ctypes.c_long, pointer,
        )
        release = self._function(foundation, "CFRelease", None, pointer)
        queue_create = self._function(dispatch, "dispatch_queue_create", pointer, ctypes.c_char_p, pointer)
        queue_release = self._function(dispatch, "dispatch_release", None, pointer)
        sync = self._function(dispatch, "dispatch_sync_f", None, pointer, pointer, _DISPATCH_CALLBACK)
        create = self._function(
            core, "FSEventStreamCreate", pointer, pointer, _MAC_CALLBACK, ctypes.POINTER(_MacContext), pointer,
            ctypes.c_uint64, ctypes.c_double, ctypes.c_uint32,
        )
        set_queue = self._function(core, "FSEventStreamSetDispatchQueue", None, pointer, pointer)
        start = self._function(core, "FSEventStreamStart", ctypes.c_bool, pointer)
        stop = self._function(core, "FSEventStreamStop", None, pointer)
        invalidate = self._function(core, "FSEventStreamInvalidate", None, pointer)
        stream_release = self._function(core, "FSEventStreamRelease", None, pointer)

        self._temporary = tempfile.TemporaryDirectory(prefix="aster-source-events-")
        marker_root = Path(self._temporary.name).resolve()
        start_marker, end_marker = marker_root / "start", marker_root / "end"
        marker_device = marker_root.stat().st_dev
        if any(directory.stat().st_dev != marker_device for directory in directories):
            raise OSError("Source folders and event marker span different devices")
        ready = threading.Event()
        finished = threading.Event()
        paths_seen = self.paths
        lost = threading.Event()

        def on_events(count, paths, flags):
            end_seen = False
            try:
                for index in range(count):
                    path = Path(os.fsdecode(paths[index]))
                    mask = flags[index]
                    if mask & _FSEVENT_LOSS:
                        lost.set()
                    if path == start_marker:
                        ready.set()
                    elif path == end_marker:
                        end_seen = True
                    elif ready.is_set() and path != marker_root and mask & _FSEVENT_MUTATION:
                        is_directory = bool(mask & _FSEVENT_DIRECTORY)
                        entry = bool(mask & _FSEVENT_ENTRY)
                        if len(paths_seen) == _MAX_EVENTS:
                            lost.set()
                        else:
                            paths_seen.add((path, is_directory, entry))
            except (OSError, ValueError, TypeError):
                lost.set()
            if end_seen or lost.is_set():
                finished.set()

        roots = []
        for directory in sorted({*directories, marker_root}, key=lambda path: len(path.parts)):
            if not any(directory.is_relative_to(root) for root in roots):
                roots.append(directory)
        strings = []
        array = None
        context_id = next(_MAC_CONTEXT_IDS)
        context = _MacContext(0, context_id)
        try:
            for root in roots:
                string = create_string(None, os.fsencode(root), 0x08000100)
                if not string:
                    raise OSError(f"Cannot observe source directory: {root}")
                strings.append(string)
            callbacks = ctypes.addressof(ctypes.c_byte.in_dll(foundation, "kCFTypeArrayCallBacks"))
            array = create_array(None, (pointer * len(strings))(*strings), len(strings), callbacks)
            if not array:
                raise OSError("Cannot prepare source event roots")
            stream = create(None, _receive_macos, ctypes.byref(context), array, 2**64 - 1, 0.02, 0x16)
        finally:
            if array:
                release(array)
            for string in strings:
                release(string)
        if not stream:
            raise OSError("Cannot create source event stream")
        queue = queue_create(b"aster.source.events", None)
        if not queue:
            stream_release(stream)
            raise OSError("Cannot create source event queue")
        _MAC_OBSERVERS[context_id] = on_events
        set_queue(stream, queue)
        started = start(stream)

        def close_stream(handle):
            if started:
                stop(handle)
            invalidate(handle)
            sync(queue, None, _dispatch_barrier)
            _MAC_OBSERVERS.pop(context_id, None)
            stream_release(handle)
            queue_release(queue)

        self._handle = stream
        self._stop = close_stream
        if not started:
            raise OSError("Cannot start source event stream")

        owner_ref = ref(self)

        def wait_for_marker(marker, delivered):
            owner = owner_ref()
            timeout = 0.25
            if owner.deadline is not None:
                timeout = min(timeout, owner.deadline - time.monotonic())
            if timeout <= 0 or not owner.complete or lost.is_set():
                owner.complete = False
                return
            marker.touch(exist_ok=False)
            if not delivered.wait(timeout) or lost.is_set():
                owner.complete = False

        wait_for_marker(start_marker, ready)
        self._read = lambda _handle: wait_for_marker(end_marker, finished)

    def _record(self, path, is_directory, entry):
        if len(self.paths) >= _MAX_EVENTS:
            self.complete = False
        else:
            self.paths.add((path, is_directory, entry))

    def finish(self):
        try:
            if self._handle is not None and self._read is not None:
                self._read(self._handle)
        except OSError:
            self.complete = False
        finally:
            self.close()
        return self.paths, self.complete

    def close(self):
        handle, self._handle = self._handle, None
        try:
            if handle is not None:
                self._stop(handle)
        finally:
            self._read = self._stop = None
            if self._temporary is not None:
                temporary, self._temporary = self._temporary, None
                temporary.cleanup()

    def __del__(self):
        self.close()
