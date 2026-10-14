import fcntl
import hashlib
import os
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

SOURCE_SUFFIXES = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".java",
        ".sh",
        ".sql",
        ".kt",
        ".swift",
        ".rb",
        ".php",
        ".vue",
        ".svelte",
        ".html",
        ".css",
        ".scss",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
    }
)
SOURCE_NAMES = frozenset({"Dockerfile", "Makefile", ".env", ".env.example"})
GENERATED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".agent-artifacts",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",
    }
)
CODE_SUFFIXES = (SOURCE_SUFFIXES - {".json", ".toml", ".yaml", ".yml"}) | {".mjs", ".cjs"}
_UNSPECIFIED = object()
_FILE_LOCKS = {}
_FILE_LOCKS_GUARD = threading.Lock()
_HELD_FILE_LOCKS = threading.local()


def local_code_files(root: Path) -> list[Path]:
    files = []
    for directory, directories, filenames in os.walk(root):
        parent = Path(directory)
        directories[:] = [
            name
            for name in directories
            if name not in GENERATED_DIRECTORIES and not name.startswith(".")
        ]
        for name in directories:
            if (parent / name).is_symlink():
                raise ValueError(f"本地代码包不能包含符号链接：{parent / name}")
        for name in filenames:
            path = parent / name
            if name.startswith(".") or (
                path.suffix.lower() not in CODE_SUFFIXES
                and name not in {"package.json", "tsconfig.json"}
            ):
                continue
            if path.is_symlink():
                raise ValueError(f"本地代码包不能包含符号链接：{path}")
            files.append(path)
            if len(files) > 4096:
                raise ValueError("本地代码包超过限制（4096 个文件）")
    return sorted(files)


class FileChangedError(RuntimeError):
    pass


@contextmanager
def file_lock(path: Path, check=None):
    identity = str(path.resolve())
    held = getattr(_HELD_FILE_LOCKS, "paths", None)
    if held is None:
        held = _HELD_FILE_LOCKS.paths = set()
    if identity in held:
        if check:
            check()
        yield
        return
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.setdefault(identity, threading.Lock())
    while not lock.acquire(timeout=0.05):
        if check:
            check()
    try:
        if check:
            check()
        directory = Path(tempfile.gettempdir()) / f"ds-agent-file-locks-{os.getuid()}"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = directory / hashlib.sha256(identity.encode()).hexdigest()
        with lock_path.open("a+b") as stream:
            while True:
                if check:
                    check()
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.01)
            held.add(identity)
            try:
                yield
            finally:
                held.remove(identity)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        lock.release()


def file_digest(path: Path, check=None) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                if check:
                    check()
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    return digest.hexdigest()
                digest.update(chunk)
    except FileNotFoundError:
        return None


def atomic_write_text(path: Path, content: str, *, expected=_UNSPECIFIED, check=None) -> None:
    with file_lock(path, check):
        _atomic_write_text(path, content, expected=expected, check=check)


def _atomic_write_text(path: Path, content: str, *, expected=_UNSPECIFIED, check=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            if check:
                check()
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary.chmod(mode)
        if check:
            check()
        if expected is not _UNSPECIFIED and file_digest(path, check) != expected:
            raise FileChangedError(
                f"File changed since it was read: {path}. Read it again before editing."
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cached_file_digest(path: Path, cache: dict, check=None) -> str | None:
    for _attempt in range(3):
        if check:
            check()
        try:
            before = path.stat()
        except FileNotFoundError:
            return None
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        previous = cache.get(str(path))
        if previous is not None and previous[0] == identity:
            return previous[1]
        digest = file_digest(path, check)
        if digest is None:
            return None
        try:
            after = path.stat()
        except FileNotFoundError:
            return None
        if identity == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            cache[str(path)] = (identity, digest)
            return digest
    raise OSError(f"File kept changing while tracking its version: {path}")


def source_snapshot(roots, *, ignored=(), deadline=None, cache=None) -> dict[str, str]:
    ignored = {path.resolve() for path in ignored}
    cache = cache if cache is not None else {}
    result = {}

    def check():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Task deadline reached while tracking file changes")

    for root in roots:
        root = root.resolve()
        for directory, directories, filenames in os.walk(root):
            check()
            parent = Path(directory)
            directories[:] = [
                name
                for name in directories
                if name not in GENERATED_DIRECTORIES and parent / name not in ignored
            ]
            for name in filenames:
                path = parent / name
                if path in ignored or (
                    path.suffix.lower() not in SOURCE_SUFFIXES and name not in SOURCE_NAMES
                ):
                    continue
                digest = cached_file_digest(path, cache, check)
                if digest is not None:
                    result[str(path)] = digest
    return result
