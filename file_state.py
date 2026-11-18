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
CONFIG_NAMES = frozenset({
    "package.json", "package-lock.json", "tsconfig.json", "jsconfig.json",
    "babel.config.json", ".babelrc", ".babelrc.json", ".eslintrc",
    ".eslintrc.json", ".eslintrc.yaml", ".eslintrc.yml",
    "pyproject.toml", "uv.lock", "poetry.lock", "Cargo.toml", "Cargo.lock",
    "go.mod", "go.sum", "setup.cfg", "tox.ini", "pytest.ini", "ruff.toml", ".ruff.toml",
    ".gitlab-ci.yml", "bitbucket-pipelines.yml", "azure-pipelines.yml", "Jenkinsfile",
    "compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml",
    "pnpm-lock.yaml", "pnpm-workspace.yaml", "yarn.lock", ".yarnrc.yml", "deno.json",
    "CMakeLists.txt", "meson.build", "Gemfile", "Gemfile.lock", "composer.json", "pom.xml",
    "mcp.json", ".mcp.json",
})
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


# Filesystem timestamps have limited resolution, so two same-size writes within the same
# tick can produce an identical (size, mtime, ctime) identity. A cached digest is only safe
# to trust once that tick has passed; until then a later write could reuse the same identity
# with different content. 2s covers coarse-resolution filesystems (FAT, some NFS) as well.
_DIGEST_CACHE_SKEW_NS = 2_000_000_000


def _digest_identity_is_settled(info) -> bool:
    newest = max(info.st_mtime_ns, info.st_ctime_ns)
    return time.time_ns() - newest > _DIGEST_CACHE_SKEW_NS


def _file_identity(info):
    return (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
        info.st_mode & 0o111,
    )


def cached_file_digest(
    path: Path, cache: dict, check=None, *, mode_aware=False, identities=None,
) -> str | None:
    cache_key = (str(path), True) if mode_aware else str(path)
    for _attempt in range(3):
        if check:
            check()
        try:
            before = path.stat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(before.st_mode):
            raise OSError(f"Cannot track a non-regular source file: {path}")
        identity = _file_identity(before)
        previous = cache.get(cache_key)
        if previous is not None and previous[0] == identity:
            if identities is not None:
                identities[str(path)] = identity
            return previous[1]
        digest = file_digest(path, check)
        if digest is None:
            return None
        try:
            after = path.stat()
        except FileNotFoundError:
            return None
        if identity == _file_identity(after):
            if identities is not None:
                identities[str(path)] = identity
            if mode_aware:
                digest = f"{digest}:{after.st_mode & 0o111:o}"
            # Only cache the identity once its timestamps can no longer collide with a
            # future same-size write; otherwise recompute next time to avoid a stale digest.
            if _digest_identity_is_settled(after):
                cache[cache_key] = (identity, digest)
            return digest
    raise OSError(f"File kept changing while tracking its version: {path}")


def verification_file(path: Path, suffixes=CODE_SUFFIXES) -> bool:
    parts = path.parts
    directories = tuple(zip(parts, parts[1:]))
    declared = (
        path.suffix.lower() in suffixes
        or path.name in SOURCE_NAMES | CONFIG_NAMES
        or path.match("tsconfig.*.json")
        or path.match("requirements*.txt")
        or path.match("Dockerfile.*")
        or (
            path.suffix.lower() in {".yaml", ".yml"}
            and (
                ".circleci" in parts or ".buildkite" in parts
                or (".github", "workflows") in directories
                or (".github", "actions") in directories
            )
        )
        or (path.suffix.lower() == ".json" and ("builtin", "environment-templates") in directories)
        or (path.suffix.lower() == ".json" and (".agent", "environments") in directories)
    )
    if declared or path.suffix or path.name.startswith("."):
        return declared
    if path.parent.name == "bin":
        return True
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISREG(mode) and bool(mode & 0o111)


def directory_identity(path, identities):
    """Capture entry churn, including creation and removal of the same name."""
    while True:
        try:
            info = path.stat()
        except FileNotFoundError:
            if path.parent == path:
                raise
            path = path.parent
            continue
        if stat.S_ISDIR(info.st_mode):
            identities[path] = (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns)
        return


def source_snapshot(
    roots, *, ignored=(), deadline=None, cache=None, relevant=verification_file, identities=None,
    directory_identities=None,
) -> dict[str, str]:
    ignored = {path.resolve() for path in ignored}
    cache = cache if cache is not None else {}
    result = {}

    def check():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Task deadline reached while tracking file changes")

    def walk_error(error):
        raise error

    for root in roots:
        root = root.resolve()
        check()
        try:
            mode = root.stat().st_mode
        except FileNotFoundError:
            if directory_identities is not None:
                directory_identity(root, directory_identities)
            continue
        if not stat.S_ISDIR(mode):
            if directory_identities is not None:
                directory_identity(root.parent, directory_identities)
            if root not in ignored and relevant(root):
                digest = cached_file_digest(root, cache, check, mode_aware=True, identities=identities)
                if digest is not None:
                    result[str(root)] = digest
            continue
        for directory, directories, filenames in os.walk(root, onerror=walk_error):
            check()
            parent = Path(directory)
            if directory_identities is not None:
                directory_identity(parent, directory_identities)
            directories[:] = [
                name
                for name in directories
                if name not in GENERATED_DIRECTORIES and parent / name not in ignored
            ]
            for name in filenames:
                path = parent / name
                if path in ignored or not relevant(path):
                    continue
                digest = cached_file_digest(path, cache, check, mode_aware=True, identities=identities)
                if digest is not None:
                    result[str(path)] = digest
    return result
