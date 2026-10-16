import hashlib
import json
import threading
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from pathlib import Path

from file_state import FileChangedError, atomic_write_text, file_digest, file_lock
from tool_runtime import ArtifactStore

MAX_AUTO_MERGE_CHARS = 1_000_000
LEDGER_CHARS = 16_000_000
WRITES_PER_PATH = 32


def text_digest(content: str | None) -> str | None:
    return None if content is None else hashlib.sha256(content.encode("utf-8")).hexdigest()


class FileMergeConflict(FileChangedError):
    pass


def merge_text(base: str, current: str, proposed: str) -> str:
    if current in (base, proposed):
        return proposed
    if proposed == base:
        return current
    if max(map(len, (base, current, proposed))) > MAX_AUTO_MERGE_CHARS:
        raise FileMergeConflict("File is too large for automatic merging; review the saved patch.")
    original_lines = base.splitlines(keepends=True)

    def differences(content):
        lines = content.splitlines(keepends=True)
        changes = []
        for operation, start, end, new_start, new_end in SequenceMatcher(
            None, original_lines, lines
        ).get_opcodes():
            if operation == "equal":
                continue
            if operation == "replace" and end - start == new_end - new_start:
                changes.extend(
                    (start + offset, start + offset + 1, [line])
                    for offset, line in enumerate(lines[new_start:new_end])
                    if original_lines[start + offset] != line
                )
            else:
                changes.append((start, end, lines[new_start:new_end]))
        return changes

    existing = differences(current)
    incoming = differences(proposed)
    combined = list(existing)
    for change in incoming:
        start, end, replacement = change
        duplicate = False
        for other_start, other_end, other_replacement in existing:
            if change == (other_start, other_end, other_replacement):
                duplicate = True
                break
            if start == end:
                overlap = (
                    start == other_start
                    if other_start == other_end
                    else other_start <= start < other_end
                )
            elif other_start == other_end:
                overlap = start <= other_start < end
            else:
                overlap = max(start, other_start) < min(end, other_end)
            if overlap:
                raise FileMergeConflict("Overlapping changes; review and resolve the saved patch.")
        if not duplicate:
            combined.append((start, end, replacement))
    merged = list(original_lines)
    for start, end, replacement in sorted(combined, key=lambda change: change[:2], reverse=True):
        merged[start:end] = replacement
    return "".join(merged)


class EditLedger:
    """Contents the agents of a team read and wrote, shared by all of them.

    An agent whose read is stale may still apply its edit when the file has since been
    written by another agent's file tool: the edit is merged three-way with the content
    the agent read. Changes from anywhere else (the user, a shell command) are unknown
    here and still require a fresh read; overlapping changes always conflict.
    """

    def __init__(self, limit_chars: int = LEDGER_CHARS):
        self.lock = threading.Lock()
        self.limit_chars = limit_chars
        self.contents: OrderedDict[str, str] = OrderedDict()
        self.size = 0
        self.written: dict[Path, deque[str]] = {}

    def remember(self, content: str) -> str:
        digest = text_digest(content)
        if len(content) > MAX_AUTO_MERGE_CHARS:
            return digest
        with self.lock:
            if digest in self.contents:
                self.contents.move_to_end(digest)
                return digest
            self.contents[digest] = content
            self.size += len(content)
            while self.size > self.limit_chars and len(self.contents) > 1:
                _, dropped = self.contents.popitem(last=False)
                self.size -= len(dropped)
        return digest

    def content(self, digest: str | None) -> str | None:
        with self.lock:
            return self.contents.get(digest) if digest is not None else None

    def record_write(self, path: Path, content: str) -> None:
        digest = self.remember(content)
        with self.lock:
            writes = self.written.setdefault(path.resolve(), deque(maxlen=WRITES_PER_PATH))
            if digest in writes:
                writes.remove(digest)
            writes.append(digest)

    def written_by_agent(self, path: Path, digest: str | None) -> bool:
        with self.lock:
            return digest is not None and digest in self.written.get(path.resolve(), ())

    def merge(self, path: Path, base_digest: str, current: str, proposed: str) -> str:
        """Merge an edit based on a stale read with a teammate's newer content."""
        base = self.content(base_digest)
        if base is None or not self.written_by_agent(path, text_digest(current)):
            raise FileChangedError(f"File changed since it was read: {path}. Read it again.")
        try:
            return merge_text(base, current, proposed)
        except FileMergeConflict as error:
            raise FileMergeConflict(
                f"Another agent changed {path} since you read it and the changes conflict "
                f"({error}). Read it again and reapply your edit."
            ) from error


@dataclass
class FileChange:
    path: Path
    base: str | None
    proposed: str | None


class FileTransaction:
    def __init__(
        self,
        artifacts: ArtifactStore,
        on_write: Callable[[Path, str], None] | None = None,
    ):
        self.artifacts = artifacts
        self.on_write = on_write
        self.entries: dict[Path, FileChange] = {}
        self.artifact_path: Path | None = None

    @property
    def changes(self) -> list[FileChange]:
        return [entry for entry in self.entries.values() if entry.base != entry.proposed]

    def capture(self, path: Path, check) -> FileChange:
        path = path.resolve()
        if path not in self.entries:
            with file_lock(path, check):
                content = self._read_current(path, check)
            self.entries[path] = FileChange(path, content, content)
        return self.entries[path]

    def read(self, path: Path, check) -> str:
        content = self.capture(path, check).proposed
        if content is None:
            raise FileNotFoundError(f"Not a regular file: {path}")
        return content

    def write(self, path: Path, content: str, check) -> None:
        entry = self.capture(path, check)
        check()
        previous = entry.proposed
        entry.proposed = content
        try:
            self.persist()
        except BaseException:
            entry.proposed = previous
            raise

    def persist(self) -> None:
        if not self.changes and self.artifact_path is None:
            return
        if self.artifact_path is None:
            self.artifact_path = self.artifacts.new_path(".changes.json")
        payload = {
            "version": 1,
            "files": [
                {
                    "path": str(entry.path),
                    "base_sha256": text_digest(entry.base),
                    "proposed_sha256": text_digest(entry.proposed),
                    "base": entry.base,
                    "proposed": entry.proposed,
                    "diff": "".join(
                        unified_diff(
                            (entry.base or "").splitlines(keepends=True),
                            (entry.proposed or "").splitlines(keepends=True),
                            fromfile=str(entry.path),
                            tofile=str(entry.path),
                        )
                    ),
                }
                for entry in self.changes
            ],
        }
        atomic_write_text(self.artifact_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def summary(self, status: str, **extra) -> dict:
        return {
            "status": status,
            "files": [str(entry.path) for entry in self.changes],
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            **extra,
        }

    def commit(self, check) -> dict:
        changes = sorted(self.changes, key=lambda entry: str(entry.path))
        if not changes:
            return self.summary("unchanged")
        plans = []
        with ExitStack() as locks:
            for entry in changes:
                locks.enter_context(file_lock(entry.path, check))
            for entry in changes:
                check()
                current = self._read_current(entry.path, check)
                if current == entry.proposed:
                    merged = current
                elif current == entry.base:
                    merged = entry.proposed
                elif entry.base is None or current is None or entry.proposed is None:
                    raise FileMergeConflict(f"Creation/deletion conflict: {entry.path}")
                else:
                    try:
                        merged = merge_text(entry.base, current, entry.proposed)
                    except FileMergeConflict as error:
                        raise FileMergeConflict(f"{entry.path}: {error}") from error
                plans.append((entry.path, current, merged))
            written = []
            try:
                for path, current, merged in plans:
                    check()
                    if current != merged:
                        atomic_write_text(path, merged, expected=text_digest(current), check=check)
                        written.append((path, current, merged))
            except BaseException as error:
                rollback_errors = []
                for path, previous, applied in reversed(written):
                    try:
                        if previous is None:
                            if file_digest(path) != text_digest(applied):
                                raise FileChangedError(f"File changed during rollback: {path}")
                            path.unlink()
                        else:
                            atomic_write_text(path, previous, expected=text_digest(applied))
                    except Exception as rollback_error:
                        rollback_errors.append(f"{path}: {rollback_error}")
                if rollback_errors:
                    raise FileChangedError(
                        f"Commit failed ({error}); incomplete rollback: {'; '.join(rollback_errors)}"
                    ) from error
                raise
            # Only a committed transaction counts as written: a rollback leaves nothing.
            if self.on_write is not None:
                for path, _, merged in written:
                    self.on_write(path, merged)
        return self.summary("merged")

    @staticmethod
    def _read_current(path: Path, check) -> str | None:
        check()
        version = file_digest(path, check)
        if version is None:
            return None
        with path.open(encoding="utf-8", newline="") as stream:
            content = stream.read()
        check()
        if file_digest(path, check) != version or text_digest(content) != version:
            raise FileChangedError(f"File changed while capturing a snapshot: {path}")
        return content
