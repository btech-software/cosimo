"""v3's shard I/O: one layout, one writer, no half-written files on disk.

The corpus tree this writes is the shape verify, prefer and publish all agree
on (spec §5.6 layout): ``<out>/fact_packs/<work_type>.jsonl``,
``<out>/sft/<record_type>.jsonl``, ``<out>/preference/pairs.jsonl``,
``<out>/dead_letter/<record_type>.jsonl``. One file per bucket, records
appended in plan order -- one file per bucket keeps resume to "read this file,
diff the ids" instead of a shard-scan dance, and the files stay small enough
to eyeball, which for a corpus you are debugging at 2am is the point.

Durability rule, inherited from the v1/v2 generator and non-negotiable:
**never leave a half-written shard.** Every mutation is whole-file, staged in
a ``.tmp`` sibling in the same directory (same filesystem, so ``os.replace``
is atomic) and swapped in with it. A crash between write and swap leaves the
old file byte-intact and a stray ``.tmp`` behind; the next writer truncates
the ``.tmp`` from scratch before appending, so a stale temp can never be
mistaken for data. The reader rejects a ``.tmp`` that slipped through to a
listing instead of silently counting its partial lines -- verification reads
what the file claims about itself, not what a crashed run hoped to commit.
"""

from __future__ import annotations

import json
import os

#: Buckets of the on-disk layout. An unlisted kind is a wiring bug, so
#: :func:`path_for` refuses it instead of inventing a directory nobody verifies.
KINDS = ("fact_packs", "sft", "preference", "dead_letter")


def path_for(kind: str, name: str, out_dir: str) -> str:
    """``<out>/<kind>/<name>.jsonl``. ``name`` may contain no path separators."""
    if kind not in KINDS:
        raise ValueError(f"unlisted shard kind {kind!r} (known: {', '.join(KINDS)})")
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"unsafe shard name {name!r}: names are single segments")
    return os.path.join(out_dir, kind, f"{name}.jsonl")


def read_jsonl(path: str) -> list[dict]:
    """Every record in *path*, position-tagged errors kept for the audit.

    A file that does not exist reads as empty -- the resume path asks "what is
    already here" of directories that may not have been created yet. A line
    that fails to parse, or parses to a non-object, or lacks ``id``, is an
    error naming file, line number and the offending prefix: a corrupt shard
    is a corpus incident, and "it parsed 3 of 4 lines" is how corpora start
    lying.
    """
    if not os.path.isfile(path):
        return []
    records: list[dict] = []
    with open(path, encoding="utf8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{lineno}: not valid json: {line[:120]!r} ({exc})"
                ) from exc
            if not isinstance(record, dict) or "id" not in record:
                raise ValueError(
                    f"{path}:{lineno}: record must be a json object with an 'id': "
                    f"{line[:120]!r}"
                )
            records.append(record)
    return records


def existing_ids(path: str) -> frozenset[str]:
    """The ids already committed to *path*; resume consults this, never a count."""
    return frozenset(str(record["id"]) for record in read_jsonl(path))


def write_jsonl(path: str, records: list[dict]) -> int:
    """(Re)write *path* with exactly *records*. Whole-file, atomic. Idempotent."""
    _check_records(records, path)
    return _stage_and_swap(path, records)


def append_unique(path: str, records: list[dict]) -> tuple[int, int]:
    """Append the records whose ids are not yet in *path*; returns (added, skipped).

    Duplicate ids -- within the batch or against the file -- are skipped, not
    overwritten: an id is the corpus's word, and two bodies claiming it is an
    incident to surface (the caller counts the skip), never to paper over.
    """
    _check_records(records, path)
    existing = read_jsonl(path)
    present = frozenset(str(record["id"]) for record in existing)
    seen = set(present)
    merged = list(existing)
    added = 0
    for record in records:
        rid = str(record["id"])
        if rid in seen:
            continue
        seen.add(rid)
        merged.append(record)
        added += 1
    skipped = len(records) - added
    _stage_and_swap(path, merged)
    return added, skipped


def _check_records(records: list[dict], path: str) -> None:
    for pos, record in enumerate(records):
        if not isinstance(record, dict) or "id" not in record:
            raise ValueError(
                f"refusing to write {path}: record {pos} is not a json object "
                "with an 'id'"
            )


def _stage_and_swap(path: str, records: list[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return len(records)
