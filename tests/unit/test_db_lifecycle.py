"""Tests for v0.4 DB lifecycle: size / flush / backup / load / inspect.

Covers:
- ``backend.file_sizes`` / ``checkpoint_wal``
- ``backend.flush`` — full + date-bounded, all three time axes
- ``backend.backup_to`` — full + date-bounded
- ``backend.merge_from`` — INSERT-OR-IGNORE idempotency
- ``backend.open_readonly`` — read-only guard
- ``Memory.db_size`` / ``flush`` / ``backup_to`` / ``load_from`` / ``inspect_file``
- ``thought db size`` / ``flush`` / ``backup`` / ``load`` / ``inspect`` CLI
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from thought.cli import app
from thought.memory import Memory
from thought.models import ScopeFilter
from thought.storage.sqlite.backend import SQLiteBackend

# Counter to keep entity names distinct across _seed() calls within one test —
# the entity canonical-key dedup would otherwise reuse a row from an earlier
# batch, defeating the point of seeding "at different times".
_seed_counter = [0]


def _seed(mem: Memory, *, now: datetime, n: int = 4, scope: str = "shared") -> None:
    """Seed `n` entities at `now`, plus one edge between the first two.

    Names include a monotonic salt so repeated calls don't collide on the
    canonical-name dedup, which would defeat date-filter tests.
    """
    _seed_counter[0] += 1
    salt = _seed_counter[0]
    src = mem._backend.upsert_source(f"seed#{salt} @ {now.isoformat()}")
    eids = []
    for i in range(n):
        eid = mem._backend.upsert_entity(
            type_="PERSON", name=f"P{salt}_{i}",
            scope=scope, valid_from=now, learned_at=now, source_ref=src,
        )
        eids.append(eid)
    if len(eids) >= 2:
        mem._backend.upsert_edge(
            source_id=eids[0], target_id=eids[1],
            relation_type="WORKS_WITH", source_ref=src,
            confidence_score=0.9, valid_from=now, learned_at=now,
        )


@pytest.fixture()
def mem(tmp_path):
    m = Memory.open(
        db_path=str(tmp_path / "lc.db"),
        embedder_choice="deterministic", embedder_dim=64,
    )
    yield m
    m.close()


# ---------------------------------------------------------------- file_sizes

def test_file_sizes_zero_for_empty_db(tmp_path) -> None:
    # On a fresh DB the file might be 0 or small; main is the only field that
    # should be > 0 after migrate. WAL/SHM may or may not exist yet.
    m = Memory.open(
        db_path=str(tmp_path / "fs.db"),
        embedder_choice="deterministic", embedder_dim=64,
    )
    try:
        sizes = m._backend.file_sizes()
        assert sizes["main"] > 0
        assert sizes["total_bytes"] == sizes["main"] + sizes["wal"] + sizes["shm"]
    finally:
        m.close()


def test_file_sizes_grows_after_writes(mem) -> None:
    before = mem._backend.file_sizes()["total_bytes"]
    _seed(mem, now=datetime.now(UTC), n=20)
    after = mem._backend.file_sizes()["total_bytes"]
    assert after >= before  # WAL absorbs writes; main may or may not grow until checkpoint


def test_close_truncates_wal(tmp_path) -> None:
    """``close()`` runs ``PRAGMA wal_checkpoint(TRUNCATE)`` so subsequent
    reopen + ``file_sizes`` doesn't show stale WAL data."""
    db_path = str(tmp_path / "wal.db")
    m1 = Memory.open(
        db_path=db_path, embedder_choice="deterministic", embedder_dim=64,
    )
    _seed(m1, now=datetime.now(UTC), n=10)
    m1.close()
    wal = Path(db_path + "-wal")
    assert wal.stat().st_size if wal.exists() else 0 == 0


# ---------------------------------------------------------------- flush (full)

def test_full_flush_clears_everything(mem) -> None:
    _seed(mem, now=datetime.now(UTC), n=5)
    counts = mem._backend.flush()
    # Returns prior-state counts (what got wiped).
    assert counts["entities"] >= 5
    # Tables exist and are empty post-flush.
    assert mem.stats()["entities_total"] == 0
    assert mem.stats()["edges_total"] == 0


def test_full_flush_preserves_schema(mem) -> None:
    _seed(mem, now=datetime.now(UTC), n=3)
    mem._backend.flush()
    # Migrations re-ran; we can still write.
    _seed(mem, now=datetime.now(UTC), n=2)
    assert mem.stats()["entities_total"] >= 2


# ---------------------------------------------------------------- flush (date-bounded)

def test_flush_before_deletes_only_old(mem) -> None:
    # valid_from is caller-supplied so we can position entities on the
    # time axis deterministically; created_at is set by the backend's
    # wall clock and can't be backdated from tests.
    old = datetime.now(UTC) - timedelta(days=10)
    new = datetime.now(UTC)
    _seed(mem, now=old, n=3)
    _seed(mem, now=new, n=2)
    cut = old + timedelta(days=5)
    result = mem._backend.flush(before=cut, time_axis="valid")
    assert result["entities"] == 3
    # New entities survive.
    assert mem.stats()["entities_total"] >= 2


def test_flush_since_deletes_only_new(mem) -> None:
    old = datetime.now(UTC) - timedelta(days=10)
    new = datetime.now(UTC)
    _seed(mem, now=old, n=3)
    _seed(mem, now=new, n=2)
    cut = old + timedelta(days=5)
    result = mem._backend.flush(since=cut, time_axis="valid")
    assert result["entities"] == 2
    assert mem.stats()["entities_total"] == 3


def test_flush_range_keeps_middle(mem) -> None:
    t0 = datetime.now(UTC) - timedelta(days=30)
    t1 = datetime.now(UTC) - timedelta(days=20)
    t2 = datetime.now(UTC) - timedelta(days=10)
    _seed(mem, now=t0, n=2)
    _seed(mem, now=t1, n=3)
    _seed(mem, now=t2, n=4)
    # Delete everything OUTSIDE [t1, t2). Two flush calls compose the range.
    mem._backend.flush(before=t1, time_axis="valid")
    mem._backend.flush(since=t2, time_axis="valid")
    # t1 batch (3) survives.
    assert mem.stats()["entities_total"] == 3


def test_flush_time_axis_valid(mem) -> None:
    """``valid_from`` filter — world-time axis."""
    now_ts = datetime.now(UTC)
    # Manually upsert with valid_from in the past but created_at = now (default).
    past = now_ts - timedelta(days=100)
    src = mem._backend.upsert_source("axis test")
    mem._backend.upsert_entity(
        type_="PERSON", name="OldFact", scope="shared",
        valid_from=past, learned_at=now_ts, source_ref=src,
    )
    mem._backend.upsert_entity(
        type_="PERSON", name="NewFact", scope="shared",
        valid_from=now_ts, learned_at=now_ts, source_ref=src,
    )
    cut = past + timedelta(days=50)
    result = mem._backend.flush(before=cut, time_axis="valid")
    assert result["entities"] == 1
    assert mem.stats()["entities_total"] == 1


def test_flush_invalid_time_axis_raises(mem) -> None:
    with pytest.raises(ValueError, match="time_axis"):
        mem._backend.flush(time_axis="nonsense")


# ---------------------------------------------------------------- backup + load round-trip

def test_full_backup_load_round_trip(tmp_path) -> None:
    src_path = str(tmp_path / "src.db")
    snap = tmp_path / "snap.db"

    src_mem = Memory.open(
        db_path=src_path, embedder_choice="deterministic", embedder_dim=64,
    )
    _seed(src_mem, now=datetime.now(UTC), n=5)
    src_stats = src_mem.stats()
    src_mem.backup_to(snap)
    src_mem.close()

    # Snapshot exists, has size, is valid SQLite.
    assert snap.exists() and snap.stat().st_size > 0

    # Open the snapshot as a Memory and verify contents match.
    snap_mem = Memory.open(
        db_path=str(snap), embedder_choice="deterministic", embedder_dim=64,
    )
    try:
        assert snap_mem.stats()["entities_total"] == src_stats["entities_total"]
        assert snap_mem.stats()["edges_total"] == src_stats["edges_total"]
    finally:
        snap_mem.close()


def test_backup_refuses_overwrite_without_force(mem, tmp_path) -> None:
    _seed(mem, now=datetime.now(UTC), n=2)
    snap = tmp_path / "snap.db"
    snap.write_bytes(b"already exists")
    with pytest.raises(FileExistsError):
        mem.backup_to(snap)
    # With force=True it succeeds.
    bytes_written = mem.backup_to(snap, force=True)
    assert bytes_written > 0


def test_backup_with_date_filter(tmp_path) -> None:
    """Date-bounded backup contains only entities matching the filter."""
    src_path = str(tmp_path / "src2.db")
    snap = tmp_path / "old-only.db"

    src_mem = Memory.open(
        db_path=src_path, embedder_choice="deterministic", embedder_dim=64,
    )
    old = datetime.now(UTC) - timedelta(days=30)
    new = datetime.now(UTC)
    _seed(src_mem, now=old, n=3)
    _seed(src_mem, now=new, n=4)
    src_mem.backup_to(snap, before=old + timedelta(days=15), time_axis="valid")
    src_mem.close()

    snap_mem = Memory.open(
        db_path=str(snap), embedder_choice="deterministic", embedder_dim=64,
    )
    try:
        # Only the 3 old entities should be in the snapshot.
        assert snap_mem.stats()["entities_total"] == 3
    finally:
        snap_mem.close()


# ---------------------------------------------------------------- merge

def test_merge_idempotent(tmp_path) -> None:
    src_path = str(tmp_path / "src3.db")
    snap = tmp_path / "snap3.db"
    src_mem = Memory.open(
        db_path=src_path, embedder_choice="deterministic", embedder_dim=64,
    )
    _seed(src_mem, now=datetime.now(UTC), n=4)
    src_mem.backup_to(snap)
    src_mem.close()

    dest_path = str(tmp_path / "dest3.db")
    dest_mem = Memory.open(
        db_path=dest_path, embedder_choice="deterministic", embedder_dim=64,
    )
    try:
        r1 = dest_mem._backend.merge_from(snap)
        assert r1["new_entities"] >= 4
        # Second merge: everything is a duplicate.
        r2 = dest_mem._backend.merge_from(snap)
        assert r2["new_entities"] == 0
    finally:
        dest_mem.close()


def test_merge_with_date_filter(tmp_path) -> None:
    src_path = str(tmp_path / "src4.db")
    snap = tmp_path / "snap4.db"
    src_mem = Memory.open(
        db_path=src_path, embedder_choice="deterministic", embedder_dim=64,
    )
    old = datetime.now(UTC) - timedelta(days=30)
    new = datetime.now(UTC)
    _seed(src_mem, now=old, n=2)
    _seed(src_mem, now=new, n=3)
    src_mem.backup_to(snap)
    src_mem.close()

    dest_path = str(tmp_path / "dest4.db")
    dest_mem = Memory.open(
        db_path=dest_path, embedder_choice="deterministic", embedder_dim=64,
    )
    try:
        r = dest_mem._backend.merge_from(
            snap, since=old + timedelta(days=15), time_axis="valid",
        )
        # Only the 3 new entities should land.
        assert r["new_entities"] == 3
    finally:
        dest_mem.close()


# ---------------------------------------------------------------- open_readonly

def test_open_readonly_refuses_writes(tmp_path) -> None:
    src_mem = Memory.open(
        db_path=str(tmp_path / "ro.db"),
        embedder_choice="deterministic", embedder_dim=64,
    )
    _seed(src_mem, now=datetime.now(UTC), n=2)
    src_mem.close()

    ro = SQLiteBackend.open_readonly(tmp_path / "ro.db")
    try:
        # Reads work.
        rows = ro._conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()
        assert rows["n"] >= 2
        # Writes raise.
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            ro._conn.execute("DELETE FROM entities")
    finally:
        ro.close()


# ---------------------------------------------------------------- inspect_file

def test_inspect_file_returns_counts(mem, tmp_path) -> None:
    _seed(mem, now=datetime.now(UTC), n=5)
    snap = tmp_path / "ins.db"
    mem.backup_to(snap)
    data = mem.inspect_file(snap)
    assert data["entities_total"] >= 5
    assert data["edges"] >= 1
    assert data["size_bytes"] > 0
    assert "entity_types" not in data


def test_inspect_file_with_schema(mem, tmp_path) -> None:
    _seed(mem, now=datetime.now(UTC), n=3)
    snap = tmp_path / "ins2.db"
    mem.backup_to(snap)
    data = mem.inspect_file(snap, include_schema=True)
    assert "entity_types" in data
    assert "PERSON" in data["entity_types"]
    assert "relation_types" in data
    assert "WORKS_WITH" in data["relation_types"]


def test_inspect_file_missing(mem, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        mem.inspect_file(tmp_path / "does-not-exist.db")


# ---------------------------------------------------------------- Memory.load_from

def test_load_from_merge_round_trip(tmp_path) -> None:
    src_path = str(tmp_path / "lf_src.db")
    snap = tmp_path / "lf_snap.db"
    src_mem = Memory.open(
        db_path=src_path, embedder_choice="deterministic", embedder_dim=64,
    )
    _seed(src_mem, now=datetime.now(UTC), n=4)
    src_mem.backup_to(snap)
    src_mem.close()

    dest_path = str(tmp_path / "lf_dest.db")
    dest_mem = Memory.open(
        db_path=dest_path, embedder_choice="deterministic", embedder_dim=64,
    )
    try:
        result = dest_mem.load_from(snap, merge=True)
        assert result["action"] == "merge"
        assert result["new_entities"] >= 4
    finally:
        dest_mem.close()


def test_load_from_replace_returns_metadata(tmp_path) -> None:
    src_path = str(tmp_path / "lf_src2.db")
    snap = tmp_path / "lf_snap2.db"
    src_mem = Memory.open(
        db_path=src_path, embedder_choice="deterministic", embedder_dim=64,
    )
    _seed(src_mem, now=datetime.now(UTC), n=2)
    src_mem.backup_to(snap)
    src_mem.close()

    dest_path = str(tmp_path / "lf_dest2.db")
    dest_mem = Memory.open(
        db_path=dest_path, embedder_choice="deterministic", embedder_dim=64,
    )
    try:
        result = dest_mem.load_from(snap, merge=False)
        # Replace doesn't actually move files (CLI does); facade returns metadata.
        assert result["action"] == "replace"
        assert result["source"].endswith("lf_snap2.db")
        assert result["size"] > 0
    finally:
        dest_mem.close()


def test_load_from_missing_file(mem, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        mem.load_from(tmp_path / "missing.db")


def test_flush_without_confirm_raises(mem) -> None:
    with pytest.raises(ValueError, match="confirm=True"):
        mem.flush(confirm=False)


# ---------------------------------------------------------------- CLI

def _cli_init(runner: CliRunner, workspace: Path) -> None:
    r = runner.invoke(
        app,
        ["init", "--db-path", str(workspace / ".thought" / "t.db"),
         "--embedder", "deterministic", "--quick", "--no-claude-md"],
    )
    assert r.exit_code == 0, r.stdout


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_db_size_cli(workspace) -> None:
    runner = CliRunner()
    _cli_init(runner, workspace)
    runner.invoke(app, ["ingest", "Alice owns Acme.", "--scope", "shared"])
    r = runner.invoke(app, ["db", "size"])
    assert r.exit_code == 0, r.stdout
    assert "Database on-disk size" in r.stdout or "total" in r.stdout.lower()


def test_db_size_cli_json(workspace) -> None:
    runner = CliRunner()
    _cli_init(runner, workspace)
    r = runner.invoke(app, ["db", "size", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert "main" in data
    assert "total_bytes" in data


def test_db_flush_cli_with_yes(workspace) -> None:
    runner = CliRunner()
    _cli_init(runner, workspace)
    runner.invoke(app, ["ingest", "Alice owns Acme.", "--scope", "shared"])
    r = runner.invoke(app, ["db", "flush", "--yes"])
    assert r.exit_code == 0, r.stdout
    assert "Flush summary" in r.stdout or "entities" in r.stdout.lower()


def test_db_backup_load_inspect_cli_round_trip(workspace) -> None:
    runner = CliRunner()
    _cli_init(runner, workspace)
    runner.invoke(app, ["ingest", "Alice owns Acme.", "--scope", "shared"])
    snap = workspace / "snap.db"
    r = runner.invoke(app, ["db", "backup", str(snap)])
    assert r.exit_code == 0, r.stdout
    assert snap.exists() and snap.stat().st_size > 0

    r = runner.invoke(app, ["db", "inspect", str(snap), "--json"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(r.stdout)
    assert data["entities_total"] >= 1

    r = runner.invoke(app, ["db", "load", str(snap), "--yes", "--merge"])
    assert r.exit_code == 0, r.stdout


def test_db_backup_refuses_overwrite_without_force(workspace) -> None:
    runner = CliRunner()
    _cli_init(runner, workspace)
    runner.invoke(app, ["ingest", "x", "--scope", "shared"])
    snap = workspace / "exists.db"
    runner.invoke(app, ["db", "backup", str(snap)])
    assert snap.exists()
    r = runner.invoke(app, ["db", "backup", str(snap)])
    assert r.exit_code == 1


def test_db_inspect_missing_file(workspace) -> None:
    runner = CliRunner()
    _cli_init(runner, workspace)
    r = runner.invoke(app, ["db", "inspect", str(workspace / "nope.db")])
    assert r.exit_code == 1


def test_db_flush_invalid_time_axis(workspace) -> None:
    runner = CliRunner()
    _cli_init(runner, workspace)
    r = runner.invoke(app, ["db", "flush", "--yes", "--time-axis", "bogus"])
    assert r.exit_code == 2


# Suppress unused-warning for ScopeFilter import used implicitly by Memory.
_ = ScopeFilter
