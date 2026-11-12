"""SQLite export and queries for simulation logs and history snapshots."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    architecture TEXT,
    topology_json TEXT NOT NULL,
    q_required INTEGER,
    routing_mode TEXT,
    total_initial_requests INTEGER
);
CREATE TABLE IF NOT EXISTS ticks (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    tick INTEGER NOT NULL,
    active_count INTEGER,
    completed_count INTEGER,
    rejected_count INTEGER,
    total_generated INTEGER,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (run_id, tick)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    tick INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
    run_id INTEGER NOT NULL REFERENCES runs(run_id),
    request_id INTEGER NOT NULL,
    created_tick INTEGER,
    completed_tick INTEGER,
    status TEXT,
    origin_row INTEGER,
    origin_col INTEGER,
    path_json TEXT,
    PRIMARY KEY (run_id, request_id)
);
CREATE INDEX IF NOT EXISTS idx_events_run_tick ON events(run_id, tick);
CREATE INDEX IF NOT EXISTS idx_ticks_run_tick ON ticks(run_id, tick);
"""


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def export_architecture(architecture: Any, db_path: str | Path, *, replace: bool = False) -> int:
    """Export one in-memory architecture run into SQLite and return run_id."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        if replace:
            db.execute("DELETE FROM events")
            db.execute("DELETE FROM requests")
            db.execute("DELETE FROM ticks")
            db.execute("DELETE FROM runs")

        topo = _plain(architecture.get_topology_info())
        db.execute(
            "INSERT INTO runs(created_at, architecture, topology_json, q_required, routing_mode, total_initial_requests) VALUES(?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), topo.get("type"), json.dumps(topo, ensure_ascii=False),
             getattr(architecture, "q_required", None), getattr(architecture, "return_to_origin", None),
             getattr(architecture, "total_initial_requests", None)),
        )
        run_id = int(db.execute("SELECT last_insert_rowid()").fetchone()[0])

        for snapshot in getattr(architecture, "history", []):
            snap = _plain(snapshot)
            tick = int(snap.get("tick", 0))
            db.execute(
                "INSERT OR REPLACE INTO ticks VALUES(?,?,?,?,?,?,?)",
                (run_id, tick, snap.get("active_count"), snap.get("completed_count"),
                 snap.get("rejected_count"), snap.get("total_generated"), json.dumps(snap, ensure_ascii=False)),
            )
            for event in snap.get("new_logs", []):
                db.execute("INSERT INTO events(run_id,tick,event_type,message) VALUES(?,?,?,?)",
                           (run_id, tick, event.get("type", "INFO"), event.get("msg", "")))

            for req in snap.get("requests", []):
                origin = req.get("origin") or req.get("input_neuron_index") or [None, None]
                path_json = json.dumps(req.get("path", []), ensure_ascii=False)
                db.execute(
                    "INSERT OR REPLACE INTO requests VALUES(?,?,?,?,?,?,?,?)",
                    (run_id, req.get("id"), req.get("created_at_tick"), req.get("completed_at_tick"),
                     str(req.get("status")) if req.get("status") is not None else None,
                     origin[0] if len(origin) > 0 else None, origin[1] if len(origin) > 1 else None, path_json),
                )
        return run_id


def event_counts(db_path: str | Path, run_id: Optional[int] = None) -> list[tuple]:
    with sqlite3.connect(db_path) as db:
        query = "SELECT event_type, COUNT(*) FROM events"
        args: tuple = ()
        if run_id is not None:
            query += " WHERE run_id=?"
            args = (run_id,)
        query += " GROUP BY event_type ORDER BY COUNT(*) DESC"
        return db.execute(query, args).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect exported simulation analytics")
    parser.add_argument("db", help="SQLite database path")
    parser.add_argument("--run", type=int, dest="run_id")
    args = parser.parse_args()
    for event_type, count in event_counts(args.db, args.run_id):
        print(f"{event_type}\t{count}")


if __name__ == "__main__":
    main()
