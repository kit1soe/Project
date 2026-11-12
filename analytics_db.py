"""Persistent SQLite analytics for NS ADC simulation runs.

The recorder keeps the complete in-memory snapshots instead of reducing a run
to a few counters.  This makes the database useful for later experiments,
branch comparison, failure analysis and plotting without rerunning a scenario.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    architecture TEXT,
    topology_json TEXT NOT NULL,
    q_required INTEGER,
    routing_mode INTEGER,
    total_initial_requests INTEGER,
    single_use INTEGER,
    generation_ticks_json TEXT,
    app_version TEXT
);

CREATE TABLE IF NOT EXISTS ticks (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    history_index INTEGER NOT NULL,
    tick INTEGER NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0,
    active_count INTEGER,
    completed_count INTEGER,
    rejected_count INTEGER,
    total_generated INTEGER,
    next_input_idx INTEGER,
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY (run_id, history_index)
);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    history_index INTEGER NOT NULL,
    tick INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    history_index INTEGER NOT NULL,
    request_id INTEGER NOT NULL,
    created_tick INTEGER,
    completed_tick INTEGER,
    status TEXT,
    status_code TEXT,
    q_required INTEGER,
    q_remaining INTEGER,
    service_time INTEGER,
    remaining_service_time INTEGER,
    origin_row INTEGER,
    origin_col INTEGER,
    path_json TEXT,
    color TEXT,
    rejection_reason TEXT,
    PRIMARY KEY (run_id, history_index, request_id)
);

CREATE TABLE IF NOT EXISTS neuron_states (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    history_index INTEGER NOT NULL,
    tick INTEGER NOT NULL,
    row INTEGER NOT NULL,
    col INTEGER NOT NULL,
    state TEXT,
    state_code TEXT,
    is_operational INTEGER,
    pending_fault INTEGER,
    remaining_time INTEGER,
    q_stage INTEGER,
    request_id INTEGER,
    processed_requests_count INTEGER,
    unprocessed_requests_count INTEGER,
    trail_request_id INTEGER,
    has_computed INTEGER,
    PRIMARY KEY (run_id, history_index, row, col)
);

CREATE TABLE IF NOT EXISTS topology_nodes (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    row INTEGER NOT NULL,
    col INTEGER NOT NULL,
    label TEXT,
    x REAL,
    y REAL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (run_id, row, col)
);

CREATE TABLE IF NOT EXISTS topology_edges (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    edge_index INTEGER NOT NULL,
    from_row INTEGER,
    from_col INTEGER,
    to_row INTEGER,
    to_col INTEGER,
    edge_type TEXT,
    color TEXT,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (run_id, edge_index)
);

CREATE TABLE IF NOT EXISTS actions (
    action_id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    history_index INTEGER,
    tick INTEGER,
    action_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticks_run_order ON ticks(run_id, history_index);
CREATE INDEX IF NOT EXISTS idx_events_run_tick ON events(run_id, tick);
CREATE INDEX IF NOT EXISTS idx_events_run_type ON events(run_id, event_type);
CREATE INDEX IF NOT EXISTS idx_requests_run_status ON requests(run_id, status_code);
CREATE INDEX IF NOT EXISTS idx_neurons_run_address ON neuron_states(run_id, row, col);
CREATE INDEX IF NOT EXISTS idx_actions_run_tick ON actions(run_id, tick);
"""


def _plain(value: Any) -> Any:
    """Convert model objects and enums to JSON-friendly values."""
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return _plain(value.value)
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _json(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, separators=(",", ":"))


def _address(value: Any) -> tuple[Optional[int], Optional[int]]:
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[0], value[1]
    return None, None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


class AnalyticsRecorder:
    """Write live simulation state into a durable SQLite database."""

    def __init__(self, db_path: str | Path, *, replace: bool = False):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = _connect(self.db_path)
        self.run_id: Optional[int] = None
        if replace:
            self.clear()

    def clear(self) -> None:
        """Remove recorded data while keeping the schema."""
        for table in (
            "actions", "topology_edges", "topology_nodes", "neuron_states",
            "requests", "events", "ticks", "runs"
        ):
            self.db.execute(f"DELETE FROM {table}")
        self.db.commit()

    def start_run(self, architecture: Any, *, label: Optional[str] = None) -> int:
        """Create a new run and persist its initial topology and snapshot."""
        topology = _plain(architecture.get_topology_info())
        architecture_name = label or topology.get("type") or type(architecture).__name__
        cursor = self.db.execute(
            """
            INSERT INTO runs(
                created_at, architecture, topology_json, q_required, routing_mode,
                total_initial_requests, single_use, generation_ticks_json, app_version
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                _now(),
                architecture_name,
                _json(topology),
                getattr(architecture, "q_required", None),
                int(bool(getattr(architecture, "return_to_origin", True))),
                getattr(architecture, "total_initial_requests", None),
                int(bool(getattr(architecture, "single_use", False))),
                _json(getattr(architecture, "generation_ticks", None)),
                "ns-adc-analytics-v2",
            ),
        )
        self.run_id = int(cursor.lastrowid)
        self._write_topology(topology)
        self.record_snapshot(architecture, action_type="run_start")
        return self.run_id

    def _write_topology(self, topology: dict[str, Any]) -> None:
        if self.run_id is None:
            return
        for node in topology.get("nodes", []):
            row, col = _address(node.get("address"))
            if row is None or col is None:
                continue
            metadata = {k: v for k, v in node.items() if k not in {"address", "label", "x", "y"}}
            self.db.execute(
                """
                INSERT OR REPLACE INTO topology_nodes
                (run_id,row,col,label,x,y,metadata_json) VALUES(?,?,?,?,?,?,?)
                """,
                (self.run_id, row, col, node.get("label"), node.get("x"), node.get("y"), _json(metadata)),
            )
        for edge_index, edge in enumerate(topology.get("edges", [])):
            from_row, from_col = _address(edge.get("from"))
            to_row, to_col = _address(edge.get("to"))
            metadata = {k: v for k, v in edge.items() if k not in {"from", "to", "type", "color"}}
            self.db.execute(
                """
                INSERT OR REPLACE INTO topology_edges
                (run_id,edge_index,from_row,from_col,to_row,to_col,edge_type,color,metadata_json)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    self.run_id, edge_index, from_row, from_col, to_row, to_col,
                    edge.get("type"), edge.get("color"), _json(metadata),
                ),
            )

    def record_action(
        self,
        architecture: Any,
        action_type: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.run_id is None:
            self.start_run(architecture)
        self.db.execute(
            """
            INSERT INTO actions(run_id,history_index,tick,action_type,payload_json,created_at)
            VALUES(?,?,?,?,?,?)
            """,
            (
                self.run_id,
                getattr(architecture, "current_history_index", None),
                getattr(architecture, "current_tick", None),
                action_type,
                _json(payload or {}),
                _now(),
            ),
        )
        self.db.commit()

    def record_snapshot(
        self,
        architecture: Any,
        *,
        action_type: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """Upsert the current history branch and all detailed state tables."""
        if self.run_id is None:
            self.start_run(architecture)
            return
        history = getattr(architecture, "history", [])
        history_index = int(getattr(architecture, "current_history_index", 0))
        if not history or history_index >= len(history):
            return

        snapshot = history[history_index]
        snapshot_data = _plain(snapshot)
        current_tick = int(getattr(snapshot, "tick", getattr(architecture, "current_tick", 0)))

        self.db.execute("UPDATE ticks SET is_current=0 WHERE run_id=?", (self.run_id,))
        for table in ("events", "requests", "neuron_states"):
            self.db.execute(
                f"DELETE FROM {table} WHERE run_id=? AND history_index=?",
                (self.run_id, history_index),
            )

        self.db.execute(
            """
            INSERT OR REPLACE INTO ticks(
                run_id,history_index,tick,is_current,active_count,completed_count,
                rejected_count,total_generated,next_input_idx,snapshot_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                self.run_id,
                history_index,
                current_tick,
                1,
                getattr(snapshot, "active_count", None),
                getattr(snapshot, "completed_count", None),
                getattr(snapshot, "rejected_count", None),
                getattr(snapshot, "total_generated", None),
                getattr(snapshot, "next_input_idx", None),
                _json(snapshot_data),
            ),
        )

        for event in getattr(snapshot, "new_logs", []) or []:
            event_data = _plain(event)
            self.db.execute(
                "INSERT INTO events(run_id,history_index,tick,event_type,message) VALUES(?,?,?,?,?)",
                (
                    self.run_id,
                    history_index,
                    int(event_data.get("tick", current_tick)),
                    event_data.get("type", "INFO"),
                    event_data.get("msg", ""),
                ),
            )

        for request in getattr(snapshot, "requests", []) or []:
            request_data = _plain(request)
            origin_row, origin_col = _address(getattr(request, "input_neuron_index", None))
            status = getattr(request, "status", None)
            self.db.execute(
                """
                INSERT OR REPLACE INTO requests(
                    run_id,history_index,request_id,created_tick,completed_tick,status,status_code,
                    q_required,q_remaining,service_time,remaining_service_time,origin_row,origin_col,
                    path_json,color,rejection_reason
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    self.run_id,
                    history_index,
                    request_data.get("id"),
                    request_data.get("created_at_tick"),
                    request_data.get("completed_at_tick"),
                    getattr(status, "value", str(status) if status is not None else None),
                    getattr(status, "name", None),
                    request_data.get("q_required"),
                    request_data.get("q_remaining"),
                    request_data.get("service_time"),
                    request_data.get("remaining_service_time"),
                    origin_row,
                    origin_col,
                    _json(getattr(request, "path", [])),
                    request_data.get("color"),
                    request_data.get("rejection_reason"),
                ),
            )

        for neuron in getattr(architecture, "all_neurons", lambda: [])():
            address = getattr(neuron, "address", (None, None))
            row, col = _address(address)
            if row is None or col is None:
                continue
            state = getattr(neuron, "state", None)
            current_request = getattr(neuron, "current_request", None)
            self.db.execute(
                """
                INSERT OR REPLACE INTO neuron_states(
                    run_id,history_index,tick,row,col,state,state_code,is_operational,pending_fault,
                    remaining_time,q_stage,request_id,processed_requests_count,
                    unprocessed_requests_count,trail_request_id,has_computed
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    self.run_id,
                    history_index,
                    current_tick,
                    row,
                    col,
                    getattr(state, "value", str(state) if state is not None else None),
                    getattr(state, "name", None),
                    int(bool(getattr(neuron, "is_operational", True))),
                    int(bool(getattr(neuron, "pending_fault", False))),
                    getattr(neuron, "remaining_time", 0),
                    getattr(neuron, "q_stage", 0),
                    getattr(current_request, "id", None),
                    getattr(neuron, "processed_requests_count", 0),
                    getattr(neuron, "unprocessed_requests_count", 0),
                    getattr(neuron, "trail_request_id", None),
                    int(bool(getattr(neuron, "has_computed", False))),
                ),
            )

        if action_type:
            self.db.execute(
                """
                INSERT INTO actions(run_id,history_index,tick,action_type,payload_json,created_at)
                VALUES(?,?,?,?,?,?)
                """,
                (self.run_id, history_index, current_tick, action_type, _json(payload or {}), _now()),
            )
        if getattr(architecture, "is_finished", lambda: False)():
            self.db.execute("UPDATE runs SET finished_at=? WHERE run_id=?", (_now(), self.run_id))
        self.db.commit()

    def _current_history_index(self, run_id: Optional[int] = None) -> Optional[int]:
        run_id = run_id or self.run_id
        row = self.db.execute(
            "SELECT history_index FROM ticks WHERE run_id=? AND is_current=1 ORDER BY history_index DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return int(row[0]) if row else None

    def latest_run_id(self) -> Optional[int]:
        row = self.db.execute("SELECT MAX(run_id) FROM runs").fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def get_run_summary(self, run_id: Optional[int] = None) -> dict[str, Any]:
        run_id = run_id or self.run_id
        if run_id is None:
            return {}
        run = self.db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not run:
            return {}
        counts = self.db.execute(
            "SELECT COUNT(*) AS requests, SUM(status_code='COMPLETED') AS completed, "
            "SUM(status_code='REJECTED') AS rejected FROM requests WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return {
            "run_id": run_id,
            "architecture": run["architecture"],
            "created_at": run["created_at"],
            "finished_at": run["finished_at"],
            "ticks": self.db.execute("SELECT COUNT(*) FROM ticks WHERE run_id=?", (run_id,)).fetchone()[0],
            "events": self.db.execute("SELECT COUNT(*) FROM events WHERE run_id=?", (run_id,)).fetchone()[0],
            "neurons": self.db.execute("SELECT COUNT(DISTINCT row || ':' || col) FROM neuron_states WHERE run_id=?", (run_id,)).fetchone()[0],
            "requests": counts["requests"] or 0,
            "completed": counts["completed"] or 0,
            "rejected": counts["rejected"] or 0,
        }

    def get_tick_metrics(self, run_id: Optional[int] = None) -> list[dict[str, Any]]:
        run_id = run_id or self.run_id
        if run_id is None:
            return []
        rows = self.db.execute(
            """
            SELECT history_index,tick,active_count,completed_count,rejected_count,total_generated,next_input_idx
            FROM ticks WHERE run_id=? ORDER BY history_index
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_event_counts(self, run_id: Optional[int] = None) -> list[tuple[str, int]]:
        run_id = run_id or self.run_id
        if run_id is None:
            return []
        rows = self.db.execute(
            "SELECT event_type,COUNT(*) AS count FROM events WHERE run_id=? GROUP BY event_type ORDER BY count DESC,event_type",
            (run_id,),
        ).fetchall()
        return [(str(row["event_type"]), int(row["count"])) for row in rows]

    def get_request_outcomes(self, run_id: Optional[int] = None) -> list[tuple[str, int]]:
        run_id = run_id or self.run_id
        if run_id is None:
            return []
        current_idx = self._current_history_index(run_id)
        if current_idx is None:
            return []
        rows = self.db.execute(
            "SELECT COALESCE(status_code,'UNKNOWN') AS status_code,COUNT(*) AS count "
            "FROM requests WHERE run_id=? AND history_index=? GROUP BY status_code ORDER BY count DESC",
            (run_id, current_idx),
        ).fetchall()
        return [(str(row["status_code"]), int(row["count"])) for row in rows]

    def get_neuron_heatmap(self, run_id: Optional[int] = None) -> list[dict[str, Any]]:
        run_id = run_id or self.run_id
        if run_id is None:
            return []
        current_idx = self._current_history_index(run_id)
        if current_idx is None:
            return []
        rows = self.db.execute(
            """
            SELECT row,col,state,state_code,is_operational,pending_fault,processed_requests_count,
                   unprocessed_requests_count,request_id
            FROM neuron_states WHERE run_id=? AND history_index=? ORDER BY row,col
            """,
            (run_id, current_idx),
        ).fetchall()
        return [dict(row) for row in rows]

    def export_csv(self, directory: str | Path, run_id: Optional[int] = None) -> list[Path]:
        """Export detailed tables for a run as CSV files."""
        run_id = run_id or self.run_id
        if run_id is None:
            return []
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        exports = {
            "ticks": "SELECT * FROM ticks WHERE run_id=? ORDER BY history_index",
            "events": "SELECT * FROM events WHERE run_id=? ORDER BY event_id",
            "requests": "SELECT * FROM requests WHERE run_id=? ORDER BY history_index,request_id",
            "neuron_states": "SELECT * FROM neuron_states WHERE run_id=? ORDER BY history_index,row,col",
            "actions": "SELECT * FROM actions WHERE run_id=? ORDER BY action_id",
        }
        paths: list[Path] = []
        for name, query in exports.items():
            rows = self.db.execute(query, (run_id,)).fetchall()
            path = output / f"run_{run_id}_{name}.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                if rows:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(dict(row) for row in rows)
                else:
                    handle.write("\n")
            paths.append(path)
        return paths

    def close(self) -> None:
        if self.db:
            self.db.commit()
            self.db.close()

    def __enter__(self) -> "AnalyticsRecorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def export_architecture(architecture: Any, db_path: str | Path, *, replace: bool = False) -> int:
    """Export one in-memory architecture run into SQLite and return run_id."""
    with AnalyticsRecorder(db_path, replace=replace) as recorder:
        return recorder.start_run(architecture)


def event_counts(db_path: str | Path, run_id: Optional[int] = None) -> list[tuple[str, int]]:
    """Compatibility helper used by the command-line inspector."""
    with AnalyticsRecorder(db_path) as recorder:
        return recorder.get_event_counts(run_id or recorder.latest_run_id())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect exported NS ADC simulation analytics")
    parser.add_argument("db", help="SQLite database path")
    parser.add_argument("--run", type=int, dest="run_id")
    parser.add_argument("--csv", type=Path, dest="csv_dir", help="Export detailed CSV files to a directory")
    args = parser.parse_args()

    with AnalyticsRecorder(args.db) as recorder:
        run_id = args.run_id or recorder.latest_run_id()
        for event_type, count in recorder.get_event_counts(run_id):
            print(f"{event_type}\t{count}")
        if args.csv_dir and run_id:
            for path in recorder.export_csv(args.csv_dir, run_id):
                print(path)


if __name__ == "__main__":
    main()
