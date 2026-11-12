"""Generate reproducible analytics data for every supported topology.

The command appends synthetic but model-generated runs to
``analytics/simulation.sqlite``.  It is intentionally deterministic so that
the same database can be rebuilt for demos, screenshots and regression checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from analytics_db import AnalyticsRecorder
from models import (
    BaseArchitecture,
    ChordalFunnel8Architecture,
    Funnel8Architecture,
    HoneycombPipelineArchitecture,
    PyramidalArchitecture,
    RhomboidArchitecture,
    Star8Architecture,
    SymmetricStar8Architecture,
    ToroidalGridArchitecture,
)
from controller import SimulationController


Factory = Callable[[int, int, bool], BaseArchitecture]


def _factories() -> list[tuple[str, Factory]]:
    return [
        (
            "torus",
            lambda q, n, return_to_origin: ToroidalGridArchitecture(
                rows=6, cols=6, q_required=q, total_initial_requests=n,
                return_to_origin=return_to_origin,
            ),
        ),
        (
            "pyramid",
            lambda q, n, return_to_origin: PyramidalArchitecture(
                widths=[4, 5, 6, 7, 8], q_required=q,
                total_initial_requests=n, return_to_origin=return_to_origin,
            ),
        ),
        (
            "rhombus",
            lambda q, n, return_to_origin: RhomboidArchitecture(
                widths=[4, 6, 8, 6, 4], q_required=q,
                total_initial_requests=n, return_to_origin=return_to_origin,
            ),
        ),
        (
            "star8_4in",
            lambda q, n, return_to_origin: Star8Architecture(
                ray_depth=4, num_inputs=4, q_required=q, total_initial_requests=n,
            ),
        ),
        (
            "star8_6in",
            lambda q, n, return_to_origin: Star8Architecture(
                ray_depth=4, num_inputs=6, q_required=q, total_initial_requests=n,
            ),
        ),
        (
            "honeycomb",
            lambda q, n, return_to_origin: HoneycombPipelineArchitecture(
                rows=5, cols=6, q_required=q, total_initial_requests=n,
            ),
        ),
        (
            "sym_star8",
            lambda q, n, return_to_origin: SymmetricStar8Architecture(
                ray_depth=4, num_inputs=6, q_required=q, total_initial_requests=n,
            ),
        ),
        (
            "funnel8",
            lambda q, n, return_to_origin: Funnel8Architecture(
                q_required=q, total_initial_requests=n,
            ),
        ),
        (
            "chordal_funnel8",
            lambda q, n, return_to_origin: ChordalFunnel8Architecture(
                q_required=q, total_initial_requests=n,
            ),
        ),
    ]


def _record_until_finished(
    controller: SimulationController,
    recorder: AnalyticsRecorder,
    *,
    max_ticks: int = 160,
) -> int:
    steps = 0
    while not controller.arch.is_finished() and steps < max_ticks:
        controller.step_forward()
        recorder.record_snapshot(controller.arch)
        steps += 1
    return steps


def _fault_address(architecture: BaseArchitecture) -> tuple[int, int] | None:
    candidates = [
        node.get("address")
        for node in architecture.get_topology_info().get("nodes", [])
        if node.get("address") and node["address"][0] > 0
    ]
    if not candidates:
        candidates = [node.get("address") for node in architecture.get_topology_info().get("nodes", [])]
    address = candidates[0] if candidates else None
    return tuple(address) if address else None


def generate(db_path: str | Path, *, replace: bool = False) -> list[dict[str, int | str]]:
    results: list[dict[str, int | str]] = []
    with AnalyticsRecorder(db_path, replace=replace) as recorder:
        for name, factory in _factories():
            # Baseline: normal routing with no injected faults.
            baseline = factory(3, 8, True)
            run_id = recorder.start_run(baseline, label=f"{name}:baseline")
            steps = _record_until_finished(SimulationController(baseline), recorder)
            results.append({"run_id": run_id, "scenario": f"{name}:baseline", "steps": steps})

            # Fault run: one busy-neuron fault is deferred, then the system is
            # allowed to finish with the recorded failure visible in the DB.
            faulty = factory(3, 8, False)
            controller = SimulationController(faulty)
            run_id = recorder.start_run(faulty, label=f"{name}:fault")
            controller.step_forward()
            recorder.record_snapshot(faulty)
            address = _fault_address(faulty)
            if address:
                controller.toggle_fault(*address)
                recorder.record_action(faulty, "generated_fault", {"address": list(address), "seed": 20261112})
                recorder.record_snapshot(faulty)
            steps = 1 + _record_until_finished(controller, recorder)
            results.append({"run_id": run_id, "scenario": f"{name}:fault", "steps": steps})

            # Branch run: walk back in the Memento history, inject a different
            # fault and continue.  Both branches remain queryable by index.
            branched = factory(4, 6, True)
            controller = SimulationController(branched)
            run_id = recorder.start_run(branched, label=f"{name}:branch")
            for _ in range(5):
                controller.step_forward()
                recorder.record_snapshot(branched)
            rewind_index = max(1, branched.current_history_index - 2)
            controller.go_to_tick(rewind_index)
            recorder.record_action(branched, "history_rewind", {"history_index": rewind_index})
            recorder.record_snapshot(branched)
            address = _fault_address(branched)
            if address:
                controller.toggle_fault(*address)
                recorder.record_action(branched, "branch_fault", {"address": list(address)})
                recorder.record_snapshot(branched)
            steps = 5 + _record_until_finished(controller, recorder)
            results.append({"run_id": run_id, "scenario": f"{name}:branch", "steps": steps})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NS ADC analytics data for all topologies")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).with_name("analytics") / "simulation.sqlite",
        help="SQLite output path",
    )
    parser.add_argument("--replace", action="store_true", help="Clear the database before generating")
    args = parser.parse_args()
    for result in generate(args.db, replace=args.replace):
        print(f"run={result['run_id']}\t{result['scenario']}\tsteps={result['steps']}")


if __name__ == "__main__":
    main()
