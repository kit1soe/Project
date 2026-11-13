"""Пакетные headless-эксперименты НС АЦП.

Модуль не требует PySide6 и запускает одну или несколько серий по всем
поддерживаемым топологиям. Результаты сохраняются в CSV, а рядом создаётся
небольшой HTML-отчёт с графиками вероятности отказа.

Пример:

    python batch_experiments.py --requests 100,500,1000 \
        --q 3,4 --lambdas 0.25,0.5,1.0 --fault-counts 0,1,2
"""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from controller import SimulationController
from generate_analytics import _factories
from models import ArrivalMode, BaseArchitecture, NeuronState


@dataclass
class BatchResult:
    topology: str
    q_required: int
    request_count: int
    arrival_mode: str
    arrival_rate: float
    arrival_seed: Optional[int]
    failure_count: int
    failure_mode: str
    run_index: int
    ticks: int
    generated: int
    completed: int
    rejected: int
    rejection_probability: float
    mean_path_length: float
    mean_completed_path_length: float
    mean_duration_ticks: Optional[float]
    utilization: float
    active_peak: int
    throughput_per_tick: float
    faulty_neurons: int
    applied_fault_events: int
    timed_out: bool


def _parse_list(value: str, cast):
    values = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(cast(raw))
    if not values:
        raise ValueError(f"Пустой список параметров: {value!r}")
    return values


def _factory_map() -> dict[str, object]:
    return dict(_factories())


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def run_single(
    topology: str,
    q_required: int,
    request_count: int,
    *,
    arrival_mode: str = ArrivalMode.POISSON.value,
    arrival_rate: float = 0.5,
    arrival_seed: Optional[int] = 0,
    failure_count: int = 0,
    failure_mode: str = "immediate",
    run_index: int = 1,
    fault_seed: Optional[int] = None,
    max_ticks: int = 20_000,
) -> BatchResult:
    """Запустить один сценарий и вернуть агрегированные показатели."""
    factories = _factory_map()
    if topology not in factories:
        raise ValueError(f"Неизвестная топология: {topology}")
    if q_required < 1:
        raise ValueError("q_required должен быть положительным")
    if request_count < 1:
        raise ValueError("request_count должен быть положительным")
    if failure_count < 0:
        raise ValueError("failure_count не может быть отрицательным")
    if failure_mode not in {"immediate", "scheduled"}:
        raise ValueError("failure_mode должен быть immediate или scheduled")

    factory = factories[topology]
    architecture: BaseArchitecture = factory(q_required, request_count, True)  # type: ignore[operator]
    architecture.set_arrival_process(
        mode=arrival_mode,
        rate=arrival_rate,
        seed=arrival_seed,
    )
    # История нужна интерактивной GUI, но для 100/500/1000 заявок превращается
    # в квадратичный расход памяти. Начальный снимок оставляем для совместимых
    # вызовов контроллера, последующие не сохраняем.
    architecture.set_history_enabled(False)
    controller = SimulationController(architecture)

    effective_fault_seed = fault_seed if fault_seed is not None else (
        arrival_seed + 1 if arrival_seed is not None else None
    )
    if failure_count:
        if failure_mode == "scheduled":
            architecture.schedule_random_faults(
                count=failure_count,
                probability=1.0,
                seed=effective_fault_seed,
            )
        else:
            architecture.randomize_faults(
                count=failure_count,
                probability=1.0,
                seed=effective_fault_seed,
            )

    steps = 0
    busy_ticks = 0
    active_peak = 0
    while not architecture.is_finished() and steps < max_ticks:
        controller.step_forward()
        steps += 1
        active = sum(
            1 for request in architecture.all_requests
            if request.status.name == "IN_PROGRESS"
        )
        active_peak = max(active_peak, active)
        busy_ticks += sum(
            1 for neuron in architecture.all_neurons()
            if neuron.state == NeuronState.BUSY
        )

    requests = list(architecture.all_requests)
    generated = len(requests)
    completed = sum(request.status.name == "COMPLETED" for request in requests)
    rejected = sum(request.status.name == "REJECTED" for request in requests)
    paths = [float(len(request.path)) for request in requests]
    completed_paths = [
        float(len(request.path)) for request in requests
        if request.status.name == "COMPLETED"
    ]
    durations = [
        float(request.completed_at_tick - request.created_at_tick)
        for request in requests
        if request.completed_at_tick is not None
        and request.status.name == "COMPLETED"
    ]
    neuron_count = max(1, len(architecture.all_neurons()))
    utilization = busy_ticks / max(1, neuron_count * steps)
    fault_events = sum(1 for entry in architecture.logs if entry["type"] == "FAULT")
    faulty_neurons = sum(1 for neuron in architecture.all_neurons() if not neuron.is_operational)

    return BatchResult(
        topology=topology,
        q_required=q_required,
        request_count=request_count,
        arrival_mode=arrival_mode.value if isinstance(arrival_mode, ArrivalMode) else str(arrival_mode),
        arrival_rate=float(arrival_rate),
        arrival_seed=arrival_seed,
        failure_count=failure_count,
        failure_mode=failure_mode if failure_count else "none",
        run_index=run_index,
        ticks=steps,
        generated=generated,
        completed=completed,
        rejected=rejected,
        rejection_probability=round(rejected / max(1, generated), 6),
        mean_path_length=_mean(paths),
        mean_completed_path_length=_mean(completed_paths),
        mean_duration_ticks=round(sum(durations) / len(durations), 4) if durations else None,
        utilization=round(utilization, 6),
        active_peak=active_peak,
        throughput_per_tick=round(completed / max(1, steps), 6),
        faulty_neurons=faulty_neurons,
        applied_fault_events=fault_events,
        timed_out=not architecture.is_finished(),
    )


def run_batch(
    *,
    output_dir: str | Path,
    request_counts: Iterable[int] = (100,),
    q_values: Iterable[int] = (4,),
    arrival_rates: Iterable[float] = (0.5,),
    failure_counts: Iterable[int] = (0, 1, 2),
    arrival_mode: str = ArrivalMode.POISSON.value,
    failure_mode: str = "immediate",
    repetitions: int = 1,
    seed: int = 20261113,
    max_ticks: int = 20_000,
    topologies: Optional[Iterable[str]] = None,
) -> tuple[list[dict[str, object]], Path, Path]:
    """Выполнить сетку экспериментов и сохранить CSV/HTML."""
    if repetitions < 1:
        raise ValueError("repetitions должен быть положительным")
    names = list(topologies) if topologies is not None else list(_factory_map())
    rows: list[dict[str, object]] = []
    run_index = 0
    for request_count in request_counts:
        for q_required in q_values:
            for arrival_rate in arrival_rates:
                for failure_count in failure_counts:
                    for repetition in range(repetitions):
                        run_index += 1
                        arrival_seed = seed + run_index
                        for topology in names:
                            result = run_single(
                                topology,
                                q_required,
                                request_count,
                                arrival_mode=arrival_mode,
                                arrival_rate=arrival_rate,
                                arrival_seed=arrival_seed,
                                failure_count=failure_count,
                                failure_mode=failure_mode,
                                run_index=repetition + 1,
                                fault_seed=seed + 10_000 + run_index,
                                max_ticks=max_ticks,
                            )
                            rows.append(asdict(result))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    html_path = output_dir / "report.html"
    _write_csv(rows, csv_path)
    _write_html_report(rows, html_path)
    return rows, csv_path, html_path


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _svg_plot(rows: list[dict[str, object]], *, title: str) -> str:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    fault_values = sorted({int(row["failure_count"]) for row in rows})
    for row in rows:
        label = str(row["topology"])
        grouped[label][int(row["failure_count"])] = float(row["rejection_probability"])

    width, height = 900, 360
    left, right, top, bottom = 70, 25, 45, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    x_max = max(fault_values or [1])
    y_max = max(1.0, max((value for series in grouped.values() for value in series.values()), default=0.0))
    colors = ["#00bcd4", "#ffca28", "#66bb6a", "#ef5350", "#ab47bc", "#ff7043", "#8d6e63", "#26a69a", "#5c6bc0"]

    def point(fault: int, value: float) -> tuple[float, float]:
        x = left + (fault / x_max) * plot_w if x_max else left
        y = top + plot_h - (value / y_max) * plot_h
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#121820"/>',
        f'<text x="{left}" y="25" fill="#e6edf3" font-size="16">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" y2="{top + plot_h}" stroke="#90a4ae"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#90a4ae"/>',
        f'<text x="{width/2:.0f}" y="{height-12}" fill="#90a4ae" text-anchor="middle">число отказов</text>',
        f'<text x="15" y="{top + plot_h/2:.0f}" fill="#90a4ae" transform="rotate(-90 15 {top + plot_h/2:.0f})" text-anchor="middle">Pотк</text>',
    ]
    for fault in fault_values or [0]:
        x, _ = point(fault, 0)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#90a4ae"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 20}" fill="#90a4ae" text-anchor="middle" font-size="11">{fault}</text>')
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h - fraction * plot_h
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" fill="#90a4ae" text-anchor="end" font-size="11">{fraction*y_max:.2f}</text>')
    for index, (label, series) in enumerate(sorted(grouped.items())):
        coords = [point(fault, series[fault]) for fault in fault_values if fault in series]
        if not coords:
            continue
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        color = colors[index % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        x, y = coords[-1]
        parts.append(f'<text x="{min(width-right, x+8):.1f}" y="{y-4:.1f}" fill="{color}" font-size="11">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _write_html_report(rows: list[dict[str, object]], path: Path) -> None:
    groups: dict[tuple[int, float, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["q_required"]), float(row["arrival_rate"]), int(row["request_count"]))].append(row)

    plots = []
    for (q_required, arrival_rate, request_count), group in sorted(groups.items()):
        title = f"Pотк по топологиям · N={request_count}, q={q_required}, λ={arrival_rate:g}"
        plots.append(_svg_plot(group, title=title))

    headers = list(rows[0]) if rows else []
    table = ""
    if rows:
        head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
        body = []
        for row in rows:
            body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers) + "</tr>")
        table = f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    sections = []
    for (q_required, arrival_rate, request_count), plot in zip(sorted(groups), plots):
        title = f"N={request_count}, q={q_required}, λ={arrival_rate:g}"
        sections.append(f"<h2>{html.escape(title)}</h2>{plot}")

    content = """<!doctype html>
<meta charset="utf-8">
<title>НС АЦП — пакетные эксперименты</title>
<style>
body{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:24px}
h1{font-size:22px} h2{font-size:16px;color:#90a4ae;margin-top:28px}
svg{max-width:100%;border:1px solid #263241;margin:8px 0 20px}
table{border-collapse:collapse;font-size:11px;display:block;overflow:auto;max-height:520px}
th,td{border:1px solid #303b49;padding:4px 6px;white-space:nowrap}
th{position:sticky;top:0;background:#161d26}
</style>
<h1>Пакетные эксперименты НС АЦП</h1>
<p>Вероятность отказа: число отклонённых заявок / число сгенерированных заявок.</p>
""" + "".join(sections) + table
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Пакетная статистика НС АЦП без GUI")
    parser.add_argument("--output-dir", type=Path, default=Path("analytics") / "batch")
    parser.add_argument("--requests", default="100", help="N через запятую, например 100,500,1000")
    parser.add_argument("--q", dest="q_values", default="4", help="q через запятую")
    parser.add_argument("--lambdas", default="0.5", help="λ через запятую")
    parser.add_argument("--fault-counts", default="0,1,2", help="число отказов через запятую")
    parser.add_argument("--topologies", default="", help="ограничить топологии, через запятую")
    parser.add_argument("--mode", choices=[mode.value for mode in ArrivalMode], default=ArrivalMode.POISSON.value)
    parser.add_argument("--fault-mode", choices=["immediate", "scheduled"], default="immediate")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--max-ticks", type=int, default=20_000)
    args = parser.parse_args()

    topologies = [item.strip() for item in args.topologies.split(",") if item.strip()] or None
    rows, csv_path, html_path = run_batch(
        output_dir=args.output_dir,
        request_counts=_parse_list(args.requests, int),
        q_values=_parse_list(args.q_values, int),
        arrival_rates=_parse_list(args.lambdas, float),
        failure_counts=_parse_list(args.fault_counts, int),
        arrival_mode=args.mode,
        failure_mode=args.fault_mode,
        repetitions=args.repetitions,
        seed=args.seed,
        max_ticks=args.max_ticks,
        topologies=topologies,
    )
    print(f"Строк результатов: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Отчёт: {html_path}")


if __name__ == "__main__":
    main()
