"""
ООП-модель нейронного самомаршрутизирующегося преобразователя (НС АЦП).

Архитектура:
- BaseArchitecture (ABC): Базовый класс для топологий НС АЦП.
- ToroidalGridArchitecture: Тороидальная матрица 6x6.
- RhomboidArchitecture: Ромбовидная архитектура W=[4, 6, 8, 6, 4].
- Star8Architecture: Восьмиконечная звезда Star-8 (4 или 6 входов, глубина 4).
- HoneycombPipelineArchitecture: Гексагональная сотовая конвейерная топология (deg=3).
- SymmetricStar8Architecture: Изоморфная симметричная 8-лучевая звезда C8 (deg=3).
- SimulationController: Выделенный MVC-контроллер симуляции.
- SimulationEngine: Фасад для обратной совместимости.
"""

from __future__ import annotations
import abc
import copy
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

RAY_COLORS = [
    "#ab47bc",  # Луч 0: Север
    "#ffb74d",  # Луч 1: СВ
    "#66bb6a",  # Луч 2: Восток
    "#26a69a",  # Луч 3: ЮВ
    "#3f51b5",  # Луч 4: Юг
    "#e91e63",  # Луч 5: ЮЗ
    "#9fa8da",  # Луч 6: Запад
    "#00bcd4",  # Луч 7: СЗ
]


class NeuronState(Enum):
    FREE = "Свободен"
    BUSY = "Занят"
    FAULTY = "Сбой"
    EXHAUSTED = "Завершен"


class RequestStatus(Enum):
    PENDING = "Ожидание"
    IN_PROGRESS = "В обработке"
    COMPLETED = "Завершена"
    REJECTED = "Отказ"


class PlayState(Enum):
    STOPPED = "Остановлено"
    PLAYING = "Воспроизведение"
    PAUSED = "Пауза"
    FINISHED = "Завершено"


class Request:
    PALETTE = [
        "#00BCD4", "#4CAF50", "#FF9800", "#E91E63",
        "#9C27B0", "#3F51B5", "#009688", "#FF5722",
        "#795548", "#607D8B", "#FFC107", "#8BC34A"
    ]

    def __init__(
        self,
        request_id: int,
        q_required: int = 4,
        input_neuron_index: Tuple[int, int] = (0, 0),
        created_at_tick: int = 0
    ):
        self.id: int = request_id
        self.q_required: int = q_required
        self.q_remaining: int = q_required
        self.service_time: int = 3 * q_required + 2
        self.remaining_service_time: int = self.service_time
        self.input_neuron_index: Tuple[int, int] = input_neuron_index
        self.origin_col: int = input_neuron_index[1]
        self.created_at_tick: int = created_at_tick

        self.status: RequestStatus = RequestStatus.IN_PROGRESS
        self.path: List[Tuple[int, int]] = []
        self.color: str = self.PALETTE[(request_id - 1) % len(self.PALETTE)]
        self.completed_at_tick: Optional[int] = None
        self.rejection_reason: Optional[str] = None

    def attach_to_neuron(self, row: int, col: int) -> None:
        self.path.append((row, col))
        if self.q_remaining > 0:
            self.q_remaining -= 1

    def mark_completed(self, tick: int) -> None:
        self.status = RequestStatus.COMPLETED
        self.completed_at_tick = tick
        self.remaining_service_time = 0

    def mark_rejected(self, tick: int, reason: str = "") -> None:
        self.status = RequestStatus.REJECTED
        self.completed_at_tick = tick
        self.rejection_reason = reason

    def is_finished(self) -> bool:
        return self.status in (RequestStatus.COMPLETED, RequestStatus.REJECTED)

    def is_pipeline_built(self) -> bool:
        return self.q_remaining == 0

    def get_path_string(self) -> str:
        return " -> ".join([f"[{r},{c}]" for r, c in self.path])

    def __repr__(self) -> str:
        return f"<Request #{self.id} q={self.q_required} rem={self.q_remaining} status={self.status.value}>"


class Neuron:
    def __init__(self, row: int, col: int):
        self.row: int = row
        self.col: int = col
        self.address: Tuple[int, int] = (row, col)
        self.is_operational: bool = True
        self.state: NeuronState = NeuronState.FREE
        self.remaining_time: int = 0
        self.current_request: Optional[Request] = None
        self.q_stage: int = 0
        self.processed_requests_count: int = 0
        self.unprocessed_requests_count: int = 0
        self.trail_color: Optional[str] = None
        self.trail_request_id: Optional[int] = None
        self.has_computed: bool = False

    def can_accept(self, single_use: bool = False) -> bool:
        if not self.is_operational:
            return False
        if single_use and self.has_computed:
            return False
        return self.state == NeuronState.FREE

    def accept_request(self, request: Request, stage: int = 1) -> bool:
        if not self.can_accept():
            return False
        self.state = NeuronState.BUSY
        self.current_request = request
        self.q_stage = stage
        self.remaining_time = request.remaining_service_time
        self.trail_color = request.color
        self.trail_request_id = request.id
        request.attach_to_neuron(self.row, self.col)
        return True

    def finalize_computation(self, single_use: bool = False) -> Optional[Request]:
        req = self.current_request
        self.current_request = None
        self.has_computed = True
        self.processed_requests_count += 1
        if self.state != NeuronState.FAULTY:
            self.state = NeuronState.EXHAUSTED if single_use else NeuronState.FREE
        self.remaining_time = 0
        self.q_stage = 0
        return req

    def release_aborted(self) -> None:
        self.current_request = None
        if self.state != NeuronState.FAULTY:
            self.state = NeuronState.FREE
        self.remaining_time = 0
        self.q_stage = 0

    def record_unprocessed(self) -> None:
        self.unprocessed_requests_count += 1

    def toggle_fault(self) -> Tuple[bool, Optional[Request]]:
        dropped_req = None
        if self.state == NeuronState.FAULTY:
            self.is_operational = True
            self.state = NeuronState.FREE
            self.remaining_time = 0
            self.current_request = None
            return False, None
        else:
            self.is_operational = False
            if self.current_request:
                dropped_req = self.current_request
                self.record_unprocessed()
            self.state = NeuronState.FAULTY
            self.remaining_time = 0
            self.current_request = None
            return True, dropped_req

    def get_display_info(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "row": self.row,
            "col": self.col,
            "is_operational": self.is_operational,
            "state": self.state,
            "remaining_time": self.remaining_time,
            "ticks_left": self.remaining_time,
            "request_id": self.current_request.id if self.current_request else None,
            "request_color": self.current_request.color if self.current_request else None,
            "q_stage": self.q_stage,
            "processed_requests_count": self.processed_requests_count,
            "unprocessed_requests_count": self.unprocessed_requests_count,
            "trail_color": self.trail_color,
            "trail_request_id": self.trail_request_id,
            "has_computed": self.has_computed,
        }


@dataclass
class SimulationSnapshot:
    tick: int
    neurons_state: Dict[Tuple[int, int], Dict[str, Any]] | List[List[Dict[str, Any]]]
    requests: List[Request]
    active_count: int
    completed_count: int
    rejected_count: int
    total_generated: int
    new_logs: List[Dict[str, str]]
    next_input_idx: int


class BaseArchitecture(abc.ABC):
    """
    Абстрактный суперкласс для топологий НС АЦП.
    Инкапсулирует потактовую симуляцию, историю (Memento), управление сбоями и снапшотами.
    """

    def __init__(
        self,
        q_required: int = 4,
        total_initial_requests: int = 4,
        single_use: bool = False
    ):
        self.q_required: int = q_required
        self.total_initial_requests: int = total_initial_requests
        self.single_use: bool = single_use

        self.current_tick: int = 0
        self.total_generated: int = 0
        self.completed_count: int = 0
        self.rejected_count: int = 0
        self.next_input_idx: int = 0

        self.all_requests: List[Request] = []
        self.logs: List[Dict[str, str]] = []
        self.history: List[SimulationSnapshot] = []
        self.current_history_index: int = 0

        self.neurons_map: Dict[Tuple[int, int], Neuron] = {}

    @property
    def input_columns_count(self) -> int:
        """Количество входных узлов на слое 0."""
        inputs = [addr[1] for addr in self.neurons_map if addr[0] == 0]
        return max(1, len(inputs))

    @abc.abstractmethod
    def reset_topology(self) -> None:
        """Построение/сброс узлов топологии и заполнение self.neurons_map."""
        pass

    @abc.abstractmethod
    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        """Локальный поиск смежных физических линий связи без опроса соседей."""
        pass

    @abc.abstractmethod
    def get_topology_info(self) -> Dict[str, Any]:
        """Возвращает метаданные для визуализации графа (узлы, координаты, ребра)."""
        pass

    def get_neuron(self, address: Any) -> Optional[Neuron]:
        if isinstance(address, (tuple, list)) and len(address) == 2:
            return self.neurons_map.get((address[0], address[1]))
        return self.neurons_map.get(address)

    def all_neurons(self) -> List[Neuron]:
        return list(self.neurons_map.values())

    def is_finished(self) -> bool:
        if self.next_input_idx < self.total_initial_requests:
            return False
        return self.current_tick > 0 and all(r.is_finished() for r in self.all_requests)

    def inject_initial_requests(self, tick: int) -> List[Request]:
        if self.next_input_idx >= self.total_initial_requests:
            return []

        if tick % 2 == 1:
            col = self.next_input_idx % self.input_columns_count
            req_id = self.next_input_idx + 1
            self.next_input_idx += 1

            req = Request(
                request_id=req_id,
                q_required=self.q_required,
                input_neuron_index=(0, col),
                created_at_tick=tick
            )
            return [req]
        return []

    def log_event(self, event_type: str, message: str) -> None:
        self.logs.append({
            "tick": self.current_tick,
            "type": event_type,
            "msg": message
        })

    def truncate_future(self) -> None:
        if self.current_history_index < len(self.history) - 1:
            self.history = self.history[:self.current_history_index + 1]
            self.logs = [entry for entry in self.logs if entry["tick"] <= self.current_tick]

    def _record_snapshot(self, new_logs: List[Dict[str, str]]) -> None:
        grid_data = {
            n.address: n.get_display_info()
            for n in self.all_neurons()
        }
        active = sum(1 for req in self.all_requests if req.status == RequestStatus.IN_PROGRESS)

        snapshot = SimulationSnapshot(
            tick=self.current_tick,
            neurons_state=grid_data,
            requests=copy.deepcopy(self.all_requests),
            active_count=active,
            completed_count=self.completed_count,
            rejected_count=self.rejected_count,
            total_generated=self.total_generated,
            new_logs=list(new_logs),
            next_input_idx=self.next_input_idx
        )
        self.history.append(snapshot)
        self.current_history_index = len(self.history) - 1

    def _update_current_snapshot(self) -> None:
        if not self.history or self.current_history_index >= len(self.history):
            return
        grid_data = {
            n.address: n.get_display_info()
            for n in self.all_neurons()
        }
        active = sum(1 for req in self.all_requests if req.status == RequestStatus.IN_PROGRESS)
        curr = self.history[self.current_history_index]
        self.history[self.current_history_index] = SimulationSnapshot(
            tick=self.current_tick,
            neurons_state=grid_data,
            requests=copy.deepcopy(self.all_requests),
            active_count=active,
            completed_count=self.completed_count,
            rejected_count=self.rejected_count,
            total_generated=self.total_generated,
            new_logs=curr.new_logs,
            next_input_idx=self.next_input_idx
        )

    def _restore_from_snapshot(self, s: SimulationSnapshot) -> None:
        self.current_tick = s.tick
        self.total_generated = s.total_generated
        self.completed_count = s.completed_count
        self.rejected_count = s.rejected_count
        self.next_input_idx = s.next_input_idx
        self.all_requests = copy.deepcopy(s.requests)

        is_dict = isinstance(s.neurons_state, dict)
        for n in self.all_neurons():
            if is_dict:
                info = s.neurons_state.get(n.address)
            else:
                r, c = n.address
                if r < len(s.neurons_state) and c < len(s.neurons_state[r]):
                    info = s.neurons_state[r][c]
                else:
                    info = None
            if info:
                n.is_operational = info.get("is_operational", True)
                n.state = info["state"]
                n.remaining_time = info.get("remaining_time", 0)
                n.q_stage = info.get("q_stage", 0)
                n.processed_requests_count = info.get("processed_requests_count", 0)
                n.unprocessed_requests_count = info.get("unprocessed_requests_count", 0)
                n.trail_color = info.get("trail_color")
                n.trail_request_id = info.get("trail_request_id")
                n.has_computed = info.get("has_computed", False)
                req_id = info.get("request_id")
                n.current_request = next((rq for rq in self.all_requests if rq.id == req_id), None) if req_id else None

    def toggle_neuron_fault(self, row: int, col: int) -> bool:
        self.truncate_future()
        neuron = self.get_neuron((row, col))
        if not neuron:
            return False
        is_faulty, dropped_req = neuron.toggle_fault()

        if is_faulty:
            self.log_event("FAULT", f"Нейрон {neuron.address} отключен.")
            if dropped_req:
                self.rejected_count += 1
                dropped_req.mark_rejected(self.current_tick, f"Сбой узла {neuron.address}")
                self.log_event("REJECT", f"Отказ: Заявка #{dropped_req.id} прервана из-за сбоя {neuron.address}.")
                for pos in dropped_req.path:
                    other_n = self.get_neuron(pos)
                    if other_n and other_n != neuron:
                        other_n.release_aborted()
        else:
            self.log_event("SYS", f"Нейрон {neuron.address} включен.")

        if self.history:
            self._update_current_snapshot()
        return is_faulty

    def step_simulation(self) -> None:
        if self.is_finished():
            return

        self.current_tick += 1
        cycle_logs_start = len(self.logs)

        # 1. Ввод новых заявок
        new_injected = self.inject_initial_requests(self.current_tick)
        for req in new_injected:
            self.total_generated += 1
            self.all_requests.append(req)
            entry = self.get_neuron(req.input_neuron_index)
            if entry and entry.accept_request(req, stage=1):
                self.log_event(
                    "INPUT",
                    f"Заявка #{req.id} (q={req.q_required}, T={req.service_time}) вошла в {entry.address}. Осталось занять: {req.q_remaining}."
                )
            else:
                if entry:
                    entry.record_unprocessed()
                req.mark_rejected(self.current_tick, f"Вход {req.input_neuron_index} недоступен")
                self.rejected_count += 1
                self.log_event("REJECT", f"Отказ: вход {req.input_neuron_index} недоступен для заявки #{req.id}.")

        # 2. Продвижение конвейера (для заявок с q_remaining > 0, исключая только что вошедшие)
        active_pipeline_reqs = [
            r for r in self.all_requests
            if r.status == RequestStatus.IN_PROGRESS and r.q_remaining > 0 and r not in new_injected
        ]

        for req in active_pipeline_reqs:
            last_pos = req.path[-1]
            last_neuron = self.get_neuron(last_pos)
            if not last_neuron:
                continue

            candidates = self.get_candidate_neighbors(last_neuron, req)
            routed = False
            for target_neuron, direction in candidates:
                if target_neuron.can_accept(self.single_use):
                    stage = req.q_required - req.q_remaining + 1
                    target_neuron.accept_request(req, stage)
                    routed = True
                    self.log_event(
                        "ROUTE",
                        f"Заявка #{req.id}: {last_neuron.address} -> {target_neuron.address} ({direction}). Осталось занять: {req.q_remaining}."
                    )
                    if req.is_pipeline_built():
                        self.log_event("BUILT", f"Заявка #{req.id}: цепочка {req.q_required} ОИН построена! {req.get_path_string()}")
                    break

            if not routed:
                req.mark_rejected(self.current_tick, "Все кандидаты на следующем слое недоступны")
                self.rejected_count += 1
                self.log_event("REJECT", f"Отказ: Заявка #{req.id} сброшена из-за тупика в {last_neuron.address}.")
                for pos in req.path:
                    n = self.get_neuron(pos)
                    if n:
                        n.record_unprocessed()
                        n.release_aborted()

        # 3. Потактовый декремент времени обслуживания
        active_computing_reqs = [
            r for r in self.all_requests
            if r.status == RequestStatus.IN_PROGRESS
        ]

        for req in active_computing_reqs:
            req.remaining_service_time -= 1
            for pos in req.path:
                n = self.get_neuron(pos)
                if n and n.current_request == req:
                    n.remaining_time = max(0, req.remaining_service_time)

            if req.remaining_service_time <= 0:
                req.mark_completed(self.current_tick)
                self.completed_count += 1
                self.log_event("DONE", f"Заявка #{req.id} завершена успешно. Путь: {req.get_path_string()}")
                for pos in req.path:
                    n = self.get_neuron(pos)
                    if n and n.current_request == req:
                        n.finalize_computation(self.single_use)

        new_logs = self.logs[cycle_logs_start:]
        self._record_snapshot(new_logs)

    def step_forward(self) -> None:
        if self.current_history_index < len(self.history) - 1:
            self.current_history_index += 1
            self._restore_from_snapshot(self.history[self.current_history_index])
        else:
            self.step_simulation()

    def step_backward(self) -> None:
        if self.current_history_index > 0:
            self.current_history_index -= 1
            self._restore_from_snapshot(self.history[self.current_history_index])

    def fast_forward_to_end(self, max_ticks: int = 500) -> None:
        while not self.is_finished() and max_ticks > 0:
            self.step_forward()
            max_ticks -= 1
        self.go_to_history_tick(len(self.history) - 1)

    def go_to_history_tick(self, idx: int) -> None:
        if 0 <= idx < len(self.history):
            self.current_history_index = idx
            self._restore_from_snapshot(self.history[idx])

    def reset(self) -> None:
        self.current_tick = 0
        self.total_generated = 0
        self.completed_count = 0
        self.rejected_count = 0
        self.next_input_idx = 0
        self.all_requests.clear()
        self.logs.clear()
        self.history.clear()
        self.reset_topology()
        self._record_snapshot([])
        self.log_event("SYS", "Сброс архитектуры к начальному состоянию.")


class ToroidalGridArchitecture(BaseArchitecture):
    """
    Тороидальная матричная архитектура 6x6.
    Закольцована по горизонтали и вертикали.
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 6,
        q_required: int = 4,
        total_initial_requests: int = 6,
        single_use: bool = False
    ):
        super().__init__(q_required=q_required, total_initial_requests=total_initial_requests, single_use=single_use)
        self.rows: int = rows
        self.cols: int = cols
        self.neurons: List[List[Neuron]] = []
        self.reset_topology()
        self._record_snapshot([])
        self.log_event("SYS", f"Инициализация тора {rows}x{cols}: {total_initial_requests} заявок, q={q_required}.")

    @property
    def input_columns_count(self) -> int:
        return self.cols

    def reset_topology(self) -> None:
        self.neurons = [
            [Neuron(r, c) for c in range(self.cols)]
            for r in range(self.rows)
        ]
        self.neurons_map = {n.address: n for row in self.neurons for n in row}
        self.next_input_idx = 0

    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        next_row = (neuron.row + 1) % self.rows
        cur_col = neuron.col
        orig_col = request.origin_col

        col_center = cur_col
        col_left = (cur_col - 1) % self.cols
        col_right = (cur_col + 1) % self.cols

        n_center = self.neurons[next_row][col_center]
        n_left = self.neurons[next_row][col_left]
        n_right = self.neurons[next_row][col_right]

        if cur_col == orig_col:
            return [
                (n_center, "центр"),
                (n_left, "лево"),
                (n_right, "право")
            ]

        # Маятниковый возврат к исходному каналу
        dist_left = (cur_col - orig_col) % self.cols
        dist_right = (orig_col - cur_col) % self.cols

        if dist_left < dist_right:
            return [
                (n_left, "возврат к каналу (влево)"),
                (n_center, "продолжение смещения (центр)"),
                (n_right, "маневр вправо")
            ]
        else:
            return [
                (n_right, "возврат к каналу (вправо)"),
                (n_center, "продолжение смещения (центр)"),
                (n_left, "маневр влево")
            ]

    def get_topology_info(self) -> Dict[str, Any]:
        nodes = []
        for r in range(self.rows):
            for c in range(self.cols):
                nodes.append({
                    "address": (r, c),
                    "label": f"[{r}, {c}]",
                    "row": r,
                    "col": c,
                    "x": (c + 0.5) / self.cols,
                    "y": (r + 0.5) / self.rows,
                    "is_input": (r == 0)
                })

        edges = []
        for r in range(self.rows):
            next_r = (r + 1) % self.rows
            for c in range(self.cols):
                edges.append({"from": (r, c), "to": (next_r, c), "type": "primary", "color": "#00bcd4"})
                edges.append({"from": (r, c), "to": (next_r, (c - 1) % self.cols), "type": "lateral", "color": "#546e7a"})
                edges.append({"from": (r, c), "to": (next_r, (c + 1) % self.cols), "type": "lateral", "color": "#546e7a"})

        return {
            "type": "toroidal_grid",
            "name": f"Тороидальная матрица {self.rows}x{self.cols}",
            "rows": self.rows,
            "cols": self.cols,
            "nodes": nodes,
            "edges": edges,
            "wrap_horizontal": True
        }


class RhomboidArchitecture(BaseArchitecture):
    """
    Ромбовидная архитектура НС АЦП.
    Ширина слоев расширяется от входа к центру, затем сужается к выходу: W = [4, 6, 8, 6, 4].
    """
    DEFAULT_WIDTHS = [4, 6, 8, 6, 4]

    def __init__(
        self,
        widths: Optional[List[int]] = None,
        q_required: int = 4,
        total_initial_requests: int = 4,
        single_use: bool = False
    ):
        super().__init__(q_required=q_required, total_initial_requests=total_initial_requests, single_use=single_use)
        self.widths = list(widths or self.DEFAULT_WIDTHS)
        self.rows = len(self.widths)
        self.neurons: List[List[Neuron]] = []
        self.reset_topology()
        self._record_snapshot([])
        self.log_event("SYS", f"Инициализация ромба W={self.widths}: {total_initial_requests} заявок, q={q_required}.")

    @property
    def input_columns_count(self) -> int:
        return self.widths[0]

    def reset_topology(self) -> None:
        self.neurons = [
            [Neuron(r, c) for c in range(w)]
            for r, w in enumerate(self.widths)
        ]
        self.neurons_map = {n.address: n for row in self.neurons for n in row}
        self.next_input_idx = 0

    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        next_row = (neuron.row + 1) % self.rows
        cur_w = self.widths[neuron.row]
        next_w = self.widths[next_row]
        cur_c = neuron.col

        center_c = int(round(cur_c * (next_w - 1) / max(1, cur_w - 1)))
        left_c = (center_c - 1) % next_w
        right_c = (center_c + 1) % next_w

        n_center = self.neurons[next_row][center_c]
        n_left = self.neurons[next_row][left_c]
        n_right = self.neurons[next_row][right_c]

        orig_w = self.widths[0]
        orig_proj = int(round(request.origin_col * (next_w - 1) / max(1, orig_w - 1)))

        if center_c == orig_proj:
            return [
                (n_center, "по каналу (центр)"),
                (n_left, "веерное расширение (влево)"),
                (n_right, "веерное расширение (вправо)")
            ]

        dist_left = (center_c - orig_proj) % next_w
        dist_right = (orig_proj - center_c) % next_w

        if dist_left < dist_right:
            return [
                (n_left, "стягивание к каналу (влево)"),
                (n_center, "смещение (центр)"),
                (n_right, "альтернатива (вправо)")
            ]
        else:
            return [
                (n_right, "стягивание к каналу (вправо)"),
                (n_center, "смещение (центр)"),
                (n_left, "альтернатива (влево)")
            ]

    def get_topology_info(self) -> Dict[str, Any]:
        nodes, edges = [], []
        max_w = max(self.widths)

        for r, w in enumerate(self.widths):
            for c in range(w):
                norm_x = (c + 0.5) / w
                x = 0.5 + (norm_x - 0.5) * (w / max_w)
                nodes.append({
                    "address": (r, c),
                    "label": f"[{r}, {c}]",
                    "row": r,
                    "col": c,
                    "x": x,
                    "y": (r + 0.5) / self.rows,
                    "is_input": (r == 0)
                })

                if r < self.rows - 1:
                    next_w = self.widths[r + 1]
                    proj_c = int(round(c * (next_w - 1) / max(1, w - 1)))
                    for offset, kind in [(0, "primary"), (-1, "lateral"), (1, "lateral")]:
                        nc = (proj_c + offset) % next_w
                        edges.append({
                            "from": (r, c),
                            "to": (r + 1, nc),
                            "type": kind,
                            "color": "#4fc3f7" if kind == "primary" else "#555555"
                        })

                edges.append({
                    "from": (r, c),
                    "to": (r, (c + 1) % w),
                    "type": "lateral",
                    "color": "#444444"
                })

        return {
            "type": "rhomboid",
            "name": f"Ромбовидная архитектура ({'-'.join(map(str, self.widths))})",
            "widths": self.widths,
            "nodes": nodes,
            "edges": edges
        }


class Star8Architecture(BaseArchitecture):
    """
    Восьмиконечная радиальная звезда Star-8 (4 или 6 входов, глубина 4).
    Содержит 8 радиальных лучей и мостовые узлы на первом кольце.
    """
    RAY_COLORS = RAY_COLORS

    def __init__(
        self,
        ray_depth: int = 4,
        num_inputs: int = 4,
        q_required: int = 4,
        total_initial_requests: int = 4,
        single_use: bool = False
    ):
        super().__init__(q_required=q_required, total_initial_requests=total_initial_requests, single_use=single_use)
        self.ray_depth = ray_depth
        self.num_inputs = num_inputs
        self.reset_topology()
        self._record_snapshot([])
        self.log_event("SYS", f"Инициализация Star-8 (входов={num_inputs}, глубина={ray_depth}, узлов={len(self.neurons_map)}).")

    @property
    def input_columns_count(self) -> int:
        return self.num_inputs

    def reset_topology(self) -> None:
        self.neurons_map.clear()
        for c in range(self.num_inputs):
            self.neurons_map[(0, c)] = Neuron(0, c)
        for d in range(1, self.ray_depth + 1):
            for ray in range(8):
                self.neurons_map[(d, ray)] = Neuron(d, ray)
        for b_idx in [8, 9, 10, 11]:
            self.neurons_map[(1, b_idx)] = Neuron(1, b_idx)
        self.next_input_idx = 0

    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        candidates = []
        r, c = neuron.row, neuron.col

        if r == 0:
            if self.num_inputs == 6:
                in_map = {
                    0: [(1, 6), (1, 7), (1, 5), (1, 8), (1, 11)],
                    1: [(1, 7), (1, 0), (1, 8), (1, 6)],
                    2: [(1, 5), (1, 4), (1, 11), (1, 6)],
                    3: [(1, 0), (1, 1), (1, 9), (1, 2)],
                    4: [(1, 4), (1, 3), (1, 10), (1, 2)],
                    5: [(1, 2), (1, 1), (1, 3), (1, 9), (1, 10)]
                }
            else:
                in_map = {
                    0: [(1, 6), (1, 7), (1, 5), (1, 8), (1, 11)],
                    1: [(1, 0), (1, 8), (1, 4), (1, 11), (1, 7)],
                    2: [(1, 0), (1, 9), (1, 4), (1, 10), (1, 1)],
                    3: [(1, 2), (1, 1), (1, 3), (1, 9), (1, 10)]
                }
            targets = in_map.get(c, [(1, c % 8), (1, (c + 1) % 8)])
            for t_addr in targets:
                t = self.neurons_map.get(t_addr)
                if t:
                    candidates.append((t, f"вход -> {t.address}"))
            return candidates

        if r == 1 and c >= 8:
            bridge_targets = {
                8: [(2, 0), (2, 7), (1, 0), (1, 7)],
                9: [(2, 0), (2, 1), (1, 0), (1, 1)],
                10: [(2, 4), (2, 3), (1, 4), (1, 3)],
                11: [(2, 4), (2, 5), (1, 4), (1, 5)],
            }
            for t_addr in bridge_targets.get(c, []):
                t = self.neurons_map.get(t_addr)
                if t:
                    candidates.append((t, f"мост {c} -> {t.address}"))
            return candidates

        if r == 1 and c < 8:
            ray = c
            t_straight = self.neurons_map.get((2, ray))
            if t_straight:
                candidates.append((t_straight, "по лучу прямо"))
            ray_bridges = {0: [8, 9], 1: [9], 3: [10], 4: [10, 11], 5: [11], 7: [8]}
            for b_idx in ray_bridges.get(ray, []):
                t_b = self.neurons_map.get((1, b_idx))
                if t_b:
                    candidates.append((t_b, f"обходной мост [1, {b_idx}]"))
            cw_ray, ccw_ray = (ray + 1) % 8, (ray - 1) % 8
            for nr in [cw_ray, ccw_ray]:
                t_nr = self.neurons_map.get((2, nr))
                if t_nr:
                    candidates.append((t_nr, f"смежный луч {nr}"))
            return candidates

        if 2 <= r < self.ray_depth:
            ray = c
            t_straight = self.neurons_map.get((r + 1, ray))
            if t_straight:
                candidates.append((t_straight, f"по лучу прямо -> [{r + 1}, {ray}]"))
            if r < self.ray_depth - 1:
                cw_ray, ccw_ray = (ray + 1) % 8, (ray - 1) % 8
                for s_ray in [cw_ray, ccw_ray]:
                    t_s = self.neurons_map.get((r, s_ray))
                    if t_s:
                        candidates.append((t_s, f"смежный узел [{r}, {s_ray}]"))
                    t_a = self.neurons_map.get((r + 1, s_ray))
                    if t_a:
                        candidates.append((t_a, f"аварийный луч [{r + 1}, {s_ray}]"))
            return candidates

        return []

    def get_topology_info(self) -> Dict[str, Any]:
        nodes = []
        raw_edges = []

        coords: Dict[Tuple[int, int], Tuple[float, float]] = {
            (4, 7): (0.05, 0.05), (4, 0): (0.50, 0.05), (4, 1): (0.95, 0.05),
            (3, 7): (0.13, 0.16), (3, 0): (0.50, 0.16), (3, 1): (0.87, 0.16),
            (2, 7): (0.21, 0.27), (2, 0): (0.50, 0.27), (2, 1): (0.79, 0.27),
            (1, 7): (0.29, 0.38), (1, 8): (0.40, 0.38), (1, 0): (0.50, 0.38), (1, 9): (0.60, 0.38), (1, 1): (0.71, 0.38),
            (4, 6): (0.05, 0.50), (3, 6): (0.13, 0.50), (2, 6): (0.21, 0.50), (1, 6): (0.29, 0.50),
            (1, 2): (0.71, 0.50), (2, 2): (0.79, 0.50), (3, 2): (0.87, 0.50), (4, 2): (0.95, 0.50),
            (1, 5): (0.29, 0.62), (1, 11): (0.40, 0.62), (1, 4): (0.50, 0.62), (1, 10): (0.60, 0.62), (1, 3): (0.71, 0.62),
            (2, 5): (0.21, 0.73), (2, 4): (0.50, 0.73), (2, 3): (0.79, 0.73),
            (3, 5): (0.13, 0.84), (3, 4): (0.50, 0.84), (3, 3): (0.87, 0.84),
            (4, 5): (0.05, 0.95), (4, 4): (0.50, 0.95), (4, 3): (0.95, 0.95),
        }

        if self.num_inputs == 6:
            coords.update({(0, 0): (0.38, 0.45), (0, 1): (0.50, 0.45), (0, 3): (0.62, 0.45),
                           (0, 2): (0.38, 0.55), (0, 4): (0.50, 0.55), (0, 5): (0.62, 0.55)})
        else:
            coords.update({(0, 0): (0.38, 0.50), (0, 1): (0.46, 0.50), (0, 2): (0.54, 0.50), (0, 3): (0.62, 0.50)})

        for addr, n in self.neurons_map.items():
            r, c = addr
            is_input = (r == 0)
            x, y = coords.get(addr, (0.5, 0.5))
            color = "#ffca28" if is_input else ("#90a4ae" if r == 1 and c >= 8 else self.RAY_COLORS[c % 8])
            nodes.append({
                "address": addr,
                "label": f"[{r}, {c}]",
                "row": r,
                "col": c,
                "x": x,
                "y": y,
                "is_input": is_input,
                "color": color
            })

        for ray in range(8):
            color = self.RAY_COLORS[ray]
            for d in range(1, self.ray_depth):
                raw_edges.append({"from": (d, ray), "to": (d + 1, ray), "type": "primary", "color": color})

        in_primary = [
            ((0, 0), (1, 6)), ((0, 0), (1, 7)), ((0, 0), (1, 5)),
            ((0, 1), (1, 7)), ((0, 1), (1, 0)),
            ((0, 2), (1, 5)), ((0, 2), (1, 4)),
            ((0, 3), (1, 0)), ((0, 3), (1, 1)),
            ((0, 4), (1, 4)), ((0, 4), (1, 3)),
            ((0, 5), (1, 2)), ((0, 5), (1, 1)), ((0, 5), (1, 3)),
        ] if self.num_inputs == 6 else [
            ((0, 0), (1, 6)), ((0, 0), (1, 7)), ((0, 0), (1, 5)),
            ((0, 1), (1, 0)), ((0, 1), (1, 4)),
            ((0, 2), (1, 0)), ((0, 2), (1, 4)),
            ((0, 3), (1, 1)), ((0, 3), (1, 2)), ((0, 3), (1, 3)),
        ]

        for src, dst in in_primary:
            c = self.RAY_COLORS[dst[1] % 8] if dst[0] == 1 and dst[1] < 8 else "#78909c"
            raw_edges.append({"from": src, "to": dst, "type": "primary", "color": c})

        bridge_edges = [
            ((1, 0), (1, 8)), ((1, 7), (1, 8)), ((1, 8), (2, 0)), ((1, 8), (2, 7)),
            ((1, 0), (1, 9)), ((1, 1), (1, 9)), ((1, 9), (2, 0)), ((1, 9), (2, 1)),
            ((1, 4), (1, 10)), ((1, 3), (1, 10)), ((1, 10), (2, 4)), ((1, 10), (2, 3)),
            ((1, 4), (1, 11)), ((1, 5), (1, 11)), ((1, 11), (2, 4)), ((1, 11), (2, 5)),
        ]
        for src, dst in bridge_edges:
            raw_edges.append({"from": src, "to": dst, "type": "lateral", "color": "#78909c"})

        seen = set()
        unique_edges = []
        for e in raw_edges:
            pair = tuple(sorted([e["from"], e["to"]]))
            if pair not in seen:
                seen.add(pair)
                unique_edges.append(e)

        return {
            "type": "star8",
            "name": f"Восьмиконечная звезда (Star-8, {self.num_inputs} вх, глубина {self.ray_depth})",
            "ray_depth": self.ray_depth,
            "num_inputs": self.num_inputs,
            "nodes": nodes,
            "edges": unique_edges
        }


class HoneycombPipelineArchitecture(BaseArchitecture):
    """
    Гексагональная сотовая конвейерная топология (deg=3).
    Обход сбоев происходит со сдвигом на следующий слой (0°, -60°, +60°).
    """

    def __init__(
        self,
        rows: int = 5,
        cols: int = 6,
        q_required: int = 4,
        total_initial_requests: int = 6,
        single_use: bool = False
    ):
        super().__init__(q_required=q_required, total_initial_requests=total_initial_requests, single_use=single_use)
        self.rows: int = rows
        self.cols: int = cols
        self.neurons: List[List[Neuron]] = []
        self.reset_topology()
        self._record_snapshot([])
        self.log_event("SYS", f"Инициализация соты {rows}x{cols} (deg=3): {total_initial_requests} заявок, q={q_required}.")

    @property
    def input_columns_count(self) -> int:
        return self.cols

    def reset_topology(self) -> None:
        self.neurons = [
            [Neuron(r, c) for c in range(self.cols)]
            for r in range(self.rows)
        ]
        self.neurons_map = {n.address: n for row in self.neurons for n in row}
        self.next_input_idx = 0

    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        r, c = neuron.row, neuron.col
        if r >= self.rows - 1:
            return []

        n_straight = self.neurons[r + 1][c]
        n_left = self.neurons[r + 1][(c - 1) % self.cols]
        n_right = self.neurons[r + 1][(c + 1) % self.cols]

        return [
            (n_straight, "прямо вниз (0°)"),
            (n_left, "обход влево (-60°)"),
            (n_right, "обход вправо (+60°)")
        ]

    def get_topology_info(self) -> Dict[str, Any]:
        nodes = []
        for r in range(self.rows):
            offset_x = 0.25 if (r % 2 == 1) else -0.25
            for c in range(self.cols):
                norm_x = (c + 0.5 + offset_x) / self.cols
                nodes.append({
                    "address": (r, c),
                    "label": f"[{r}, {c}]",
                    "row": r,
                    "col": c,
                    "x": max(0.04, min(0.96, norm_x)),
                    "y": (r + 0.5) / self.rows,
                    "is_input": (r == 0),
                    "color": "#ffca28" if r == 0 else "#00bcd4"
                })

        edges = []
        for r in range(self.rows - 1):
            for c in range(self.cols):
                edges.append({"from": (r, c), "to": (r + 1, c), "type": "primary", "color": "#00bcd4"})
                edges.append({"from": (r, c), "to": (r + 1, (c - 1) % self.cols), "type": "lateral", "color": "#546e7a"})
                edges.append({"from": (r, c), "to": (r + 1, (c + 1) % self.cols), "type": "lateral", "color": "#546e7a"})

        return {
            "type": "honeycomb",
            "name": f"Гексагональная сота {self.rows}x{self.cols} (deg=3)",
            "rows": self.rows,
            "cols": self.cols,
            "nodes": nodes,
            "edges": edges
        }


class SymmetricStar8Architecture(BaseArchitecture):
    """
    Симметричная 8-лучевая радиальная звезда с кольцом хорд C8 (deg=3).
    Строгая вращательная симметрия 8-го порядка.
    """
    RAY_COLORS = RAY_COLORS

    def __init__(
        self,
        ray_depth: int = 4,
        num_inputs: int = 6,
        q_required: int = 4,
        total_initial_requests: int = 6,
        single_use: bool = False
    ):
        super().__init__(q_required=q_required, total_initial_requests=total_initial_requests, single_use=single_use)
        self.ray_depth = ray_depth
        self.num_inputs = num_inputs
        self.reset_topology()
        self._record_snapshot([])
        self.log_event("SYS", f"Инициализация Sym-Star8 (deg=3, входов={num_inputs}, лучей=8, глубина={ray_depth}).")

    @property
    def input_columns_count(self) -> int:
        return self.num_inputs

    def reset_topology(self) -> None:
        self.neurons_map.clear()
        for c in range(self.num_inputs):
            self.neurons_map[(0, c)] = Neuron(0, c)
        for d in range(1, self.ray_depth + 1):
            for ray in range(8):
                self.neurons_map[(d, ray)] = Neuron(d, ray)
        self.next_input_idx = 0

    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        r, c = neuron.row, neuron.col

        if r == 0:
            target_ray = int(round(c * 8.0 / self.num_inputs)) % 8
            n_target = self.neurons_map.get((1, target_ray))
            n_cw = self.neurons_map.get((1, (target_ray + 1) % 8))
            n_ccw = self.neurons_map.get((1, (target_ray - 1) % 8))

            cands = []
            if n_target:
                cands.append((n_target, f"домашний луч {target_ray}"))
            if n_cw:
                cands.append((n_cw, f"смежный луч {(target_ray + 1) % 8}"))
            if n_ccw:
                cands.append((n_ccw, f"смежный луч {(target_ray - 1) % 8}"))
            return cands

        if r >= self.ray_depth:
            return []

        n_straight = self.neurons_map.get((r + 1, c))
        n_cw = self.neurons_map.get((r, (c + 1) % 8))
        n_ccw = self.neurons_map.get((r, (c - 1) % 8))

        cands = []
        if n_straight:
            cands.append((n_straight, "по лучу прямо"))
        if n_cw:
            cands.append((n_cw, f"обход по кольцу -> луч {(c + 1) % 8}"))
        if n_ccw:
            cands.append((n_ccw, f"обход по кольцу -> луч {(c - 1) % 8}"))
        return cands

    def get_topology_info(self) -> Dict[str, Any]:
        nodes = []
        raw_edges = []
        cx, cy = 0.5, 0.5

        r_inputs = 0.09
        for c in range(self.num_inputs):
            angle = -math.pi / 2 + c * (2 * math.pi / self.num_inputs)
            nodes.append({
                "address": (0, c),
                "label": f"[0, {c}]",
                "row": 0,
                "col": c,
                "x": cx + r_inputs * math.cos(angle),
                "y": cy + r_inputs * math.sin(angle),
                "is_input": True,
                "color": "#ffca28"
            })

        radii = [0.18, 0.28, 0.38, 0.46]
        for d in range(1, self.ray_depth + 1):
            rad = radii[d - 1] if d - 1 < len(radii) else (0.15 + d * 0.08)
            for ray in range(8):
                angle = -math.pi / 2 + ray * (2 * math.pi / 8)
                color = self.RAY_COLORS[ray % 8]
                nodes.append({
                    "address": (d, ray),
                    "label": f"[{d}, {ray}]",
                    "row": d,
                    "col": ray,
                    "x": cx + rad * math.cos(angle),
                    "y": cy + rad * math.sin(angle),
                    "is_input": False,
                    "color": color
                })

        for ray in range(8):
            color = self.RAY_COLORS[ray]
            for d in range(1, self.ray_depth):
                raw_edges.append({"from": (d, ray), "to": (d + 1, ray), "type": "primary", "color": color})

        for d in range(1, self.ray_depth):
            for ray in range(8):
                raw_edges.append({"from": (d, ray), "to": (d, (ray + 1) % 8), "type": "lateral", "color": "#455a64"})

        for c in range(self.num_inputs):
            target_ray = int(round(c * 8.0 / self.num_inputs)) % 8
            for ray in [target_ray, (target_ray + 1) % 8, (target_ray - 1) % 8]:
                raw_edges.append({"from": (0, c), "to": (1, ray), "type": "primary", "color": "#78909c"})

        seen = set()
        unique_edges = []
        for e in raw_edges:
            pair = tuple(sorted([e["from"], e["to"]]))
            if pair not in seen:
                seen.add(pair)
                unique_edges.append(e)

        return {
            "type": "sym_star8",
            "name": f"Симметричная звезда C8 (Sym-Star8, deg=3, входов {self.num_inputs})",
            "ray_depth": self.ray_depth,
            "num_inputs": self.num_inputs,
            "nodes": nodes,
            "edges": unique_edges
        }


class Funnel8Architecture(BaseArchitecture):
    """
    Расширенная воронка 6->8 (W=[6, 8, 8, 8, 8], всего 38 узлов, deg=3).
    Слой 0: 6 входных каналов.
    Слои 1..4: по 8 узлов (каналы 3 и 7 — постоянный пространственный резерв).
    Выдерживает 100% одиночных сбоев на любом слое при одновременной подаче 6 заявок.
    """
    WIDTHS = [6, 8, 8, 8, 8]
    IN_MAP = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6}

    def __init__(
        self,
        q_required: int = 4,
        total_initial_requests: int = 6,
        single_use: bool = False
    ):
        super().__init__(q_required=q_required, total_initial_requests=total_initial_requests, single_use=single_use)
        self.rows = len(self.WIDTHS)
        self.neurons: List[List[Neuron]] = []
        self.reset_topology()
        self._record_snapshot([])
        self.log_event("SYS", f"Инициализация Funnel-8 (W={self.WIDTHS}): 6 входов, 2 буфера на слой, q={q_required}.")

    @property
    def input_columns_count(self) -> int:
        return 6

    def reset_topology(self) -> None:
        self.neurons = [
            [Neuron(r, c) for c in range(w)]
            for r, w in enumerate(self.WIDTHS)
        ]
        self.neurons_map = {n.address: n for row in self.neurons for n in row}
        self.next_input_idx = 0

    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        r, c = neuron.row, neuron.col
        if r >= self.rows - 1:
            return []

        if r == 0:
            home = self.IN_MAP.get(c, c)
            return [
                (self.neurons[1][home], "основной канал"),
                (self.neurons[1][(home - 1) % 8], "буферный маневр влево"),
                (self.neurons[1][(home + 1) % 8], "буферный маневр вправо")
            ]

        orig_home = self.IN_MAP.get(request.origin_col, request.origin_col)
        if c == orig_home:
            return [
                (self.neurons[r + 1][c], "по каналу прямо"),
                (self.neurons[r + 1][(c - 1) % 8], "обход влево"),
                (self.neurons[r + 1][(c + 1) % 8], "обход вправо")
            ]

        dist_left = (c - orig_home) % 8
        dist_right = (orig_home - c) % 8
        if dist_left < dist_right:
            return [
                (self.neurons[r + 1][(c - 1) % 8], "возврат к каналу (влево)"),
                (self.neurons[r + 1][c], "смещение прямо"),
                (self.neurons[r + 1][(c + 1) % 8], "маневр вправо")
            ]
        else:
            return [
                (self.neurons[r + 1][(c + 1) % 8], "возврат к каналу (вправо)"),
                (self.neurons[r + 1][c], "смещение прямо"),
                (self.neurons[r + 1][(c - 1) % 8], "маневр влево")
            ]

    def get_topology_info(self) -> Dict[str, Any]:
        nodes, edges = [], []
        max_w = 8
        for r, w in enumerate(self.WIDTHS):
            for c in range(w):
                norm_x = (c + 0.5) / w
                x = 0.5 + (norm_x - 0.5) * (w / max_w)
                is_in = (r == 0)
                is_buf = (c in (3, 7) and r > 0)
                color = "#ffca28" if is_in else ("#26a69a" if is_buf else "#00bcd4")
                nodes.append({
                    "address": (r, c),
                    "label": f"[{r}, {c}]",
                    "row": r,
                    "col": c,
                    "x": x,
                    "y": (r + 0.5) / self.rows,
                    "is_input": is_in,
                    "color": color
                })

        for c in range(6):
            home = self.IN_MAP[c]
            for hc in [home, (home - 1) % 8, (home + 1) % 8]:
                edges.append({
                    "from": (0, c), "to": (1, hc),
                    "type": "primary" if hc == home else "lateral",
                    "color": "#00bcd4" if hc == home else "#546e7a"
                })

        for r in range(1, self.rows - 1):
            for c in range(8):
                edges.append({"from": (r, c), "to": (r + 1, c), "type": "primary", "color": "#00bcd4"})
                edges.append({"from": (r, c), "to": (r + 1, (c - 1) % 8), "type": "lateral", "color": "#546e7a"})
                edges.append({"from": (r, c), "to": (r + 1, (c + 1) % 8), "type": "lateral", "color": "#546e7a"})

        return {
            "type": "funnel8",
            "name": "Расширенная воронка 6->8 (W=[6,8,8,8,8], deg=3)",
            "widths": self.WIDTHS,
            "nodes": nodes,
            "edges": edges
        }


class ChordalFunnel8Architecture(BaseArchitecture):
    """
    Хордовая расширенная воронка 6->8 (W=[6, 8, 8, 8, 8], deg=3, диаметральная хорда +4).
    Сочетает пространственный резерв W=8 и хордовый перескок на противоположный сектор (+4).
    Выдерживает 78.6% парных сбоев на слое 1 при одновременной подаче 6 заявок.
    """
    WIDTHS = [6, 8, 8, 8, 8]
    IN_MAP = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6}

    def __init__(
        self,
        q_required: int = 4,
        total_initial_requests: int = 6,
        single_use: bool = False
    ):
        super().__init__(q_required=q_required, total_initial_requests=total_initial_requests, single_use=single_use)
        self.rows = len(self.WIDTHS)
        self.neurons: List[List[Neuron]] = []
        self.reset_topology()
        self._record_snapshot([])
        self.log_event("SYS", f"Инициализация Chordal-Funnel-8 (W={self.WIDTHS}, хорда +4): 6 входов, q={q_required}.")

    @property
    def input_columns_count(self) -> int:
        return 6

    def reset_topology(self) -> None:
        self.neurons = [
            [Neuron(r, c) for c in range(w)]
            for r, w in enumerate(self.WIDTHS)
        ]
        self.neurons_map = {n.address: n for row in self.neurons for n in row}
        self.next_input_idx = 0

    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        r, c = neuron.row, neuron.col
        if r >= self.rows - 1:
            return []

        if r == 0:
            home = self.IN_MAP.get(c, c)
            return [
                (self.neurons[1][home], "основной канал"),
                (self.neurons[1][(home + 1) % 8], "сдвиг +1"),
                (self.neurons[1][(home + 4) % 8], "хордовый прыжок (диаметр +4)")
            ]

        return [
            (self.neurons[r + 1][c], "по каналу прямо"),
            (self.neurons[r + 1][(c + 1) % 8], "сдвиг +1"),
            (self.neurons[r + 1][(c + 4) % 8], "хордовый прыжок (диаметр +4)")
        ]

    def get_topology_info(self) -> Dict[str, Any]:
        nodes, edges = [], []
        max_w = 8
        for r, w in enumerate(self.WIDTHS):
            for c in range(w):
                norm_x = (c + 0.5) / w
                x = 0.5 + (norm_x - 0.5) * (w / max_w)
                is_in = (r == 0)
                is_buf = (c in (3, 7) and r > 0)
                color = "#ffca28" if is_in else ("#26a69a" if is_buf else "#ab47bc")
                nodes.append({
                    "address": (r, c),
                    "label": f"[{r}, {c}]",
                    "row": r,
                    "col": c,
                    "x": x,
                    "y": (r + 0.5) / self.rows,
                    "is_input": is_in,
                    "color": color
                })

        for c in range(6):
            home = self.IN_MAP[c]
            for hc in [home, (home + 1) % 8, (home + 4) % 8]:
                edges.append({
                    "from": (0, c), "to": (1, hc),
                    "type": "primary" if hc == home else "lateral",
                    "color": "#00bcd4" if hc == home else "#ab47bc"
                })

        for r in range(1, self.rows - 1):
            for c in range(8):
                edges.append({"from": (r, c), "to": (r + 1, c), "type": "primary", "color": "#00bcd4"})
                edges.append({"from": (r, c), "to": (r + 1, (c + 1) % 8), "type": "lateral", "color": "#546e7a"})
                edges.append({"from": (r, c), "to": (r + 1, (c + 4) % 8), "type": "lateral", "color": "#ab47bc"})

        return {
            "type": "chordal_funnel8",
            "name": "Хордовая воронка 6->8 (W=[6,8,8,8,8], deg=3, хорда +4)",
            "widths": self.WIDTHS,
            "nodes": nodes,
            "edges": edges
        }


class SimulationController:
    """MVC-контроллер симуляции."""

    def __init__(self, architecture: BaseArchitecture, speed_ms: int = 400):
        self.arch: BaseArchitecture = architecture
        self.speed_ms: int = speed_ms
        self.state: PlayState = PlayState.STOPPED
        self.listeners: Dict[str, List[Callable]] = {
            "tick": [],
            "state_change": [],
            "log": [],
            "reset": []
        }

    @property
    def current_tick(self) -> int:
        return self.arch.current_tick

    @property
    def architecture(self) -> BaseArchitecture:
        return self.arch

    def subscribe(self, event: str, callback: Callable) -> None:
        if event in self.listeners:
            self.listeners[event].append(callback)

    def notify(self, event: str, *args, **kwargs) -> None:
        for cb in self.listeners.get(event, []):
            try:
                cb(*args, **kwargs)
            except Exception as e:
                print(f"Error in listener {cb}: {e}")

    def set_architecture(self, new_arch: BaseArchitecture) -> None:
        self.state = PlayState.STOPPED
        self.arch = new_arch
        self.notify("state_change", self.state)
        self.notify("reset")

    def play(self) -> None:
        if self.arch.is_finished():
            self.state = PlayState.FINISHED
        else:
            self.state = PlayState.PLAYING
        self.notify("state_change", self.state)

    def pause(self) -> None:
        self.state = PlayState.PAUSED
        self.notify("state_change", self.state)

    def toggle_play(self) -> None:
        if self.state == PlayState.PLAYING:
            self.pause()
        else:
            self.play()

    def step_forward(self) -> None:
        self.arch.step_forward()
        if self.arch.is_finished() and self.state == PlayState.PLAYING:
            self.state = PlayState.FINISHED
            self.notify("state_change", self.state)
        self.notify("tick", self.arch.current_tick)

    def step_backward(self) -> None:
        self.pause()
        self.arch.step_backward()
        self.notify("tick", self.arch.current_tick)

    def fast_forward(self, max_ticks: int = 500) -> None:
        self.arch.fast_forward_to_end(max_ticks=max_ticks)
        self.state = PlayState.FINISHED
        self.notify("state_change", self.state)
        self.notify("tick", self.arch.current_tick)

    def go_to_tick(self, tick_idx: int) -> None:
        self.pause()
        self.arch.go_to_history_tick(tick_idx)
        self.notify("tick", self.arch.current_tick)

    def reset(self) -> None:
        self.state = PlayState.STOPPED
        self.arch.reset()
        self.notify("state_change", self.state)
        self.notify("reset")

    def toggle_fault(self, row: int, col: int) -> bool:
        res = self.arch.toggle_neuron_fault(row, col)
        self.notify("tick", self.arch.current_tick)
        return res

    def set_speed(self, speed_ms: int) -> None:
        self.speed_ms = speed_ms

    def get_kpi(self) -> Dict[str, Any]:
        total = self.arch.total_generated
        completed = self.arch.completed_count
        rejected = self.arch.rejected_count
        active = sum(1 for r in self.arch.all_requests if r.status == RequestStatus.IN_PROGRESS)
        fail_rate = (rejected / total * 100.0) if total > 0 else 0.0
        faults_count = sum(1 for n in self.arch.all_neurons() if not n.is_operational)
        total_neurons = len(self.arch.all_neurons())

        return {
            "total": total,
            "completed": completed,
            "rejected": rejected,
            "active": active,
            "fail_rate": fail_rate,
            "faults_count": faults_count,
            "total_neurons": total_neurons
        }


class SimulationEngine(ToroidalGridArchitecture):
    """Алиас для обратной совместимости."""
    pass
