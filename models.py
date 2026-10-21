"""
ООП-модель нейронного самомаршрутизирующегося преобразователя (НС АЦП).

Архитектура:
- BaseArchitecture (ABC): Абстрактный суперкласс для топологий НС АЦП.
- ToroidalGridArchitecture: Тороидальная матрица 6x4.
- RhomboidArchitecture: Ромбовидная архитектура.
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
    "#ab47bc",  # Луч 0: Север (фиолетовый)
    "#ffb74d",  # Луч 1: СВ (оранжево-желтый)
    "#66bb6a",  # Луч 2: Восток (зеленый)
    "#26a69a",  # Луч 3: ЮВ (бирюзовый)
    "#3f51b5",  # Луч 4: Юг (синий)
    "#e91e63",  # Луч 5: ЮЗ (розовый)
    "#9fa8da",  # Луч 6: Запад (лавандовый)
    "#00bcd4",  # Луч 7: СЗ (голубой/cyan)
]

class NeuronState(Enum):
    FREE = "Свободен"
    BUSY = "Занят"
    EXHAUSTED = "Завершен"
    FAULTY = "Сбой"

class RequestStatus(Enum):
    PENDING = "Ожидание"
    IN_PROGRESS = "Обрабатывается"
    COMPLETED = "Обработано"
    REJECTED = "Отказано"

class PlayState(Enum):
    STOPPED = "Остановлено"
    PLAYING = "Выполняется"
    PAUSED = "Пауза"
    FINISHED = "Завершено"

PALETTE = [
    "#00BCD4",  # Cyan
    "#FFB300",  # Amber
    "#AB47BC",  # Purple
    "#26A69A",  # Teal
    "#E91E63",  # Pink
    "#3F51B5",  # Indigo
    "#FF5722",  # Deep Orange
    "#8BC34A",  # Light Green
]

class Request:
    def __init__(
        self,
        request_id: int,
        q_required: int = 4,
        input_neuron_index: Tuple[int, int] = (0, 0),
        color: Optional[str] = None,
        created_at_tick: int = 0
    ):
        self.id: int = request_id
        self.q_required: int = q_required
        self.q_remaining: int = q_required
        self.input_neuron_index: Tuple[int, int] = input_neuron_index
        self.origin_col: int = input_neuron_index[1]

        # время жизни = 2 * q + q + 2 (= 3*q + 2)
        self.service_time: int = 2 * q_required + q_required + 2
        self.remaining_service_time: int = self.service_time

        self.color: str = color or PALETTE[(request_id - 1) % len(PALETTE)]
        self.path: List[Tuple[int, int]] = []
        self.status: RequestStatus = RequestStatus.PENDING
        self.created_at_tick: int = created_at_tick
        self.completed_at_tick: Optional[int] = None
        self.reject_reason: str = ""

    def attach_to_neuron(self, row: int, col: int) -> None:
        self.path.append((row, col))
        if self.q_remaining > 0:
            self.q_remaining -= 1
        self.status = RequestStatus.IN_PROGRESS

    def mark_completed(self, tick: int) -> None:
        self.status = RequestStatus.COMPLETED
        self.completed_at_tick = tick

    def mark_rejected(self, tick: int, reason: str) -> None:
        self.status = RequestStatus.REJECTED
        self.completed_at_tick = tick
        self.reject_reason = reason

    def is_pipeline_built(self) -> bool:
        return self.q_remaining <= 0

    def is_finished(self) -> bool:
        return self.status in (RequestStatus.COMPLETED, RequestStatus.REJECTED)

    def get_path_string(self) -> str:
        return "->".join([f"[{r},{c}]" for r, c in self.path])

    def __repr__(self) -> str:
        return (
            f"<Request #{self.id} Вход={self.input_neuron_index} "
            f"q_rem={self.q_remaining}/{self.q_required} T={self.remaining_service_time} "
            f"{self.status.value}>"
        )


class Neuron:
    def __init__(self, row: int, col: int):
        self.row: int = row
        self.col: int = col

        self.is_operational: bool = True
        self.state: NeuronState = NeuronState.FREE
        self.remaining_time: int = 0
        self.current_request: Optional[Request] = None
        self.q_stage: int = 0

        self.processed_requests_count: int = 0
        self.unprocessed_requests_count: int = 0
        self.has_computed: bool = False
        self.trail_color: Optional[str] = None
        self.trail_request_id: Optional[int] = None

    @property
    def address(self) -> str:
        return f"[{self.row}, {self.col}]"

    def can_accept(self, single_use: bool = False) -> bool:
        if not self.is_operational:
            return False
        if self.state != NeuronState.FREE:
            return False
        if single_use and self.has_computed:
            return False
        return True

    def accept_request(self, request: Request, stage: int) -> bool:
        if not self.can_accept():
            return False

        self.current_request = request
        self.state = NeuronState.BUSY
        self.q_stage = stage
        self.remaining_time = request.remaining_service_time

        self.trail_color = request.color
        self.trail_request_id = request.id
        request.attach_to_neuron(self.row, self.col)
        return True

    def process_tick(self) -> bool:
        if self.state != NeuronState.BUSY or not self.current_request:
            return False
        self.remaining_time -= 1
        return self.remaining_time <= 0

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
    neurons_state: List[List[Dict[str, Any]]]
    requests: List[Request]
    active_count: int
    completed_count: int
    rejected_count: int
    total_generated: int
    new_logs: List[Dict[str, str]]
    next_input_idx: int


class BaseArchitecture(abc.ABC):
    def __init__(self, single_use: bool = False):
        self.single_use: bool = single_use
        self.current_tick: int = 0
        self.total_generated: int = 0
        self.completed_count: int = 0
        self.rejected_count: int = 0
        self.all_requests: List[Request] = []
        self.logs: List[Dict[str, str]] = []
        self.history: List[SimulationSnapshot] = []
        self.current_history_index: int = 0

    @abc.abstractmethod
    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        pass

    @abc.abstractmethod
    def inject_initial_requests(self, tick: int) -> List[Request]:
        pass

    @abc.abstractmethod
    def is_finished(self) -> bool:
        pass

    @abc.abstractmethod
    def get_topology_info(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def reset_topology(self) -> None:
        pass

    @abc.abstractmethod
    def get_neuron(self, address: Any) -> Optional[Neuron]:
        pass

    @abc.abstractmethod
    def all_neurons(self) -> List[Neuron]:
        pass

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
        pass

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

        # 2. Продвижение конвейера: для заявок, у которых q_remaining > 0, переходим на нижний уровень
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

        # 3. Потактовый декремент времени обслуживания для всех активных заявок
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
    def __init__(
        self,
        rows: int = 6,
        cols: int = 4,
        q_required: int = 4,
        total_initial_requests: int = 4,
        single_use: bool = False
    ):
        super().__init__(single_use=single_use)
        self.rows: int = rows
        self.cols: int = cols
        self.q_required: int = q_required
        self.total_initial_requests: int = total_initial_requests
        self.next_input_idx: int = 0

        self.neurons: List[List[Neuron]] = [
            [Neuron(r, c) for c in range(cols)]
            for r in range(rows)
        ]

        self._record_snapshot([])
        self.log_event("SYS", f"Инициализация тора {rows}x{cols}: {total_initial_requests} заявок, q={q_required}, T={3*q_required+2}.")

    def get_neuron(self, address: Any) -> Optional[Neuron]:
        if isinstance(address, (tuple, list)) and len(address) == 2:
            r, c = address
            if 0 <= r < self.rows and 0 <= c < self.cols:
                return self.neurons[r][c]
        return None

    def all_neurons(self) -> List[Neuron]:
        return [self.neurons[r][c] for r in range(self.rows) for c in range(self.cols)]

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
                (n_right, "право"),
            ]

        dist_left = (cur_col - orig_col) % self.cols
        dist_right = (orig_col - cur_col) % self.cols

        if dist_right < dist_left:
            return [
                (n_right, "возврат вправо"),
                (n_center, "центр"),
                (n_left, "лево"),
            ]
        else:
            return [
                (n_left, "возврат влево"),
                (n_center, "центр"),
                (n_right, "право"),
            ]

    def inject_initial_requests(self, tick: int) -> List[Request]:
        if self.next_input_idx >= self.total_initial_requests:
            return []

        if tick % 2 == 1:
            col = self.next_input_idx % self.cols
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

    def is_finished(self) -> bool:
        if self.next_input_idx < self.total_initial_requests:
            return False
        return all(r.is_finished() for r in self.all_requests)

    def get_topology_info(self) -> Dict[str, Any]:
        return {
            "type": "toroidal_grid",
            "name": "Тороидальная матрица 6x4",
            "rows": self.rows,
            "cols": self.cols,
            "wrap_vertical": True,
            "wrap_horizontal": True
        }

    def reset_topology(self) -> None:
        self.neurons = [
            [Neuron(r, c) for c in range(self.cols)]
            for r in range(self.rows)
        ]
        self.next_input_idx = 0

    def _record_snapshot(self, new_logs: List[Dict[str, str]]) -> None:
        grid_data = [
            [self.neurons[r][c].get_display_info() for c in range(self.cols)]
            for r in range(self.rows)
        ]
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

    def _restore_from_snapshot(self, s: SimulationSnapshot) -> None:
        self.current_tick = s.tick
        self.total_generated = s.total_generated
        self.completed_count = s.completed_count
        self.rejected_count = s.rejected_count
        self.next_input_idx = s.next_input_idx
        self.all_requests = copy.deepcopy(s.requests)

        for r in range(self.rows):
            for c in range(self.cols):
                info = s.neurons_state[r][c]
                n = self.neurons[r][c]
                n.is_operational = info.get("is_operational", True)
                n.state = info["state"]
                n.remaining_time = info.get("remaining_time", 0)
                n.q_stage = info.get("q_stage", 0)
                n.processed_requests_count = info.get("processed_requests_count", 0)
                n.unprocessed_requests_count = info.get("unprocessed_requests_count", 0)
                n.trail_color = info["trail_color"]
                n.trail_request_id = info["trail_request_id"]
                n.has_computed = info["has_computed"]
                req_id = info["request_id"]
                n.current_request = next((rq for rq in self.all_requests if rq.id == req_id), None) if req_id else None

    def toggle_neuron_fault(self, row: int, col: int) -> bool:
        self.truncate_future()
        neuron = self.neurons[row][col]
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

    def _update_current_snapshot(self) -> None:
        grid_data = [
            [self.neurons[r][c].get_display_info() for c in range(self.cols)]
            for r in range(self.rows)
        ]
        self.history[self.current_history_index].neurons_state = grid_data
        self.history[self.current_history_index].requests = copy.deepcopy(self.all_requests)
        self.history[self.current_history_index].rejected_count = self.rejected_count

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
        self.all_requests.clear()
        self.logs.clear()
        self.history.clear()
        self.reset_topology()
        self._record_snapshot([])
        self.log_event("SYS", "Сброс архитектуры к начальному состоянию.")


class RhomboidArchitecture(BaseArchitecture):
    """
    Ромбовидная архитектура НС АЦП.
    Ширина слоев расширяется от входа к центру, затем сужается к выходу:
    W = [4, 6, 8, 6, 4].
    Каждый слой закольцован тороидально по горизонтали.
    """
    DEFAULT_WIDTHS = [4, 6, 8, 6, 4]

    def __init__(
        self,
        widths: Optional[List[int]] = None,
        q_required: int = 4,
        total_initial_requests: int = 4,
        single_use: bool = False
    ):
        super().__init__(single_use=single_use)
        self.widths = list(widths or self.DEFAULT_WIDTHS)
        self.rows = len(self.widths)
        self.q_required = q_required
        self.total_initial_requests = total_initial_requests
        self.next_input_idx = 0

        self.neurons: List[List[Neuron]] = [
            [Neuron(r, c) for c in range(w)]
            for r, w in enumerate(self.widths)
        ]

        self._record_snapshot([])
        self.log_event("SYS", f"Инициализация ромба W={self.widths}: {total_initial_requests} заявок, q={q_required}.")

    def get_neuron(self, address: Any) -> Optional[Neuron]:
        if isinstance(address, (tuple, list)) and len(address) == 2:
            r, c = address
            if 0 <= r < self.rows and 0 <= c < self.widths[r]:
                return self.neurons[r][c]
        return None

    def all_neurons(self) -> List[Neuron]:
        return [self.neurons[r][c] for r in range(self.rows) for c in range(self.widths[r])]

    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        next_row = (neuron.row + 1) % self.rows
        cur_w = self.widths[neuron.row]
        next_w = self.widths[next_row]
        cur_c = neuron.col

        # Проекция текущего столбца на ширину следующего слоя
        center_c = int(round(cur_c * (next_w - 1) / max(1, cur_w - 1)))
        left_c = (center_c - 1) % next_w
        right_c = (center_c + 1) % next_w

        n_center = self.neurons[next_row][center_c]
        n_left = self.neurons[next_row][left_c]
        n_right = self.neurons[next_row][right_c]

        # Если слой расширяется: веерный поиск
        # Если слой сужается: схождение обратно к своему каналу
        orig_col = request.origin_col
        target_orig_c = int(round(orig_col * (next_w - 1) / max(1, self.widths[0] - 1)))

        if center_c == target_orig_c:
            return [(n_center, "центр"), (n_left, "лево"), (n_right, "право")]
        elif (center_c - target_orig_c) % next_w > next_w / 2:
            # Исходный канал ближе справа -> возврат вправо
            return [(n_right, "возврат вправо"), (n_center, "центр"), (n_left, "лево")]
        else:
            # Исходный канал ближе слева -> возврат влево
            return [(n_left, "возврат влево"), (n_center, "центр"), (n_right, "право")]

    def inject_initial_requests(self, tick: int) -> List[Request]:
        if self.next_input_idx >= self.total_initial_requests:
            return []

        if tick % 2 == 1:
            col = self.next_input_idx % self.widths[0]
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

    def is_finished(self) -> bool:
        if self.next_input_idx < self.total_initial_requests:
            return False
        return all(r.is_finished() for r in self.all_requests)

    def get_topology_info(self) -> Dict[str, Any]:
        nodes = []
        edges = []

        max_w = max(self.widths)
        for r, w in enumerate(self.widths):
            for c in range(w):
                # Центрирование строки относительно максимальной ширины
                x_offset = (max_w - w) * 0.5 + c
                nodes.append({
                    "address": (r, c),
                    "label": f"[{r}, {c}]",
                    "row": r,
                    "col": c,
                    "x": x_offset / max_w,
                    "y": r / (self.rows - 1),
                    "is_input": (r == 0)
                })

                # Основные прямые и веерные связи на следующий слой
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

                # Горизонтальная тороидальная перемычка внутри слоя
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

    def reset_topology(self) -> None:
        self.neurons = [
            [Neuron(r, c) for c in range(w)]
            for r, w in enumerate(self.widths)
        ]
        self.next_input_idx = 0

    def _record_snapshot(self, new_logs: List[Dict[str, str]]) -> None:
        grid_data = [
            [self.neurons[r][c].get_display_info() for c in range(self.widths[r])]
            for r in range(self.rows)
        ]
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

    def _restore_from_snapshot(self, s: SimulationSnapshot) -> None:
        self.current_tick = s.tick
        self.total_generated = s.total_generated
        self.completed_count = s.completed_count
        self.rejected_count = s.rejected_count
        self.next_input_idx = s.next_input_idx
        self.all_requests = copy.deepcopy(s.requests)

        for r in range(self.rows):
            for c in range(self.widths[r]):
                info = s.neurons_state[r][c]
                n = self.neurons[r][c]
                n.is_operational = info.get("is_operational", True)
                n.state = info["state"]
                n.remaining_time = info.get("remaining_time", 0)
                n.q_stage = info.get("q_stage", 0)
                n.processed_requests_count = info.get("processed_requests_count", 0)
                n.unprocessed_requests_count = info.get("unprocessed_requests_count", 0)
                n.trail_color = info["trail_color"]
                n.trail_request_id = info["trail_request_id"]
                n.has_computed = info["has_computed"]
                req_id = info["request_id"]
                n.current_request = next((rq for rq in self.all_requests if rq.id == req_id), None) if req_id else None

    def toggle_neuron_fault(self, row: int, col: int) -> bool:
        self.truncate_future()
        neuron = self.neurons[row][col]
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

    def _update_current_snapshot(self) -> None:
        grid_data = [
            [self.neurons[r][c].get_display_info() for c in range(self.widths[r])]
            for r in range(self.rows)
        ]
        self.history[self.current_history_index].neurons_state = grid_data
        self.history[self.current_history_index].requests = copy.deepcopy(self.all_requests)
        self.history[self.current_history_index].rejected_count = self.rejected_count



class Star8Architecture(BaseArchitecture):
    """
    Восьмиконечная радиальная архитектура (Star-8).
    - 4 или 6 входных нейронов в центре.
    - 8 радиальных лучей глубиной D=4:
        слой 1: [1, 0]..[1, 7]
        слой 2: [2, 0]..[2, 7]
        слой 3: [3, 0]..[3, 7]
        слой 4: [4, 0]..[4, 7] (терминальный последовательный каскад)
    - 4 промежуточных узла внутреннего кольца ([1, 8]..[1, 11]).
    - Убывающая степень ветвления: deg+(V0) = 4..5, deg+(V1) = 4, deg+(V2) = 3, deg+(V3) = 1, deg+(V4) = 0.
    """
    RAY_COLORS = [
        "#ab47bc", "#ffb74d", "#66bb6a", "#26a69a",
        "#3f51b5", "#e91e63", "#9fa8da", "#00bcd4"
    ]

    def __init__(
        self,
        ray_depth: int = 4,
        num_inputs: int = 4,
        q_required: int = 4,
        total_initial_requests: int = 4,
        single_use: bool = False
    ):
        super().__init__(single_use=single_use)
        self.ray_depth = ray_depth
        self.num_inputs = num_inputs
        self.q_required = q_required
        self.total_initial_requests = total_initial_requests
        self.next_input_idx = 0

        self.neurons_map: Dict[Tuple[int, int], Neuron] = {}
        self._init_nodes()

        self._record_snapshot([])
        self.log_event(
            "SYS",
            f"Инициализация звезды Star-8 ({num_inputs} входов, 8 лучей, глубина {ray_depth}): {total_initial_requests} заявок, q={q_required}."
        )

    def _init_nodes(self) -> None:
        self.neurons_map.clear()
        for c in range(self.num_inputs):
            self.neurons_map[(0, c)] = Neuron(0, c)

        for d in range(1, self.ray_depth + 1):
            for ray in range(8):
                self.neurons_map[(d, ray)] = Neuron(d, ray)

        for idx in range(8, 12):
            self.neurons_map[(1, idx)] = Neuron(1, idx)

    def get_neuron(self, address: Any) -> Optional[Neuron]:
        if isinstance(address, (tuple, list)) and len(address) == 2:
            return self.neurons_map.get((address[0], address[1]))
        return None

    def all_neurons(self) -> List[Neuron]:
        return list(self.neurons_map.values())

    def get_candidate_neighbors(self, neuron: Neuron, request: Request) -> List[Tuple[Neuron, str]]:
        candidates = []
        r, c = neuron.row, neuron.col

        if r == 0:
            if self.num_inputs == 6:
                if c == 0:
                    targets = [(1, 6), (1, 7), (1, 5), (1, 8), (1, 11)]
                elif c == 1:
                    targets = [(1, 7), (1, 0), (1, 8), (1, 6)]
                elif c == 2:
                    targets = [(1, 5), (1, 4), (1, 11), (1, 6)]
                elif c == 3:
                    targets = [(1, 0), (1, 1), (1, 9), (1, 2)]
                elif c == 4:
                    targets = [(1, 4), (1, 3), (1, 10), (1, 2)]
                elif c == 5:
                    targets = [(1, 2), (1, 1), (1, 3), (1, 9), (1, 10)]
                else:
                    targets = [(1, c % 8), (1, (c + 1) % 8)]
            else:
                if c == 0:
                    targets = [(1, 6), (1, 7), (1, 5), (1, 8), (1, 11)]
                elif c == 1:
                    targets = [(1, 0), (1, 8), (1, 4), (1, 11), (1, 7)]
                elif c == 2:
                    targets = [(1, 0), (1, 9), (1, 4), (1, 10), (1, 1)]
                elif c == 3:
                    targets = [(1, 2), (1, 1), (1, 3), (1, 9), (1, 10)]
                else:
                    targets = [(1, c % 8), (1, (c + 1) % 8)]

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

            ray_bridges = {
                0: [8, 9], 1: [9], 3: [10], 4: [10, 11], 5: [11], 7: [8],
            }
            for b_idx in ray_bridges.get(ray, []):
                t_b = self.neurons_map.get((1, b_idx))
                if t_b:
                    candidates.append((t_b, f"обходной мост [1, {b_idx}]"))

            cw_ray = (ray + 1) % 8
            ccw_ray = (ray - 1) % 8
            t_cw = self.neurons_map.get((2, cw_ray))
            if t_cw:
                candidates.append((t_cw, f"смежный луч {cw_ray}"))
            t_ccw = self.neurons_map.get((2, ccw_ray))
            if t_ccw:
                candidates.append((t_ccw, f"смежный луч {ccw_ray}"))
            return candidates

        if 2 <= r < self.ray_depth:
            ray = c
            t_straight = self.neurons_map.get((r + 1, ray))
            if t_straight:
                candidates.append((t_straight, f"по лучу прямо -> [{r + 1}, {ray}]"))

            if r < self.ray_depth - 1:
                cw_ray = (ray + 1) % 8
                ccw_ray = (ray - 1) % 8
                t_cw2 = self.neurons_map.get((r, cw_ray))
                if t_cw2:
                    candidates.append((t_cw2, f"смежный узел [{r}, {cw_ray}]"))
                t_ccw2 = self.neurons_map.get((r, ccw_ray))
                if t_ccw2:
                    candidates.append((t_ccw2, f"смежный узел [{r}, {ccw_ray}]"))

                t_cw_next = self.neurons_map.get((r + 1, cw_ray))
                if t_cw_next:
                    candidates.append((t_cw_next, f"аварийный луч [{r + 1}, {cw_ray}]"))
                t_ccw_next = self.neurons_map.get((r + 1, ccw_ray))
                if t_ccw_next:
                    candidates.append((t_ccw_next, f"аварийный луч [{r + 1}, {ccw_ray}]"))
            return candidates

        return []

    def inject_initial_requests(self, tick: int) -> List[Request]:
        if self.next_input_idx >= self.total_initial_requests:
            return []

        if tick % 2 == 1:
            col = self.next_input_idx % self.num_inputs
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

    def is_finished(self) -> bool:
        if self.next_input_idx < self.total_initial_requests:
            return False
        return all(r.is_finished() for r in self.all_requests)

    def get_topology_info(self) -> Dict[str, Any]:
        nodes = []
        raw_edges = []

        if self.ray_depth >= 4:
            coords = {
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
        else:
            coords = {
                (3, 7): (0.08, 0.08), (3, 0): (0.50, 0.08), (3, 1): (0.92, 0.08),
                (2, 7): (0.17, 0.22), (2, 0): (0.50, 0.22), (2, 1): (0.83, 0.22),
                (1, 7): (0.25, 0.36), (1, 8): (0.38, 0.36), (1, 0): (0.50, 0.36), (1, 9): (0.60, 0.36), (1, 1): (0.75, 0.36),
                (3, 6): (0.04, 0.50), (2, 6): (0.13, 0.50), (1, 6): (0.22, 0.50),
                (1, 2): (0.78, 0.50), (2, 2): (0.87, 0.50), (3, 2): (0.96, 0.50),
                (1, 5): (0.25, 0.64), (1, 11): (0.38, 0.64), (1, 4): (0.50, 0.64), (1, 10): (0.60, 0.64), (1, 3): (0.75, 0.64),
                (2, 5): (0.17, 0.78), (2, 4): (0.50, 0.78), (2, 3): (0.83, 0.78),
                (3, 5): (0.08, 0.92), (3, 4): (0.50, 0.92), (3, 3): (0.92, 0.92),
            }

        if self.num_inputs == 6:
            coords[(0, 0)] = (0.38, 0.45)
            coords[(0, 1)] = (0.50, 0.45)
            coords[(0, 3)] = (0.62, 0.45)
            coords[(0, 2)] = (0.38, 0.55)
            coords[(0, 4)] = (0.50, 0.55)
            coords[(0, 5)] = (0.62, 0.55)
        else:
            coords[(0, 0)] = (0.38, 0.50)
            coords[(0, 1)] = (0.46, 0.50)
            coords[(0, 2)] = (0.54, 0.50)
            coords[(0, 3)] = (0.62, 0.50)

        for addr, n in self.neurons_map.items():
            r, c = addr
            is_input = (r == 0)
            x, y = coords.get(addr, (0.5, 0.5))

            if is_input:
                color = "#ffca28"
            elif r == 1 and c >= 8:
                color = "#90a4ae"
            else:
                ray_idx = c
                color = self.RAY_COLORS[ray_idx % 8]

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
                raw_edges.append({
                    "from": (d, ray),
                    "to": (d + 1, ray),
                    "type": "primary",
                    "color": color
                })

        if self.num_inputs == 6:
            in_primary = [
                ((0, 0), (1, 6)), ((0, 0), (1, 7)), ((0, 0), (1, 5)),
                ((0, 1), (1, 7)), ((0, 1), (1, 0)),
                ((0, 2), (1, 5)), ((0, 2), (1, 4)),
                ((0, 3), (1, 0)), ((0, 3), (1, 1)),
                ((0, 4), (1, 4)), ((0, 4), (1, 3)),
                ((0, 5), (1, 2)), ((0, 5), (1, 1)), ((0, 5), (1, 3)),
            ]
        else:
            in_primary = [
                ((0, 0), (1, 6)), ((0, 0), (1, 7)), ((0, 0), (1, 5)),
                ((0, 1), (1, 0)), ((0, 1), (1, 4)),
                ((0, 2), (1, 0)), ((0, 2), (1, 4)),
                ((0, 3), (1, 1)), ((0, 3), (1, 2)), ((0, 3), (1, 3)),
            ]
        for src, dst in in_primary:
            ray_color = self.RAY_COLORS[dst[1] % 8] if dst[0] == 1 and dst[1] < 8 else "#78909c"
            raw_edges.append({
                "from": src,
                "to": dst,
                "type": "primary",
                "color": ray_color
            })

        if self.num_inputs == 6:
            in_bridges = [
                ((0, 0), (1, 8)), ((0, 0), (1, 11)),
                ((0, 1), (1, 8)),
                ((0, 2), (1, 11)),
                ((0, 3), (1, 9)),
                ((0, 4), (1, 10)),
                ((0, 5), (1, 9)), ((0, 5), (1, 10)),
            ]
        else:
            in_bridges = [
                ((0, 0), (1, 8)), ((0, 0), (1, 11)),
                ((0, 1), (1, 8)), ((0, 1), (1, 11)),
                ((0, 2), (1, 9)), ((0, 2), (1, 10)),
                ((0, 3), (1, 9)), ((0, 3), (1, 10)),
            ]
        for src, dst in in_bridges:
            raw_edges.append({
                "from": src,
                "to": dst,
                "type": "secondary",
                "color": "#607d8b"
            })

        bridge_links = [
            ((1, 7), (1, 8)), ((1, 8), (1, 0)),
            ((1, 8), (2, 7)), ((1, 8), (2, 0)),
            ((2, 7), (2, 0)),
            ((1, 0), (1, 9)), ((1, 9), (1, 1)),
            ((1, 9), (2, 0)), ((1, 9), (2, 1)),
            ((2, 0), (2, 1)),
            ((1, 5), (1, 11)), ((1, 11), (1, 4)),
            ((1, 11), (2, 5)), ((1, 11), (2, 4)),
            ((2, 5), (2, 4)),
            ((1, 4), (1, 10)), ((1, 10), (1, 3)),
            ((1, 10), (2, 4)), ((1, 10), (2, 3)),
            ((2, 4), (2, 3)),
            ((1, 7), (1, 6)), ((1, 6), (1, 5)),
            ((1, 1), (1, 2)), ((1, 2), (1, 3)),
            ((2, 7), (2, 6)), ((2, 6), (2, 5)),
            ((2, 1), (2, 2)), ((2, 2), (2, 3)),
        ]
        for src, dst in bridge_links:
            raw_edges.append({
                "from": src,
                "to": dst,
                "type": "secondary",
                "color": "#607d8b"
            })

        for d in range(1, max(2, self.ray_depth - 1)):
            for ray in range(8):
                cw_ray = (ray + 1) % 8
                ccw_ray = (ray - 1) % 8
                raw_edges.append({
                    "from": (d, ray),
                    "to": (d + 1, cw_ray),
                    "type": "tertiary",
                    "color": "#37474f"
                })
                raw_edges.append({
                    "from": (d, ray),
                    "to": (d + 1, ccw_ray),
                    "type": "tertiary",
                    "color": "#37474f"
                })

        prio_map = {"primary": 3, "secondary": 2, "tertiary": 1}
        dedup = {}
        for e in raw_edges:
            pair = (min(e["from"], e["to"]), max(e["from"], e["to"]))
            if pair not in dedup:
                dedup[pair] = e
            else:
                if prio_map.get(e.get("type"), 0) > prio_map.get(dedup[pair].get("type"), 0):
                    dedup[pair] = e

        return {
            "type": "star8",
            "name": f"Восьмиконечная звезда Star-8 ({self.num_inputs} входов, глубина {self.ray_depth})",
            "num_inputs": self.num_inputs,
            "ray_depth": self.ray_depth,
            "nodes": nodes,
            "edges": list(dedup.values())
        }

    def reset_topology(self) -> None:
        self._init_nodes()
        self.next_input_idx = 0

    def _record_snapshot(self, new_logs: List[Dict[str, str]]) -> None:
        grid_data = {
            addr: n.get_display_info()
            for addr, n in self.neurons_map.items()
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

    def _restore_from_snapshot(self, s: SimulationSnapshot) -> None:
        self.current_tick = s.tick
        self.total_generated = s.total_generated
        self.completed_count = s.completed_count
        self.rejected_count = s.rejected_count
        self.next_input_idx = s.next_input_idx
        self.all_requests = copy.deepcopy(s.requests)

        for addr, info in s.neurons_state.items():
            n = self.neurons_map.get(addr)
            if n:
                n.is_operational = info.get("is_operational", True)
                n.state = info["state"]
                n.remaining_time = info.get("remaining_time", 0)
                n.q_stage = info.get("q_stage", 0)
                n.processed_requests_count = info.get("processed_requests_count", 0)
                n.unprocessed_requests_count = info.get("unprocessed_requests_count", 0)
                n.trail_color = info["trail_color"]
                n.trail_request_id = info["trail_request_id"]
                n.has_computed = info["has_computed"]
                req_id = info["request_id"]
                n.current_request = next((rq for rq in self.all_requests if rq.id == req_id), None) if req_id else None

    def toggle_neuron_fault(self, row: int, col: int) -> bool:
        self.truncate_future()
        neuron = self.neurons_map.get((row, col))
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

    def _update_current_snapshot(self) -> None:
        grid_data = {
            addr: n.get_display_info()
            for addr, n in self.neurons_map.items()
        }
        self.history[self.current_history_index].neurons_state = grid_data
        self.history[self.current_history_index].requests = copy.deepcopy(self.all_requests)
        self.history[self.current_history_index].rejected_count = self.rejected_count


class HoneycombPipelineArchitecture(BaseArchitecture):
    """
    Гексагональная сотовая конвейерная топология (Honeycomb Pipeline, deg=3).
    - Теоретический оптимум минимальной связности планарных сетей (deg = 3).
    - Разрядность q: слои 0..rows-1 (по умолчанию rows=5, cols=6).
    - Каждый узел [r, c] имеет ровно 3 направленных вперед выхода на следующий слой [r+1]:
        1. Прямо вниз (0°): [r+1, c] (Shortest path)
        2. Влево-вниз (-60°): [r+1, (c-1)%cols] (обход сбоя с переходом на следующий разряд)
        3. Вправо-вниз (+60°): [r+1, (c+1)%cols] (обход сбоя с переходом на следующий разряд)
    - Цилиндрическое тороидальное замыкание по горизонтали (без краевого эффекта).
    - Степень исхода: строго deg+=3 (слои 0..rows-2) и deg+=0 на терминальном слое.
    - При любом обходе конвейер разрядности НЕ теряет тактов и не пятится назад!
    """

    def __init__(
        self,
        rows: int = 5,
        cols: int = 6,
        q_required: int = 4,
        total_initial_requests: int = 6,
        single_use: bool = False
    ):
        super().__init__(single_use=single_use)
        self.rows: int = rows
        self.cols: int = cols
        self.q_required: int = q_required
        self.total_initial_requests = total_initial_requests
        self.next_input_idx: int = 0

        self.neurons: List[List[Neuron]] = [
            [Neuron(r, c) for c in range(cols)]
            for r in range(rows)
        ]

        self._record_snapshot([])
        self.log_event(
            "SYS",
            f"Инициализация соты Honeycomb {rows}x{cols} (deg=3): {total_initial_requests} заявок, q={q_required}."
        )

    def get_neuron(self, address: Any) -> Optional[Neuron]:
        if isinstance(address, (tuple, list)) and len(address) == 2:
            r, c = address
            if 0 <= r < self.rows and 0 <= c < self.cols:
                return self.neurons[r][c]
        return None

    def all_neurons(self) -> List[Neuron]:
        return [self.neurons[r][c] for r in range(self.rows) for c in range(self.cols)]

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

    def inject_initial_requests(self, tick: int) -> List[Request]:
        if self.next_input_idx >= self.total_initial_requests:
            return []

        if tick % 2 == 1:
            col = self.next_input_idx % self.cols
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

    def is_finished(self) -> bool:
        if self.next_input_idx < self.total_initial_requests:
            return False
        return all(r.is_finished() for r in self.all_requests)

    def get_topology_info(self) -> Dict[str, Any]:
        nodes = []
        edges = []

        for r in range(self.rows):
            y = (r + 0.5) / self.rows
            shift = 0.25 if r % 2 == 1 else -0.25
            for c in range(self.cols):
                x = (c + 0.5 + shift) / self.cols
                is_input = (r == 0)
                nodes.append({
                    "address": (r, c),
                    "label": f"[{r}, {c}]",
                    "row": r,
                    "col": c,
                    "x": x,
                    "y": y,
                    "is_input": is_input,
                    "color": "#ffca28" if is_input else "#00bcd4"
                })

                if r < self.rows - 1:
                    edges.append({
                        "from": (r, c),
                        "to": (r + 1, c),
                        "type": "primary",
                        "color": "#4fc3f7"
                    })
                    left_c = (c - 1) % self.cols
                    if c > 0:
                        edges.append({
                            "from": (r, c),
                            "to": (r + 1, left_c),
                            "type": "secondary",
                            "color": "#78909c"
                        })
                    right_c = (c + 1) % self.cols
                    if c < self.cols - 1:
                        edges.append({
                            "from": (r, c),
                            "to": (r + 1, right_c),
                            "type": "secondary",
                            "color": "#78909c"
                        })

        return {
            "type": "honeycomb",
            "name": "Гексагональная сота (Honeycomb, deg=3)",
            "rows": self.rows,
            "cols": self.cols,
            "nodes": nodes,
            "edges": edges
        }

    def reset_topology(self) -> None:
        self.neurons = [
            [Neuron(r, c) for c in range(self.cols)]
            for r in range(self.rows)
        ]
        self.next_input_idx = 0

    def _record_snapshot(self, new_logs: List[Dict[str, str]]) -> None:
        grid_data = [
            [self.neurons[r][c].get_display_info() for c in range(self.cols)]
            for r in range(self.rows)
        ]
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

    def _restore_from_snapshot(self, s: SimulationSnapshot) -> None:
        self.current_tick = s.tick
        self.total_generated = s.total_generated
        self.completed_count = s.completed_count
        self.rejected_count = s.rejected_count
        self.next_input_idx = s.next_input_idx
        self.all_requests = copy.deepcopy(s.requests)

        for r in range(self.rows):
            for c in range(self.cols):
                info = s.neurons_state[r][c]
                n = self.neurons[r][c]
                n.is_operational = info.get("is_operational", True)
                n.state = info["state"]
                n.remaining_time = info.get("remaining_time", 0)
                n.q_stage = info.get("q_stage", 0)
                n.processed_requests_count = info.get("processed_requests_count", 0)
                n.unprocessed_requests_count = info.get("unprocessed_requests_count", 0)
                n.trail_color = info["trail_color"]
                n.trail_request_id = info["trail_request_id"]
                n.has_computed = info["has_computed"]
                req_id = info["request_id"]
                n.current_request = next((rq for rq in self.all_requests if rq.id == req_id), None) if req_id else None

    def toggle_neuron_fault(self, row: int, col: int) -> bool:
        self.truncate_future()
        neuron = self.neurons[row][col]
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

    def _update_current_snapshot(self) -> None:
        grid_data = [
            [self.neurons[r][c].get_display_info() for c in range(self.cols)]
            for r in range(self.rows)
        ]
        self.history[self.current_history_index].neurons_state = grid_data
        self.history[self.current_history_index].requests = copy.deepcopy(self.all_requests)
        self.history[self.current_history_index].rejected_count = self.rejected_count


class SymmetricStar8Architecture(BaseArchitecture):
    """
    Изоморфная симметричная звезда (Symmetric 8-Star Chordal Ring, deg=3).
    - 8 радиальных лучей, строго изоморфных (графовая 8-кратная циклическая симметрия C8).
    - Все 8 лучей имеют АБСОЛЮТНО одинаковые условия и единый алгоритм.
    - На каждом слое d in {1..ray_depth-1} каждый узел (d, k) имеет ровно 3 связи:
        1. Прямо по лучу наружу: (d+1, k) (Primary Shortest Path)
        2. По кольцу вправо (по часовой стрелке): (d, (k+1)%8) (Secondary)
        3. По кольцу влево (против часовой стрелки): (d, (k-1)%8) (Secondary)
    - Степень исхода: строго deg+=3 на промежуточных слоях, deg+=0 на терминальном слое.
    - Полная математическая инвариантность ячеек без асимметрии и без исключений.
    """
    RAY_COLORS = [
        "#ab47bc", "#ffb74d", "#66bb6a", "#26a69a",
        "#3f51b5", "#e91e63", "#9fa8da", "#00bcd4"
    ]

    def __init__(
        self,
        ray_depth: int = 4,
        num_inputs: int = 6,
        q_required: int = 4,
        total_initial_requests: int = 6,
        single_use: bool = False
    ):
        super().__init__(single_use=single_use)
        self.ray_depth = ray_depth
        self.num_inputs = num_inputs
        self.q_required = q_required
        self.total_initial_requests = total_initial_requests
        self.next_input_idx = 0

        self.neurons_map: Dict[Tuple[int, int], Neuron] = {}
        self._init_nodes()

        self._record_snapshot([])
        self.log_event(
            "SYS",
            f"Инициализация изоморфной звезды Sym-Star8 ({num_inputs} входов, 8 лучей, deg=3): {total_initial_requests} заявок, q={q_required}."
        )

    def _init_nodes(self) -> None:
        self.neurons_map.clear()
        for c in range(self.num_inputs):
            self.neurons_map[(0, c)] = Neuron(0, c)

        for d in range(1, self.ray_depth + 1):
            for ray in range(8):
                self.neurons_map[(d, ray)] = Neuron(d, ray)

    def get_neuron(self, address: Any) -> Optional[Neuron]:
        if isinstance(address, (tuple, list)) and len(address) == 2:
            return self.neurons_map.get((address[0], address[1]))
        return None

    def all_neurons(self) -> List[Neuron]:
        return list(self.neurons_map.values())

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

    def inject_initial_requests(self, tick: int) -> List[Request]:
        if self.next_input_idx >= self.total_initial_requests:
            return []

        if tick % 2 == 1:
            col = self.next_input_idx % self.num_inputs
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

    def is_finished(self) -> bool:
        if self.next_input_idx < self.total_initial_requests:
            return False
        return all(r.is_finished() for r in self.all_requests)

    def get_topology_info(self) -> Dict[str, Any]:
        nodes = []
        raw_edges = []

        R_layers = [0.18, 0.28, 0.38, 0.46]
        R_in = 0.09

        for d in range(1, self.ray_depth + 1):
            R = R_layers[min(d - 1, len(R_layers) - 1)]
            for ray in range(8):
                angle = ray * (2 * math.pi / 8.0) - math.pi / 2.0
                x = 0.50 + R * math.cos(angle)
                y = 0.50 + R * math.sin(angle)
                color = self.RAY_COLORS[ray % 8]
                nodes.append({
                    "address": (d, ray),
                    "label": f"[{d}, {ray}]",
                    "row": d,
                    "col": ray,
                    "x": x,
                    "y": y,
                    "is_input": False,
                    "color": color
                })

        for c in range(self.num_inputs):
            angle = c * (2 * math.pi / float(self.num_inputs)) - math.pi / 2.0
            x = 0.50 + R_in * math.cos(angle)
            y = 0.50 + R_in * math.sin(angle)
            nodes.append({
                "address": (0, c),
                "label": f"[0, {c}]",
                "row": 0,
                "col": c,
                "x": x,
                "y": y,
                "is_input": True,
                "color": "#ffca28"
            })

        for c in range(self.num_inputs):
            target_ray = int(round(c * 8.0 / self.num_inputs)) % 8
            for r_idx, kind in [(target_ray, "primary"), ((target_ray + 1) % 8, "secondary"), ((target_ray - 1) % 8, "secondary")]:
                raw_edges.append({
                    "from": (0, c),
                    "to": (1, r_idx),
                    "type": kind,
                    "color": self.RAY_COLORS[r_idx] if kind == "primary" else "#607d8b"
                })

        for ray in range(8):
            color = self.RAY_COLORS[ray]
            for d in range(1, self.ray_depth):
                raw_edges.append({
                    "from": (d, ray),
                    "to": (d + 1, ray),
                    "type": "primary",
                    "color": color
                })

        for d in range(1, self.ray_depth):
            for ray in range(8):
                next_ray = (ray + 1) % 8
                raw_edges.append({
                    "from": (d, ray),
                    "to": (d, next_ray),
                    "type": "secondary",
                    "color": "#607d8b"
                })

        prio = {"primary": 2, "secondary": 1}
        dedup = {}
        for e in raw_edges:
            pair = (min(e["from"], e["to"]), max(e["from"], e["to"]))
            if pair not in dedup or prio.get(e.get("type"), 0) > prio.get(dedup[pair].get("type"), 0):
                dedup[pair] = e

        return {
            "type": "sym_star8",
            "name": "Симметричная звезда C8 (Sym-Star8, deg=3)",
            "num_inputs": self.num_inputs,
            "ray_depth": self.ray_depth,
            "nodes": nodes,
            "edges": list(dedup.values())
        }

    def reset_topology(self) -> None:
        self._init_nodes()
        self.next_input_idx = 0

    def _record_snapshot(self, new_logs: List[Dict[str, str]]) -> None:
        grid_data = {
            addr: n.get_display_info()
            for addr, n in self.neurons_map.items()
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

    def _restore_from_snapshot(self, s: SimulationSnapshot) -> None:
        self.current_tick = s.tick
        self.total_generated = s.total_generated
        self.completed_count = s.completed_count
        self.rejected_count = s.rejected_count
        self.next_input_idx = s.next_input_idx
        self.all_requests = copy.deepcopy(s.requests)

        for addr, info in s.neurons_state.items():
            n = self.neurons_map.get(addr)
            if n:
                n.is_operational = info.get("is_operational", True)
                n.state = info["state"]
                n.remaining_time = info.get("remaining_time", 0)
                n.q_stage = info.get("q_stage", 0)
                n.processed_requests_count = info.get("processed_requests_count", 0)
                n.unprocessed_requests_count = info.get("unprocessed_requests_count", 0)
                n.trail_color = info["trail_color"]
                n.trail_request_id = info["trail_request_id"]
                n.has_computed = info["has_computed"]
                req_id = info["request_id"]
                n.current_request = next((rq for rq in self.all_requests if rq.id == req_id), None) if req_id else None

    def toggle_neuron_fault(self, row: int, col: int) -> bool:
        self.truncate_future()
        neuron = self.neurons_map.get((row, col))
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

    def _update_current_snapshot(self) -> None:
        grid_data = {
            addr: n.get_display_info()
            for addr, n in self.neurons_map.items()
        }
        self.history[self.current_history_index].neurons_state = grid_data
        self.history[self.current_history_index].requests = copy.deepcopy(self.all_requests)
        self.history[self.current_history_index].rejected_count = self.rejected_count

class SimulationController:
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
