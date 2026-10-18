"""
ООП-модель нейронной сети самомаршрутизации (НС АЦП).

Архитектура:
- NeuronState / RequestStatus: Перечисления состояний.
- Request: Объект измерительной заявки (инкапсулирует параметры квантования, путь и тайминги).
- Neuron: Объект ОИН (инкапсулирует вычисления на 10 тактов, однократность работы и сбои).
- GridTopology: Топология сети (инкапсулирует геометрию, тороидальные переходы и приоритет соседей).
- SimulationSnapshot: Снимок состояния для паттерна Memento (машина времени).
- SimulationEngine: Фасад и координатор дискретно-событийного моделирования.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class NeuronState(Enum):
    FREE = "Свободен"
    BUSY = "Занят"
    EXHAUSTED = "Завершен"  # Вычисления завершены, повторно не участвует
    FAULTY = "Сбой"


class RequestStatus(Enum):
    PENDING = "Ожидание"
    IN_PROGRESS = "Обработка"
    COMPLETED = "Успешно"
    REJECTED = "Отказ"


PALETTE = [
    "#00BCD4",  # Cyan
    "#FFB300",  # Amber
    "#AB47BC",  # Purple
    "#26A69A",  # Teal
]


class Request:
    """Инкапсулирует состояние и жизненный цикл заявки."""

    def __init__(
        self,
        request_id: int,
        q_required: int = 6,
        calc_ticks_per_neuron: int = 10,
        color: Optional[str] = None,
        created_at_tick: int = 0
    ):
        self.id: int = request_id
        self.q_required: int = q_required
        self.calc_ticks_per_neuron: int = calc_ticks_per_neuron
        self.color: str = color or PALETTE[(request_id - 1) % len(PALETTE)]
        self.status: RequestStatus = RequestStatus.PENDING
        self.path: List[Tuple[int, int]] = []
        self.current_step: int = 0
        self.created_at_tick: int = created_at_tick
        self.completed_at_tick: Optional[int] = None
        self.reject_reason: str = ""

    @classmethod
    def create(
        cls,
        request_id: int,
        q_required: int = 6,
        calc_ticks: int = 10,
        tick: int = 0
    ) -> Request:
        """Фабричный метод создания заявки."""
        return cls(
            request_id=request_id,
            q_required=q_required,
            calc_ticks_per_neuron=calc_ticks,
            created_at_tick=tick
        )

    def attach_to_neuron(self, row: int, col: int) -> None:
        """Регистрация перехода в нейрон."""
        self.path.append((row, col))
        self.current_step += 1
        self.status = RequestStatus.IN_PROGRESS

    def mark_completed(self, tick: int) -> None:
        """Успешное завершение всех этапов квантования."""
        self.status = RequestStatus.COMPLETED
        self.completed_at_tick = tick

    def mark_rejected(self, tick: int, reason: str) -> None:
        """Фиксация отказа в обслуживании."""
        self.status = RequestStatus.REJECTED
        self.completed_at_tick = tick
        self.reject_reason = reason

    def is_finished(self) -> bool:
        """Проверка выполнения необходимого числа каскадов."""
        return self.current_step >= self.q_required

    def get_path_string(self) -> str:
        return "->".join([f"[{r},{c}]" for r, c in self.path])

    def __repr__(self) -> str:
        return f"<Request #{self.id} step={self.current_step}/{self.q_required} {self.status.value}>"


class Neuron:
    """
    Основной Измерительный Нейрон (ОИН).
    Каждый расчет длится 10 тиков.
    Нейрон, выполнивший расчет, больше вычислений не проводит.
    """

    DEFAULT_CALC_TICKS: int = 10

    def __init__(self, row: int, col: int, default_calc_ticks: int = DEFAULT_CALC_TICKS):
        self.row: int = row
        self.col: int = col
        self.default_calc_ticks: int = default_calc_ticks

        self.state: NeuronState = NeuronState.FREE
        self.ticks_left: int = 0
        self.current_request: Optional[Request] = None
        self.trail_color: Optional[str] = None
        self.trail_request_id: Optional[int] = None
        self.has_computed: bool = False

    @property
    def address(self) -> str:
        return f"[{self.row}, {self.col}]"

    def can_accept(self) -> bool:
        """Готовность узла принять новую заявку."""
        return self.state == NeuronState.FREE and not self.has_computed

    def accept_request(self, request: Request) -> bool:
        """Принятие заявки и запуск вычислений на 10 тиков."""
        if not self.can_accept():
            return False

        self.current_request = request
        self.state = NeuronState.BUSY
        self.ticks_left = max(1, self.default_calc_ticks)

        # Сохранение следа
        self.trail_color = request.color
        self.trail_request_id = request.id

        request.attach_to_neuron(self.row, self.col)
        return True

    def process_tick(self) -> bool:
        """
        Один такт вычислений в нейроне.
        Возвращает True, если вычисления завершены на этом такте.
        """
        if self.state != NeuronState.BUSY or not self.current_request:
            return False

        self.ticks_left -= 1
        return self.ticks_left <= 0

    def finalize_computation(self) -> Optional[Request]:
        """Завершение вычисления: узел переходит в EXHAUSTED и больше не вычисляет."""
        req = self.current_request
        self.current_request = None
        self.has_computed = True
        if self.state != NeuronState.FAULTY:
            self.state = NeuronState.EXHAUSTED
        self.ticks_left = 0
        return req

    def toggle_fault(self) -> Tuple[bool, Optional[Request]]:
        """Ручное переключение состояния исправности (клик)."""
        dropped_req = None
        if self.state == NeuronState.FAULTY:
            self.state = NeuronState.EXHAUSTED if self.has_computed else NeuronState.FREE
            self.ticks_left = 0
            self.current_request = None
            return False, None
        else:
            if self.current_request:
                dropped_req = self.current_request
            self.state = NeuronState.FAULTY
            self.ticks_left = 0
            self.current_request = None
            return True, dropped_req

    def get_display_info(self) -> Dict[str, any]:
        return {
            "address": self.address,
            "row": self.row,
            "col": self.col,
            "state": self.state,
            "ticks_left": self.ticks_left,
            "request_id": self.current_request.id if self.current_request else None,
            "request_color": self.current_request.color if self.current_request else None,
            "trail_color": self.trail_color,
            "trail_request_id": self.trail_request_id,
            "has_computed": self.has_computed,
        }

    def __repr__(self) -> str:
        return f"<Neuron {self.address} {self.state.value} T={self.ticks_left}>"


class GridTopology:
    """Инкапсулирует геометрию тороидальной матрицы и правила поиска соседей."""

    def __init__(self, rows: int = 6, cols: int = 4):
        self.rows: int = rows
        self.cols: int = cols

    def get_candidate_neighbors(self, current_row: int, current_col: int) -> List[Tuple[int, int, str]]:
        """
        Возвращает нижних соседей строго в заданном порядке приоритета:
        1) Свой нижний (по вертикали прямо)
        2) Левый нижний
        3) Правый нижний
        Замыкание тороидальное (wrap-around).
        """
        next_row = (current_row + 1) % self.rows
        col_center = current_col
        col_left = (current_col - 1) % self.cols
        col_right = (current_col + 1) % self.cols

        return [
            (next_row, col_center, "центр"),
            (next_row, col_left, "лево"),
            (next_row, col_right, "право"),
        ]


@dataclass
class SimulationSnapshot:
    """Снимок состояния системы (Паттерн Memento)."""
    tick: int
    neurons_state: List[List[Dict[str, any]]]
    requests: List[Request]
    active_count: int
    completed_count: int
    rejected_count: int
    total_generated: int
    new_logs: List[Dict[str, str]]
    next_input_idx: int

    @property
    def active_requests_count(self) -> int:
        return self.active_count


class SimulationEngine:
    """
    Фасад и координатор симулятора.
    Управляет жизненным циклом тактов, очередью 4 заявок и историей.
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 4,
        calc_ticks: int = 10,
        q_required: int = 6,
        total_initial_requests: int = 4
    ):
        self.topology: GridTopology = GridTopology(rows=rows, cols=cols)
        self.calc_ticks: int = calc_ticks
        self.q_required: int = q_required
        self.total_initial_requests: int = total_initial_requests

        self.current_tick: int = 0
        self.next_input_idx: int = 0

        self.total_generated: int = 0
        self.completed_count: int = 0
        self.rejected_count: int = 0

        self.neurons: List[List[Neuron]] = [
            [Neuron(r, c, default_calc_ticks=calc_ticks) for c in range(cols)]
            for r in range(rows)
        ]

        self.all_requests: List[Request] = []
        self.logs: List[Dict[str, str]] = []
        self.history: List[SimulationSnapshot] = []
        self.current_history_index: int = 0

        self._record_snapshot([])
        self.log_event("SYS", f"Инициализация: {rows}x{cols}, 4 заявки, 10 тиков на расчет, ввод через такт.")

    @property
    def rows(self) -> int:
        return self.topology.rows

    @property
    def cols(self) -> int:
        return self.topology.cols

    def log_event(self, event_type: str, message: str) -> None:
        self.logs.append({
            "tick": self.current_tick,
            "type": event_type,
            "msg": message
        })

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

    def is_finished(self) -> bool:
        """Завершена ли обработка всех 4 заявок."""
        if self.next_input_idx < self.total_initial_requests:
            return False
        return all(r.status in (RequestStatus.COMPLETED, RequestStatus.REJECTED) for r in self.all_requests)

    def toggle_neuron_fault(self, row: int, col: int) -> bool:
        """Переключение сбоя нейрона."""
        neuron = self.neurons[row][col]
        is_faulty, dropped_req = neuron.toggle_fault()

        if is_faulty:
            self.log_event("FAULT", f"Нейрон {neuron.address} отключен.")
            if dropped_req:
                self.rejected_count += 1
                dropped_req.mark_rejected(self.current_tick, f"Сбой узла {neuron.address}")
                self.log_event("REJECT", f"Отказ: Заявка #{dropped_req.id} прервана из-за сбоя {neuron.address}.")
        else:
            self.log_event("SYS", f"Нейрон {neuron.address} включен.")

        self._update_current_snapshot()
        return is_faulty

    def _update_current_snapshot(self) -> None:
        if not self.history:
            return
        grid_data = [
            [self.neurons[r][c].get_display_info() for c in range(self.cols)]
            for r in range(self.rows)
        ]
        self.history[self.current_history_index].neurons_state = grid_data
        self.history[self.current_history_index].requests = copy.deepcopy(self.all_requests)
        self.history[self.current_history_index].rejected_count = self.rejected_count

    def _inject_next_initial_request(self) -> None:
        """Ввод очередной заявки из 4 стартовых."""
        if self.next_input_idx >= self.total_initial_requests:
            return

        col = self.next_input_idx
        req_id = self.next_input_idx + 1
        self.next_input_idx += 1

        req = Request.create(
            request_id=req_id,
            q_required=self.q_required,
            calc_ticks=self.calc_ticks,
            tick=self.current_tick
        )
        self.total_generated += 1
        self.all_requests.append(req)

        entry_neuron = self.neurons[0][col]
        if entry_neuron.accept_request(req):
            self.log_event("INPUT", f"Заявка #{req.id} вошла в {entry_neuron.address} (расчет: 10 тиков).")
        else:
            req.mark_rejected(self.current_tick, f"Вход {entry_neuron.address} недоступен")
            self.rejected_count += 1
            self.log_event("REJECT", f"Отказ: вход {entry_neuron.address} недоступен для заявки #{req.id}.")

    def tick(self) -> None:
        """Один шаг дискретного времени."""
        if self.is_finished():
            return

        self.current_tick += 1
        cycle_logs_start = len(self.logs)

        # Подача 4 заявок через такт: на тактах 1, 3, 5, 7
        if self.next_input_idx < self.total_initial_requests:
            if (self.current_tick % 2 == 1):
                self._inject_next_initial_request()

        # 1. Потактовые вычисления в активных нейронах
        busy_neurons: List[Neuron] = [
            self.neurons[r][c]
            for r in range(self.rows)
            for c in range(self.cols)
            if self.neurons[r][c].state == NeuronState.BUSY
        ]

        ready_neurons: List[Neuron] = []
        for n in busy_neurons:
            if n.process_tick():
                ready_neurons.append(n)

        # 2. Маршрутизация заявок, закончивших расчет
        for n in ready_neurons:
            req = n.current_request
            if not req:
                continue

            # Проверка завершения всех q каскадов
            if req.is_finished():
                req.mark_completed(self.current_tick)
                self.completed_count += 1
                n.finalize_computation()
                self.log_event("DONE", f"Заявка #{req.id} завершена. Путь: {req.get_path_string()}")
                continue

            # Поиск кандидатов через GridTopology (свой нижний -> левый -> правый)
            candidates = self.topology.get_candidate_neighbors(n.row, n.col)

            routed = False
            for target_r, target_c, direction in candidates:
                target = self.neurons[target_r][target_c]
                if target.can_accept():
                    n.finalize_computation()
                    target.accept_request(req)
                    routed = True
                    self.log_event("ROUTE", f"Заявка #{req.id}: {n.address} -> {target.address} ({direction}).")
                    break

            if not routed:
                req.mark_rejected(self.current_tick, "Все нижние кандидаты недоступны")
                self.rejected_count += 1
                n.finalize_computation()
                cand_str = ", ".join([f"[{r},{c}]" for r, c, _ in candidates])
                self.log_event("REJECT", f"Отказ: Заявка #{req.id} остановлена в {n.address}. Соседи {cand_str} недоступны.")

        new_logs = self.logs[cycle_logs_start:]
        self._record_snapshot(new_logs)

    def step_forward(self) -> None:
        if self.current_history_index < len(self.history) - 1:
            self.current_history_index += 1
            self._restore_from_snapshot(self.history[self.current_history_index])
        else:
            self.tick()

    def step_backward(self) -> None:
        if self.current_history_index > 0:
            self.current_history_index -= 1
            self._restore_from_snapshot(self.history[self.current_history_index])

    def fast_forward_to_end(self) -> None:
        """Перемещение на финальный такт симуляции."""
        limit = 300
        while not self.is_finished() and limit > 0:
            self.step_forward()
            limit -= 1
        self.go_to_history_tick(len(self.history) - 1)

    def go_to_history_tick(self, idx: int) -> None:
        if 0 <= idx < len(self.history):
            self.current_history_index = idx
            self._restore_from_snapshot(self.history[idx])

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
                n.state = info["state"]
                n.ticks_left = info["ticks_left"]
                n.trail_color = info["trail_color"]
                n.trail_request_id = info["trail_request_id"]
                n.has_computed = info["has_computed"]
                req_id = info["request_id"]
                n.current_request = next((rq for rq in self.all_requests if rq.id == req_id), None) if req_id else None

    def reset(self) -> None:
        self.current_tick = 0
        self.next_input_idx = 0
        self.total_generated = 0
        self.completed_count = 0
        self.rejected_count = 0
        self.neurons = [
            [Neuron(r, c, default_calc_ticks=self.calc_ticks) for c in range(self.cols)]
            for r in range(self.rows)
        ]
        self.all_requests.clear()
        self.logs.clear()
        self.history.clear()
        self._record_snapshot([])
        self.log_event("SYS", "Сброс симулятора 6x4.")
