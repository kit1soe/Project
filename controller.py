"""
Контроллер симуляции НС АЦП (паттерн MVC).
Отвечает за логику управления потактовой симуляцией, подписку на события
и связь между моделью (models.py) и представлением (gui.py).
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from models import ArrivalMode, BaseArchitecture, PlayState, RequestStatus


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
        self.state = PlayState.FINISHED if self.arch.is_finished() else PlayState.PAUSED
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

    def randomize_faults(self, count: int = 1, probability: float = 1.0, seed: Optional[int] = None) -> list[str]:
        changed = self.arch.randomize_faults(count=count, probability=probability, seed=seed)
        self.notify("tick", self.arch.current_tick)
        return changed

    def schedule_random_faults(self, count: int = 1, probability: float = 1.0, seed: Optional[int] = None) -> dict:
        plan = self.arch.schedule_random_faults(count=count, probability=probability, seed=seed)
        self.notify("tick", self.arch.current_tick)
        return plan

    def set_speed(self, speed_ms: int) -> None:
        self.speed_ms = speed_ms

    def set_return_to_origin(self, val: bool) -> None:
        """Переключение режима маршрутизации (возврат в ячейку / движение вниз при смещении)."""
        self.arch.set_return_to_origin(val)
        self.notify("state_change", self.state)

    def set_generation_ticks(self, ticks: Optional[List[int]]) -> None:
        """Установка заданных тактов генерации заявок."""
        self.arch.set_generation_ticks(ticks)

    def set_arrival_process(
        self,
        mode: str | ArrivalMode,
        rate: float = 0.5,
        seed: Optional[int] = None,
        generation_ticks: Optional[List[int]] = None,
    ) -> None:
        """Настройка детерминированного или пуассоновского потока заявок."""
        self.arch.set_arrival_process(
            mode=mode,
            rate=rate,
            seed=seed,
            generation_ticks=generation_ticks,
        )
        self.notify("tick", self.arch.current_tick)

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
