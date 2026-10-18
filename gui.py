"""
Интерфейс самомаршрутизации (6x4) на PySide6.
Нейтральное оформление, компактное размещение элементов, отсутствие лишнего текста.
"""

from __future__ import annotations
import sys
from typing import Dict, List, Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QSpinBox,
    QTextEdit, QSlider, QSplitter
)

from models import NeuronState, RequestStatus, SimulationEngine


class NeuronButton(QPushButton):
    """Кнопка нейрона в сетке 6x4."""
    def __init__(self, row: int, col: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.row = row
        self.col = col
        self.setMinimumSize(80, 60)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 9))
        self.update_view({
            "address": f"[{row}, {col}]",
            "state": NeuronState.FREE,
            "ticks_left": 0,
            "request_id": None,
            "request_color": None,
            "trail_color": None,
            "trail_request_id": None,
            "has_computed": False
        })

    def update_view(self, info: Dict[str, any]) -> None:
        state: NeuronState = info["state"]
        ticks_left: int = info["ticks_left"]
        req_id = info["request_id"]
        req_color = info["request_color"]
        trail_color = info["trail_color"]
        trail_req_id = info["trail_request_id"]
        has_computed: bool = info["has_computed"]
        addr = info["address"]

        if state == NeuronState.FAULTY:
            text = f"{addr}\nСбой"
            style = """
                QPushButton {
                    background-color: #381b1d;
                    color: #ef5350;
                    border: 1px solid #c62828;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #4a2225;
                }
            """
        elif state == NeuronState.BUSY:
            color = req_color or "#00BCD4"
            text = f"{addr}\n#{req_id} (T:{ticks_left})"
            style = f"""
                QPushButton {{
                    background-color: {self._hex_to_rgba(color, 0.25)};
                    color: #ffffff;
                    border: 2px solid {color};
                    border-radius: 4px;
                    font-weight: bold;
                }}
            """
        elif state == NeuronState.EXHAUSTED or has_computed:
            color = trail_color or "#546e7a"
            text = f"{addr}\nЗавершен (#{trail_req_id})"
            style = f"""
                QPushButton {{
                    background-color: {self._hex_to_rgba(color, 0.12)};
                    color: #90a4ae;
                    border: 1px solid {self._hex_to_rgba(color, 0.4)};
                    border-radius: 4px;
                }}
            """
        else:
            text = f"{addr}\nСвободен"
            style = """
                QPushButton {
                    background-color: #242424;
                    color: #888888;
                    border: 1px solid #333333;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #2e2e2e;
                    border-color: #555555;
                    color: #cccccc;
                }
            """

        self.setText(text)
        self.setStyleSheet(style)

    @staticmethod
    def _hex_to_rgba(hex_code: str, alpha: float) -> str:
        hex_code = hex_code.lstrip("#")
        if len(hex_code) == 6:
            r = int(hex_code[0:2], 16)
            g = int(hex_code[2:4], 16)
            b = int(hex_code[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
        return f"rgba(0, 188, 212, {alpha})"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Симулятор 6x4")
        self.resize(980, 580)
        self.setMinimumSize(850, 480)

        # 6 строк, 4 столбца, q=6 шагов, 10 тактов на вычисление каждым нейроном
        self.sim = SimulationEngine(rows=6, cols=4, calc_ticks=10, q_required=6)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self.neuron_buttons: List[List[NeuronButton]] = []

        self._init_ui()
        self._apply_styles()
        self.refresh_all()

    def _init_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Панель управления (компактная, без баннеров)
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(6)

        self.btn_prev = QPushButton("Назад")
        self.btn_prev.clicked.connect(self.step_backward)
        ctrl_bar.addWidget(self.btn_prev)

        self.btn_play = QPushButton("Пуск")
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_bar.addWidget(self.btn_play)

        self.btn_next = QPushButton("Вперед")
        self.btn_next.clicked.connect(self.step_forward)
        ctrl_bar.addWidget(self.btn_next)

        self.btn_end = QPushButton("В конец")
        self.btn_end.clicked.connect(self.fast_forward)
        ctrl_bar.addWidget(self.btn_end)

        self.btn_reset = QPushButton("Сброс")
        self.btn_reset.clicked.connect(self.reset_simulation)
        ctrl_bar.addWidget(self.btn_reset)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setValue(0)
        self.timeline_slider.valueChanged.connect(self._on_slider_moved)
        ctrl_bar.addWidget(self.timeline_slider, 2)

        self.lbl_tick = QLabel("Такт: 0 / 0")
        self.lbl_tick.setMinimumWidth(80)
        ctrl_bar.addWidget(self.lbl_tick)

        ctrl_bar.addWidget(QLabel("Интервал:"))
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(50, 2000)
        self.spin_speed.setSingleStep(50)
        self.spin_speed.setValue(400)
        self.spin_speed.setSuffix(" мс")
        self.spin_speed.valueChanged.connect(self._on_speed_changed)
        ctrl_bar.addWidget(self.spin_speed)

        main_layout.addLayout(ctrl_bar)

        # Рабочая область: Сетка слева, панель справа
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)

        # Сетка нейронов 6x4
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(5)

        self.neuron_buttons = []
        for r in range(self.sim.rows):
            row_btns = []
            for c in range(self.sim.cols):
                btn = NeuronButton(r, c)
                btn.clicked.connect(lambda ch=False, row=r, col=c: self._on_neuron_clicked(row, col))
                grid_layout.addWidget(btn, r, c)
                row_btns.append(btn)
            self.neuron_buttons.append(row_btns)

        splitter.addWidget(grid_widget)

        # Правая панель: статистика и лог
        side_widget = QWidget()
        side_layout = QVBoxLayout(side_widget)
        side_layout.setContentsMargins(8, 0, 0, 0)
        side_layout.setSpacing(6)

        # Статистика (простая сетка, без рамок)
        stats_layout = QGridLayout()
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(4)

        self.kpi_total = QLabel("4")
        self.kpi_active = QLabel("0")
        self.kpi_completed = QLabel("0")
        self.kpi_rejected = QLabel("0")
        self.kpi_fail_rate = QLabel("0.0 %")
        self.kpi_faults = QLabel("0 / 24")

        stats_layout.addWidget(QLabel("Заявок:"), 0, 0)
        stats_layout.addWidget(self.kpi_total, 0, 1)

        stats_layout.addWidget(QLabel("Активно:"), 0, 2)
        stats_layout.addWidget(self.kpi_active, 0, 3)

        stats_layout.addWidget(QLabel("Завершено:"), 1, 0)
        stats_layout.addWidget(self.kpi_completed, 1, 1)

        stats_layout.addWidget(QLabel("Отказов:"), 1, 2)
        stats_layout.addWidget(self.kpi_rejected, 1, 3)

        stats_layout.addWidget(QLabel("P_отк:"), 2, 0)
        stats_layout.addWidget(self.kpi_fail_rate, 2, 1)

        stats_layout.addWidget(QLabel("Сбоев:"), 2, 2)
        stats_layout.addWidget(self.kpi_faults, 2, 3)

        side_layout.addLayout(stats_layout)

        # Лог
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 9))
        side_layout.addWidget(self.log_view, 1)

        splitter.addWidget(side_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter, 1)

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: #dcdcdc;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel {
                background: transparent;
                color: #b0b0b0;
            }
            QPushButton {
                background-color: #2b2b2b;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 4px 10px;
                color: #dcdcdc;
            }
            QPushButton:hover {
                background-color: #363636;
                border-color: #555555;
            }
            QSpinBox {
                background-color: #242424;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 2px 6px;
                color: #dcdcdc;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #333333;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #555555;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #888888;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QTextEdit {
                background-color: #141414;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
                color: #cccccc;
                padding: 4px;
            }
            QScrollBar:vertical {
                background: #181818;
                width: 6px;
            }
            QScrollBar::handle:vertical {
                background: #333333;
                border-radius: 3px;
            }
        """)

    def _on_neuron_clicked(self, row: int, col: int) -> None:
        self.sim.toggle_neuron_fault(row, col)
        self.refresh_all()

    def _on_timer_tick(self) -> None:
        if self.sim.is_finished():
            self.toggle_play()
            return
        self.step_forward()

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Пуск")
        else:
            self.timer.start(self.spin_speed.value())
            self.btn_play.setText("Пауза")

    def _on_speed_changed(self, val: int) -> None:
        if self.timer.isActive():
            self.timer.setInterval(val)

    def step_forward(self) -> None:
        self.sim.step_forward()
        self.refresh_all()

    def step_backward(self) -> None:
        if self.timer.isActive():
            self.toggle_play()
        self.sim.step_backward()
        self.refresh_all()

    def fast_forward(self) -> None:
        if self.timer.isActive():
            self.toggle_play()
        self.sim.fast_forward_to_end()
        self.refresh_all()

    def _on_slider_moved(self, val: int) -> None:
        if val != self.sim.current_history_index:
            self.sim.go_to_history_tick(val)
            self.refresh_all()

    def reset_simulation(self) -> None:
        if self.timer.isActive():
            self.toggle_play()
        self.sim.reset()
        self.refresh_all()

    def refresh_all(self) -> None:
        snapshot = self.sim.history[self.sim.current_history_index]
        grid_data = snapshot.neurons_state
        fault_count = 0

        for r in range(self.sim.rows):
            for c in range(self.sim.cols):
                info = grid_data[r][c]
                self.neuron_buttons[r][c].update_view(info)
                if info["state"] == NeuronState.FAULTY:
                    fault_count += 1

        total_hist = len(self.sim.history) - 1
        curr_idx = self.sim.current_history_index

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, total_hist)
        self.timeline_slider.setValue(curr_idx)
        self.timeline_slider.blockSignals(False)

        self.lbl_tick.setText(f"Такт: {self.sim.current_tick} ({curr_idx}/{total_hist})")

        total_req = self.sim.total_initial_requests
        self.kpi_total.setText(str(total_req))
        self.kpi_active.setText(str(snapshot.active_requests_count))
        self.kpi_completed.setText(str(self.sim.completed_count))
        self.kpi_rejected.setText(str(self.sim.rejected_count))
        self.kpi_faults.setText(f"{fault_count} / {self.sim.rows * self.sim.cols}")

        if self.sim.total_generated > 0:
            rate = (self.sim.rejected_count / self.sim.total_generated) * 100
            self.kpi_fail_rate.setText(f"{rate:.1f} %")
        else:
            self.kpi_fail_rate.setText("0.0 %")

        self._update_logs()

    def _update_logs(self) -> None:
        html_lines = []
        for entry in self.sim.logs:
            if entry["tick"] > self.sim.current_tick:
                continue

            etype = entry["type"]
            msg = entry["msg"]
            t = entry["tick"]

            if etype == "FAULT" or etype == "REJECT":
                color = "#ef5350"
            elif etype == "DONE":
                color = "#66bb6a"
            elif etype == "ROUTE":
                color = "#4fc3f7"
            elif etype == "INPUT":
                color = "#ffca28"
            else:
                color = "#9e9e9e"

            html_lines.append(
                f"<div style='margin-bottom: 2px;'>"
                f"<span style='color: #666666;'>[{t:02d}]</span> "
                f"<span style='color: {color};'>[{etype}]</span> "
                f"<span style='color: #cccccc;'>{msg}</span>"
                f"</div>"
            )

        self.log_view.setHtml("".join(html_lines))
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def run_gui():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
