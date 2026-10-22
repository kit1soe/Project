"""
Интерфейс самомаршрутизации НС АЦП на PySide6.
Поддерживает переключение топологий (свитчер):
- Тороидальная матрица 6x4
- Ромбовидная архитектура
- Восьмиконечная звезда (Star-8 с 6 входами и 8 лучами)
Отрисовывает граф возможных путей (сплошные и пунктирные линии) и активные маршруты.
"""

from __future__ import annotations
import sys
from typing import Any, Dict, List, Optional, Tuple
from PySide6.QtCore import Qt, QTimer, QPoint, QRectF
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QSpinBox, QComboBox,
    QTextEdit, QSlider, QSplitter
)

from models import (
    NeuronState, RequestStatus, PlayState,
    BaseArchitecture, ToroidalGridArchitecture, RhomboidArchitecture, Star8Architecture,
    HoneycombPipelineArchitecture, SymmetricStar8Architecture,
    Funnel8Architecture, ChordalFunnel8Architecture,
    SimulationController
)


class NeuronButton(QPushButton):
    """Интерактивная кнопка-узел нейрона."""

    def __init__(self, address: Tuple[int, int], label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.address = address
        self.node_label = label
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 8))
        self.update_view({
            "address": label,
            "state": NeuronState.FREE,
            "ticks_left": 0,
            "remaining_time": 0,
            "q_stage": 0,
            "request_id": None,
            "request_color": None,
            "trail_color": None,
            "trail_request_id": None,
            "has_computed": False
        })

    def update_view(self, info: Dict[str, Any]) -> None:
        state: NeuronState = info.get("state", NeuronState.FREE)
        ticks_left: int = info.get("remaining_time", info.get("ticks_left", 0))
        q_stage: int = info.get("q_stage", 0)
        req_id = info.get("request_id")
        req_color = info.get("request_color")
        trail_color = info.get("trail_color")
        trail_req_id = info.get("trail_request_id")
        has_computed: bool = info.get("has_computed", False)
        addr = info.get("address", self.node_label)
        if isinstance(addr, (tuple, list)) and len(addr) == 2:
            addr_str = f"[{addr[0]}, {addr[1]}]"
        else:
            addr_str = str(addr)

        if state == NeuronState.FAULTY:
            text = f"{addr_str}\nСБОЙ"
            bg, fg, border, hover = "#381b1d", "#ef5350", "1px solid #c62828", "background-color: #4a2225;"
        elif state == NeuronState.BUSY:
            c = req_color or "#00BCD4"
            stage_str = f"·q{q_stage}" if q_stage > 0 else ""
            text = f"{addr_str}\n#{req_id}{stage_str} (T:{ticks_left})"
            bg, fg, border, hover = self._hex_to_rgba(c, 0.32), "#ffffff", f"2px solid {c}", ""
        elif state == NeuronState.EXHAUSTED or has_computed:
            c = trail_color or "#546e7a"
            tag = f"Готов #{trail_req_id}" if trail_req_id else "Готов"
            text = f"{addr_str}\n{tag}"
            bg, fg, border, hover = self._hex_to_rgba(c, 0.14), "#90a4ae", f"1px solid {self._hex_to_rgba(c, 0.45)}", ""
        elif trail_req_id:
            c = trail_color or "#546e7a"
            text = f"{addr_str}\nСлед #{trail_req_id}"
            bg, fg, border, hover = self._hex_to_rgba(c, 0.08), "#a0a0a0", f"1px solid {self._hex_to_rgba(c, 0.25)}", f"background-color: {self._hex_to_rgba(c, 0.16)};"
        else:
            text = f"{addr_str}\nСвободен"
            bg, fg, border, hover = "#21252b", "#888888", "1px solid #323842", "background-color: #2c313a; border-color: #525c6a; color: #dcdcdc;"

        hover_rule = f"QPushButton:hover {{ {hover} }}" if hover else ""
        self.setText(text)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: {border};
                border-radius: 5px;
                padding: 1px 1px;
                font-size: 7.5pt;
                font-weight: bold;
            }}
            {hover_rule}
        """)

        is_operational = info.get("is_operational", state != NeuronState.FAULTY)
        processed = info.get("processed_requests_count", 0)
        unprocessed = info.get("unprocessed_requests_count", 0)

        self.setToolTip(
            f"Адрес: {addr}\n"
            f"Состояние: {'Исправен' if is_operational else 'Сбой'}\n"
            f"Статус: {state.value}\n"
            f"Текущая заявка: #{req_id if req_id else 'нет'} (звено {q_stage})\n"
            f"Оставшееся время жизни: {ticks_left} т.\n"
            f"Обработано этим узлом: {processed}\n"
            f"Отказов на этом узле: {unprocessed}"
        )

    @staticmethod
    def _hex_to_rgba(hex_code: str, alpha: float) -> str:
        hex_code = hex_code.lstrip("#")
        if len(hex_code) == 6:
            r = int(hex_code[0:2], 16)
            g = int(hex_code[2:4], 16)
            b = int(hex_code[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
        return f"rgba(0, 188, 212, {alpha})"


class TopologyCanvasWidget(QWidget):
    """
    Адаптивный графический холст.
    Отрисовывает:
    - Сплошные магистральные связи (primary channels).
    - Пунктирные обходные/латеральные связи (lateral bypass lines).
    - Активные траектории движения заявок (подсветка цветом заявки).
    - Интерактивные кнопки нейронов поверх графа связей.
    """

    def __init__(self, ctrl: SimulationController, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.buttons: Dict[Tuple[int, int], NeuronButton] = {}
        self.nodes_data: List[Dict[str, Any]] = []
        self.edges_data: List[Dict[str, Any]] = []
        self.node_positions: Dict[Tuple[int, int], QPoint] = {}
        self.topo_type: str = "toroidal_grid"
        self.setMinimumSize(540, 440)
        self.rebuild_canvas()

    def set_controller(self, ctrl: SimulationController) -> None:
        self.ctrl = ctrl
        self.rebuild_canvas()

    def rebuild_canvas(self) -> None:
        """Перестроение кнопок и связей при смене топологии."""
        # Удаляем старые кнопки
        for btn in self.buttons.values():
            btn.setParent(None)
            btn.deleteLater()
        self.buttons.clear()
        self.node_positions.clear()

        topo = self.ctrl.arch.get_topology_info()
        self.topo_type = topo.get("type", "toroidal_grid")
        self.nodes_data = topo.get("nodes", [])
        self.edges_data = topo.get("edges", [])

        # Создаем новые кнопки
        for node in self.nodes_data:
            addr = node["address"]
            label = node.get("label", f"[{addr[0]}, {addr[1]}]")
            btn = NeuronButton(address=addr, label=label, parent=self)
            btn.clicked.connect(lambda ch=False, a=addr: self.ctrl.toggle_fault(a[0], a[1]))
            btn.show()
            self.buttons[addr] = btn

        self._calculate_positions()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._calculate_positions()

    def _calculate_positions(self) -> None:
        w = self.width()
        h = self.height()
        if w < 100 or h < 100 or not self.nodes_data:
            return

        pad_x = 24
        pad_y = 22
        avail_w = max(100, w - 2 * pad_x)
        avail_h = max(100, h - 2 * pad_y)

        if self.topo_type in ("star8", "sym_star8"):
            btn_w = min(56, max(44, int(avail_w * 0.065)))
            btn_h = min(36, max(28, int(avail_h * 0.050)))
        elif self.topo_type in ("honeycomb", "funnel8", "chordal_funnel8"):
            btn_w = min(68, max(46, int(avail_w * 0.085)))
            btn_h = min(38, max(28, int(avail_h * 0.055)))
        else:
            btn_w = min(76, max(54, int(avail_w * 0.11)))
            btn_h = min(42, max(32, int(avail_h * 0.06)))

        self.node_positions.clear()
        for node in self.nodes_data:
            addr = node["address"]
            nx = node.get("x", 0.5)
            ny = node.get("y", 0.5)

            px = int(pad_x + nx * avail_w)
            py = int(pad_y + ny * avail_h)
            self.node_positions[addr] = QPoint(px, py)

            btn = self.buttons.get(addr)
            if btn:
                btn.setFixedSize(btn_w, btn_h)
                btn.move(px - btn_w // 2, py - btn_h // 2)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Если это Star-8 или Sym-Star8: капсула вокруг центральных узлов
        if self.topo_type in ("star8", "sym_star8"):
            input_pts = [self.node_positions[addr] for addr in self.node_positions if addr[0] == 0]
            if input_pts:
                btn = next(iter(self.buttons.values()), None)
                bw = btn.width() if btn else 54
                bh = btn.height() if btn else 34
                min_x = min(p.x() for p in input_pts) - bw // 2 - 8
                max_x = max(p.x() for p in input_pts) + bw // 2 + 8
                min_y = min(p.y() for p in input_pts) - bh // 2 - 6
                max_y = max(p.y() for p in input_pts) + bh // 2 + 6
                capsule_rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

                painter.setPen(QPen(QColor("#546e7a"), 1.5))
                painter.setBrush(QBrush(QColor(33, 40, 52, 180)))
                painter.drawRoundedRect(capsule_rect, 10, 10)

        # 2. Отрисовка связей (edges)
        for edge in self.edges_data:
            p1 = self.node_positions.get(edge["from"])
            p2 = self.node_positions.get(edge["to"])
            if not p1 or not p2:
                continue

            # Пропуск тороидальных связей через весь экран, если они портят читаемость
            if abs(p1.x() - p2.x()) > self.width() * 0.7 or abs(p1.y() - p2.y()) > self.height() * 0.7:
                continue

            kind = edge.get("type", "primary")
            color = QColor(edge.get("color", "#444444"))

            if kind == "primary":
                pen = QPen(color, 2.2, Qt.SolidLine)
            elif kind == "secondary" or kind == "lateral":
                pen = QPen(color, 1.4, Qt.DashLine)
            elif kind == "tertiary" or kind == "fallback":
                pen = QPen(color, 1.0, Qt.DotLine)
            else:
                pen = QPen(color, 1.2, Qt.DashLine)

            painter.setPen(pen)
            painter.drawLine(p1, p2)

        # 3. Отрисовка активных траекторий заявок
        for req in self.ctrl.arch.all_requests:
            if req.status == RequestStatus.IN_PROGRESS and len(req.path) > 1:
                path_pts = [self.node_positions[pos] for pos in req.path if pos in self.node_positions]
                if len(path_pts) > 1:
                    glow_pen = QPen(QColor(req.color), 3.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                    painter.setPen(glow_pen)
                    for i in range(len(path_pts) - 1):
                        painter.drawLine(path_pts[i], path_pts[i + 1])

    def refresh_views(self) -> None:
        """Обновление состояний всех узлов из текущего снимка истории."""
        if not self.ctrl.arch.history:
            return

        snapshot = self.ctrl.arch.history[self.ctrl.arch.current_history_index]
        state_data = snapshot.neurons_state

        for addr, btn in self.buttons.items():
            if isinstance(state_data, dict):
                info = state_data.get(addr, {})
            else:
                r, c = addr
                info = state_data[r][c] if r < len(state_data) and c < len(state_data[r]) else {}
            btn.update_view(info)

        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Симулятор НС АЦП — Мульти-топологическая среда")
        self.resize(1260, 760)
        self.setMinimumSize(980, 580)

        self.q_required: int = 4
        self.total_requests: int = 6

        # Базовая архитектура по умолчанию: Тороидальная матрица 6x6
        self.arch: BaseArchitecture = ToroidalGridArchitecture(
            rows=6,
            cols=6,
            q_required=self.q_required,
            total_initial_requests=self.total_requests
        )
        self.ctrl = SimulationController(self.arch, speed_ms=400)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self.ctrl.subscribe("tick", self._on_controller_tick)
        self.ctrl.subscribe("state_change", self._on_controller_state_change)
        self.ctrl.subscribe("reset", self._on_controller_reset)

        self._init_ui()
        self._apply_styles()
        self.refresh_all()

    def _init_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # 1. Верхняя панель управления и переключатель топологий
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(6)

        # Свитчер топологий
        ctrl_bar.addWidget(QLabel("Топология:"))
        self.combo_topo = QComboBox()
        self.combo_topo.addItem("Тороидальная матрица 6x6", "torus")
        self.combo_topo.addItem("Ромбовидная архитектура", "rhombus")
        self.combo_topo.addItem("Восьмиконечная звезда (Star-8, 4 входа)", "star8_4in")
        self.combo_topo.addItem("Восьмиконечная звезда (Star-8, 6 входов)", "star8_6in")
        self.combo_topo.addItem("Гексагональная сота (Honeycomb, deg=3)", "honeycomb")
        self.combo_topo.addItem("Симметричная звезда C8 (Sym-Star8, deg=3)", "sym_star8")
        self.combo_topo.addItem("Расширенная воронка 6->8 (W=[6,8,8,8,8], deg=3)", "funnel8")
        self.combo_topo.addItem("Хордовая воронка 6->8 (deg=3, хорда +4)", "chordal_funnel8")
        self.combo_topo.currentIndexChanged.connect(self._on_topo_changed)
        ctrl_bar.addWidget(self.combo_topo)

        self.btn_prev = QPushButton("Назад")
        self.btn_prev.clicked.connect(self.ctrl.step_backward)
        ctrl_bar.addWidget(self.btn_prev)

        self.btn_play = QPushButton("Пуск")
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_bar.addWidget(self.btn_play)

        self.btn_next = QPushButton("Вперед")
        self.btn_next.clicked.connect(self.ctrl.step_forward)
        ctrl_bar.addWidget(self.btn_next)

        self.btn_end = QPushButton("В конец")
        self.btn_end.clicked.connect(self.ctrl.fast_forward)
        ctrl_bar.addWidget(self.btn_end)

        self.btn_reset = QPushButton("Сброс")
        self.btn_reset.clicked.connect(self.ctrl.reset)
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

        # 2. Подпанель параметров симуляции (q, N, формула)
        params_bar = QHBoxLayout()
        params_bar.setSpacing(8)

        params_bar.addWidget(QLabel("Разрядность q:"))
        self.spin_q = QSpinBox()
        self.spin_q.setRange(2, 6)
        self.spin_q.setValue(self.q_required)
        self.spin_q.valueChanged.connect(self._on_q_changed)
        params_bar.addWidget(self.spin_q)

        params_bar.addWidget(QLabel("Заявок N:"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 24)
        self.spin_n.setValue(self.total_requests)
        self.spin_n.valueChanged.connect(self._on_n_changed)
        params_bar.addWidget(self.spin_n)

        t_life = 2 * self.q_required + self.q_required + 2
        self.lbl_formula = QLabel(f"Время жизни: 2*q + q + 2 = {t_life} т.")
        self.lbl_formula.setStyleSheet("color: #4fc3f7; font-weight: bold;")
        params_bar.addWidget(self.lbl_formula)

        params_bar.addStretch(1)
        main_layout.addLayout(params_bar)

        # 3. Рабочая область: Графический канвас слева, статистика и лог справа
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # Графический канвас топологии
        self.canvas = TopologyCanvasWidget(ctrl=self.ctrl, parent=self)
        splitter.addWidget(self.canvas)

        # Правая панель: KPI и консоль
        side_widget = QWidget()
        side_layout = QVBoxLayout(side_widget)
        side_layout.setContentsMargins(8, 0, 0, 0)
        side_layout.setSpacing(6)

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

        # Легенда иерархии маршрутизации (1: Основной, 2: Мост, 3: Резерв)
        legend_layout = QHBoxLayout()
        legend_layout.setContentsMargins(0, 2, 0, 2)
        legend_layout.setSpacing(6)
        lbl_p = QLabel("― Приоритет 1 (Осн.)")
        lbl_p.setStyleSheet("color: #4fc3f7; font-weight: bold; font-size: 8pt;")
        lbl_s = QLabel("╌ Приоритет 2 (Мост)")
        lbl_s.setStyleSheet("color: #90a4ae; font-size: 8pt;")
        lbl_t = QLabel("┄ Приоритет 3 (Резерв)")
        lbl_t.setStyleSheet("color: #757575; font-size: 8pt;")
        legend_layout.addWidget(lbl_p)
        legend_layout.addWidget(lbl_s)
        legend_layout.addWidget(lbl_t)
        side_layout.addLayout(legend_layout)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 9))
        side_layout.addWidget(self.log_view, 1)

        splitter.addWidget(side_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter, 1)

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: #dcdcdc;
                font-family: 'Segoe UI', Arial, 'DejaVu Sans', sans-serif;
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
            QComboBox {
                background-color: #242424;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 3px 8px;
                color: #ffffff;
                font-weight: 500;
            }
            QComboBox QAbstractItemView {
                background-color: #242424;
                selection-background-color: #37474f;
                color: #ffffff;
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

    def _on_topo_changed(self, index: int) -> None:
        """Переключение топологии через свитчер."""
        key = self.combo_topo.currentData()
        self.ctrl.pause()

        if key == "torus":
            self.arch = ToroidalGridArchitecture(
                rows=6, cols=6,
                q_required=self.q_required,
                total_initial_requests=self.total_requests
            )
        elif key == "rhombus":
            self.arch = RhomboidArchitecture(
                widths=[4, 6, 8, 6, 4],
                q_required=self.q_required,
                total_initial_requests=self.total_requests
            )
        elif key in ("star8", "star8_4in"):
            self.arch = Star8Architecture(
                ray_depth=4,
                num_inputs=4,
                q_required=self.q_required,
                total_initial_requests=self.total_requests
            )
        elif key == "star8_6in":
            self.arch = Star8Architecture(
                ray_depth=4,
                num_inputs=6,
                q_required=self.q_required,
                total_initial_requests=self.total_requests
            )
        elif key == "honeycomb":
            self.arch = HoneycombPipelineArchitecture(
                rows=5, cols=6,
                q_required=self.q_required,
                total_initial_requests=self.total_requests
            )
        elif key == "sym_star8":
            self.arch = SymmetricStar8Architecture(
                ray_depth=4,
                num_inputs=6,
                q_required=self.q_required,
                total_initial_requests=self.total_requests
            )
        elif key == "funnel8":
            self.arch = Funnel8Architecture(
                q_required=self.q_required,
                total_initial_requests=self.total_requests
            )
        elif key == "chordal_funnel8":
            self.arch = ChordalFunnel8Architecture(
                q_required=self.q_required,
                total_initial_requests=self.total_requests
            )

        self.ctrl.set_architecture(self.arch)
        self.canvas.rebuild_canvas()
        self.refresh_all()

    def _on_timer_tick(self) -> None:
        if self.ctrl.state == PlayState.PLAYING:
            self.ctrl.step_forward()

    def toggle_play(self) -> None:
        self.ctrl.toggle_play()

    def _on_speed_changed(self, val: int) -> None:
        self.ctrl.set_speed(val)
        if self.timer.isActive():
            self.timer.setInterval(val)

    def _on_q_changed(self, val: int) -> None:
        self.q_required = val
        t_life = 2 * val + val + 2
        self.lbl_formula.setText(f"Время жизни: 2*q + q + 2 = {t_life} т.")
        self._on_topo_changed(self.combo_topo.currentIndex())

    def _on_n_changed(self, val: int) -> None:
        self.total_requests = val
        self._on_topo_changed(self.combo_topo.currentIndex())

    def _on_slider_moved(self, val: int) -> None:
        if val != self.arch.current_history_index:
            self.ctrl.go_to_tick(val)

    def _on_controller_tick(self, tick: int) -> None:
        self.refresh_all()

    def _on_controller_state_change(self, state: PlayState) -> None:
        if state == PlayState.PLAYING:
            self.btn_play.setText("Пауза")
            if not self.timer.isActive():
                self.timer.start(self.ctrl.speed_ms)
        elif state == PlayState.PAUSED:
            self.btn_play.setText("Пуск")
            if self.timer.isActive():
                self.timer.stop()
        elif state in (PlayState.STOPPED, PlayState.FINISHED):
            self.btn_play.setText("Пуск")
            if self.timer.isActive():
                self.timer.stop()
        self.refresh_all()

    def _on_controller_reset(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self.btn_play.setText("Пуск")
        self.refresh_all()

    def refresh_all(self) -> None:
        if not self.arch.history:
            return

        self.canvas.refresh_views()

        total_hist = len(self.arch.history) - 1
        curr_idx = self.arch.current_history_index

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, total_hist)
        self.timeline_slider.setValue(curr_idx)
        self.timeline_slider.blockSignals(False)

        self.lbl_tick.setText(f"Такт: {self.arch.current_tick} ({curr_idx}/{total_hist})")

        kpi = self.ctrl.get_kpi()
        self.kpi_total.setText(str(self.arch.total_initial_requests))
        self.kpi_active.setText(str(kpi["active"]))
        self.kpi_completed.setText(str(kpi["completed"]))
        self.kpi_rejected.setText(str(kpi["rejected"]))
        self.kpi_faults.setText(f"{kpi['faults_count']} / {kpi['total_neurons']}")
        self.kpi_fail_rate.setText(f"{kpi['fail_rate']:.1f} %")

        self._update_logs()

    def _update_logs(self) -> None:
        html_lines = []
        for entry in self.arch.logs:
            if entry["tick"] > self.arch.current_tick:
                continue

            etype = entry["type"]
            msg = entry["msg"]
            t = entry["tick"]

            if etype in ("FAULT", "REJECT"):
                color = "#ef5350"
            elif etype == "DONE":
                color = "#66bb6a"
            elif etype == "BUILT":
                color = "#ab47bc"
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
