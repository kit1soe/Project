"""
Модульные тесты для проверки НС АЦП (на базе unittest):
1. Абстрактный суперкласс BaseArchitecture.
2. Архитектура ToroidalGridArchitecture (тороид 6x6).
3. Архитектура RhomboidArchitecture (ромбовидная W=[4, 6, 8, 6, 4]).
4. Архитектура Star8Architecture (восьмиконечная звезда с 6 входами).
5. Формула времени жизни t = 2*q + q + 2.
6. Потактовое построение конвейера с уменьшением q_remaining.
7. Алгоритм возврата в исходный столбец (канал).
8. Машина времени (Memento) и ветвление сценариев.
9. Контроллер симуляции SimulationController и динамическое переключение архитектур.
10. Корректность инициализации и смены топологий в GUI (в headless-режиме).
"""

import os
import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from models import (
    BaseArchitecture, ToroidalGridArchitecture, RhomboidArchitecture,
    PyramidalArchitecture,
    Star8Architecture, HoneycombPipelineArchitecture, SymmetricStar8Architecture,
    Funnel8Architecture, ChordalFunnel8Architecture,
    SimulationEngine,
    NeuronState, RequestStatus, PlayState, Request, Neuron
)
from controller import SimulationController
from analytics_db import AnalyticsRecorder


class TestSimulation(unittest.TestCase):

    def test_base_architecture_is_abstract(self):
        """Проверка, что BaseArchitecture абстрактен и не может быть создан напрямую."""
        with self.assertRaises(TypeError):
            BaseArchitecture()

    def test_lifetime_formula(self):
        """Проверка формулы времени жизни заявки: t = 2*q + q + 2 (= 3*q + 2)."""
        for q in [2, 3, 4, 5, 6]:
            req = Request(request_id=1, q_required=q)
            expected_time = 2 * q + q + 2
            self.assertEqual(req.service_time, expected_time)
            self.assertEqual(req.remaining_service_time, expected_time)
            self.assertEqual(req.q_remaining, q)

    def test_pipeline_decrements_q(self):
        """
        Проверка, что по мере продвижения сквозь слои q_remaining уменьшается на 1 за такт,
        а цепочка ОИН успешно формируется.
        """
        arch = ToroidalGridArchitecture(rows=6, cols=4, q_required=4, total_initial_requests=1)
        ctrl = SimulationController(arch)

        # Такт 1: ввод Заявки #1 на [0, 0]
        ctrl.step_forward()
        self.assertEqual(len(arch.all_requests), 1)
        req = arch.all_requests[0]
        self.assertEqual(req.path, [(0, 0)])
        self.assertEqual(req.q_remaining, 3)  # Было 4, 1 занят на входе, 3 осталось
        self.assertEqual(req.status, RequestStatus.IN_PROGRESS)

        # Такт 2: переход на [1, 0]
        ctrl.step_forward()
        self.assertEqual(req.path, [(0, 0), (1, 0)])
        self.assertEqual(req.q_remaining, 2)

        # Такт 3: переход на [2, 0]
        ctrl.step_forward()
        self.assertEqual(req.path, [(0, 0), (1, 0), (2, 0)])
        self.assertEqual(req.q_remaining, 1)

        # Такт 4: переход на [3, 0] -> цепочка полностью построена!
        ctrl.step_forward()
        self.assertEqual(req.path, [(0, 0), (1, 0), (2, 0), (3, 0)])
        self.assertEqual(req.q_remaining, 0)
        self.assertTrue(req.is_pipeline_built())

    def test_return_to_origin_column(self):
        """
        Проверка, что при сбое нейрона заявка сдвигается в соседний столбец,
        а на следующем такте возвращается обратно в свой исходный столбец.
        """
        arch = ToroidalGridArchitecture(rows=6, cols=4, q_required=4, total_initial_requests=1)
        # Создаем сбой в [1, 0] (прямо на пути Заявки #1, входящей в [0, 0])
        arch.toggle_neuron_fault(1, 0)
        ctrl = SimulationController(arch)

        # Такт 1: Заявка #1 вошла в [0, 0]
        ctrl.step_forward()
        req = arch.all_requests[0]
        self.assertEqual(req.path, [(0, 0)])

        # Такт 2: [1, 0] в сбое! Заявка вынуждена повернуть.
        # Для [0, 0] соседи: центр=[1, 0](сбой), лево=[1, 3], право=[1, 1].
        # Приоритет в своем столбце: центр -> лево -> право. Значит идет в лево [1, 3]!
        ctrl.step_forward()
        self.assertEqual(req.path[-1], (1, 3))
        self.assertEqual(req.origin_col, 0)

        # Такт 3: Заявка находится в [1, 3] (левее исходного столбца 0).
        # До столбца 0: вправо 1 шаг, влево 3 шага.
        # Приоритет возврата: вправо ([2, 0]) -> центр ([2, 3]) -> лево ([2, 2]).
        # Должна успешно вернуться в свой исходный столбец 0: [2, 0]!
        ctrl.step_forward()
        self.assertEqual(req.path[-1], (2, 0))

        # Такт 4: Снова в своем столбце 0, идет прямо вниз в [3, 0]
        ctrl.step_forward()
        self.assertEqual(req.path[-1], (3, 0))
        self.assertTrue(req.is_pipeline_built())

    def test_torus_6x6_default(self):
        """Проверка дефолтной матрицы 6x6 (36 нейронов, 6 входов)."""
        torus = ToroidalGridArchitecture()
        self.assertEqual(torus.rows, 6)
        self.assertEqual(torus.cols, 6)
        self.assertEqual(len(torus.all_neurons()), 36)
        self.assertEqual(torus.input_columns_count, 6)
        info = torus.get_topology_info()
        self.assertEqual(len(info["nodes"]), 36)
        self.assertEqual(info["type"], "toroidal_grid")

    def test_torus_wrap_only_horizontal(self):
        """
        Проверка, что ToroidalGridArchitecture закольцована ТОЛЬКО по горизонтали,
        а по вертикали закольцовывание отсутствует (с последнего слоя нет переходов на нулевой).
        """
        torus = ToroidalGridArchitecture(rows=6, cols=6)
        req = Request(request_id=1, q_required=4, input_neuron_index=(0, 0))

        # Горизонтальное закольцовывание работает: из (0, 0) сосед слева - (1, 5)
        n_top_left = torus.get_neuron((0, 0))
        cands_top = torus.get_candidate_neighbors(n_top_left, req)
        cand_addresses = [n.address for n, _ in cands_top]
        self.assertIn((1, 0), cand_addresses)
        self.assertIn((1, 5), cand_addresses)  # (0 - 1) % 6 == 5
        self.assertIn((1, 1), cand_addresses)

        # Вертикальное закольцовывание отсутствует: из последнего слоя (5, c) нет переходов на слой 0
        n_bottom = torus.get_neuron((5, 0))
        cands_bottom = torus.get_candidate_neighbors(n_bottom, req)
        self.assertEqual(cands_bottom, [], "С последнего слоя не должно быть переходов на слой 0")

        # Проверка ребер топологии: ни одно ребро не ведет с 5-го слоя на 0-й
        info = torus.get_topology_info()
        for edge in info["edges"]:
            self.assertNotEqual(
                (edge["from"][0], edge["to"][0]),
                (5, 0),
                "Ребро не должно закольцовываться с нижнего слоя на верхний"
            )

    def test_mvc_controller_separation(self):
        """
        Проверка, что SimulationController находится в отдельном модуле controller.py (MVC),
        а также доступен через models.py для обратной совместимости.
        """
        import controller
        import models
        self.assertTrue(hasattr(controller, "SimulationController"))
        self.assertTrue(hasattr(models, "SimulationController"))
        self.assertIs(controller.SimulationController, models.SimulationController)

    def test_rhomboid_architecture(self):
        """
        Проверка ромбовидной архитектуры (RhomboidArchitecture):
        1. Структура слоев W=[4, 6, 8, 6, 4] (28 нейронов).
        2. Потактовое построение конвейера с уменьшением q_remaining.
        3. Обход сбойного нейрона.
        4. Завершение работы алгоритма (fast forward).
        """
        arch = RhomboidArchitecture(widths=[4, 6, 8, 6, 4], q_required=3, total_initial_requests=2)
        self.assertEqual(len(arch.all_neurons()), 4 + 6 + 8 + 6 + 4)
        info = arch.get_topology_info()
        self.assertEqual(info["type"], "rhomboid")
        self.assertEqual(len(info["nodes"]), 28)

        # Искусственный сбой во 2-м слое
        arch.toggle_neuron_fault(1, 1)
        self.assertFalse(arch.get_neuron((1, 1)).is_operational)

        ctrl = SimulationController(arch)
        ctrl.step_forward()  # такт 1
        self.assertEqual(len(arch.all_requests), 1)
        req = arch.all_requests[0]
        self.assertEqual(req.path, [(0, 0)])

        # Проходим несколько шагов
        for _ in range(5):
            ctrl.step_forward()

        # Завершаем до конца
        ctrl.fast_forward()
        self.assertTrue(arch.is_finished())
        self.assertGreater(arch.completed_count, 0)

    def test_star8_architecture(self):
        """
        Проверка радиальной восьмиконечной архитектуры (Star8Architecture):
        1. 4 входных узла и глубина 4: 4 + 8*4 + 4 = 40 узлов.
        2. 6 входных узлов и глубина 4: 6 + 8*4 + 4 = 42 узла.
        3. Иерархия степеней: deg+(V0)=4..5 -> deg+(V1)=4 -> deg+(V2)=3 -> deg+(V3)=1 -> deg+(V4)=0.
        4. Обход локального сбоя луча через промежуточные мосты.
        5. Корректное завершение обработки.
        """
        # Конфигурация с 4 входами и глубиной 4 (40 узлов)
        arch4 = Star8Architecture(ray_depth=4, num_inputs=4, q_required=4, total_initial_requests=4)
        self.assertEqual(len(arch4.all_neurons()), 40)
        info4 = arch4.get_topology_info()
        self.assertEqual(info4["type"], "star8")
        self.assertEqual(len(info4["nodes"]), 40)
        self.assertEqual(info4["num_inputs"], 4)
        self.assertEqual(info4["ray_depth"], 4)

        # Конфигурация с 6 входами и глубиной 4 (42 узла)
        arch6 = Star8Architecture(ray_depth=4, num_inputs=6, q_required=4, total_initial_requests=6)
        self.assertEqual(len(arch6.all_neurons()), 42)
        info6 = arch6.get_topology_info()
        self.assertEqual(info6["type"], "star8")
        self.assertEqual(len(info6["nodes"]), 42)
        self.assertEqual(info6["num_inputs"], 6)

        # Проверяем, что узлы входов [0, 0]..[0, 5] присутствуют
        for c in range(6):
            n = arch6.get_neuron((0, c))
            self.assertIsNotNone(n)

        # Проверяем иерархию степеней (outgoing candidate count):
        req_dummy = Request(request_id=99, q_required=4, input_neuron_index=(0, 0))
        # Слой 3: строго 1 связь прямо в слой 4
        cands_v3 = arch6.get_candidate_neighbors(arch6.get_neuron((3, 0)), req_dummy)
        self.assertEqual(len(cands_v3), 1)
        self.assertEqual((cands_v3[0][0].row, cands_v3[0][0].col), (4, 0))

        # Слой 4 (терминальный): 0 связей
        cands_v4 = arch6.get_candidate_neighbors(arch6.get_neuron((4, 0)), req_dummy)
        self.assertEqual(len(cands_v4), 0)

        # Искусственный сбой на луче 0 первого кольца
        arch6.toggle_neuron_fault(1, 0)
        self.assertFalse(arch6.get_neuron((1, 0)).is_operational)

        ctrl = SimulationController(arch6)
        ctrl.step_forward()  # такт 1: первая заявка заходит в центр [0, 0]
        req1 = arch6.all_requests[0]
        self.assertEqual(req1.path[0], (0, 0))

        # Запускаем fast-forward и проверяем отказоустойчивость: заявки должны успешно завершиться
        ctrl.fast_forward()
        self.assertTrue(arch6.is_finished())
        self.assertGreater(arch6.completed_count, 0)

    def test_honeycomb_architecture(self):
        """
        Проверка гексагональной сотовой архитектуры (HoneycombPipelineArchitecture, deg=3):
        1. Размерность: 5 слоев x 6 колонок = 30 нейронов.
        2. Степень связности: строго deg = 3 для промежуточных узлов (вниз, влево-вниз, вправо-вниз).
        3. Обход сбоя: при выходе из строя узла [1, 0] заявка успешно обходит его по диагонали.
        4. Завершение симуляции без тупиков.
        """
        honey = HoneycombPipelineArchitecture(rows=5, cols=6, q_required=4, total_initial_requests=6)
        self.assertEqual(len(honey.all_neurons()), 30)
        info = honey.get_topology_info()
        self.assertEqual(info["type"], "honeycomb")
        self.assertEqual(len(info["nodes"]), 30)

        # Проверка степени связности deg=3 для внутренних узлов
        dummy = Request(request_id=1, q_required=4, input_neuron_index=(0, 0))
        cands_r1 = honey.get_candidate_neighbors(honey.get_neuron((1, 2)), dummy)
        self.assertEqual(len(cands_r1), 3)

        # Терминальный слой (слой 4) не имеет дальнейших переходов
        cands_r4 = honey.get_candidate_neighbors(honey.get_neuron((4, 2)), dummy)
        self.assertEqual(len(cands_r4), 0)

        # Тест обхода сбоя
        honey.toggle_neuron_fault(1, 0)
        self.assertFalse(honey.get_neuron((1, 0)).is_operational)

        ctrl = SimulationController(honey)
        ctrl.fast_forward()
        self.assertTrue(honey.is_finished())
        self.assertGreater(honey.completed_count, 0)

    def test_symmetric_star8_architecture(self):
        """
        Проверка симметричной радиальной архитектуры (SymmetricStar8Architecture, deg=3):
        1. Узлы: 6 входных + 8 лучей * 4 слоя = 38 узлов.
        2. Степень связности: строго deg = 3 для промежуточных узлов (радиально + 2 хорды кольца C8).
        3. Обход сбоя: при выходе из строя узла [2, 0] заявка обходит узел через хордовое кольцо.
        4. Завершение симуляции.
        """
        sym = SymmetricStar8Architecture(ray_depth=4, num_inputs=6, q_required=4, total_initial_requests=6)
        self.assertEqual(len(sym.all_neurons()), 38)
        info = sym.get_topology_info()
        self.assertEqual(info["type"], "sym_star8")
        self.assertEqual(len(info["nodes"]), 38)

        # Проверка степени связности deg=3 для узла на луче
        dummy = Request(request_id=1, q_required=4, input_neuron_index=(0, 0))
        cands = sym.get_candidate_neighbors(sym.get_neuron((2, 0)), dummy)
        self.assertEqual(len(cands), 3)

        # Искусственный сбой
        sym.toggle_neuron_fault(2, 0)
        self.assertFalse(sym.get_neuron((2, 0)).is_operational)

        ctrl = SimulationController(sym)
        ctrl.fast_forward()
        self.assertTrue(sym.is_finished())
        self.assertGreater(sym.completed_count, 0)

    def test_controller_architecture_switching(self):
        """
        Проверка динамической смены архитектуры в SimulationController:
        - Переключение между ToroidalGrid, Rhomboid и Star8.
        - Сброс тактов и состояния.
        """
        torus = ToroidalGridArchitecture(rows=6, cols=4, q_required=3, total_initial_requests=2)
        ctrl = SimulationController(torus)

        ctrl.step_forward()
        ctrl.step_forward()
        self.assertEqual(ctrl.current_tick, 2)

        # Переключаемся на Ромб
        rhombus = RhomboidArchitecture(widths=[4, 6, 8, 6, 4], q_required=3, total_initial_requests=2)
        ctrl.set_architecture(rhombus)
        self.assertIs(ctrl.architecture, rhombus)
        self.assertEqual(ctrl.current_tick, 0)
        self.assertEqual(ctrl.state, PlayState.STOPPED)

        ctrl.step_forward()
        self.assertEqual(ctrl.current_tick, 1)

        # Переключаемся на Звезду
        star8 = Star8Architecture(ray_depth=3, num_inputs=6, q_required=3, total_initial_requests=2)
        ctrl.set_architecture(star8)
        self.assertIs(ctrl.architecture, star8)
        self.assertEqual(ctrl.current_tick, 0)

        ctrl.step_forward()
        self.assertEqual(ctrl.current_tick, 1)

    def test_memento_time_travel(self):
        """Проверка машины времени: шаг назад, вперед, восстановление тактов."""
        arch = ToroidalGridArchitecture(rows=6, cols=4, q_required=4, total_initial_requests=4)
        ctrl = SimulationController(arch)

        for _ in range(10):
            ctrl.step_forward()

        tick_10 = arch.current_tick
        self.assertEqual(tick_10, 10)

        # Шаг назад
        ctrl.step_backward()
        self.assertEqual(arch.current_tick, 9)

        # Шаг назад
        ctrl.step_backward()
        self.assertEqual(arch.current_tick, 8)

        # Шаг вперед
        ctrl.step_forward()
        self.assertEqual(arch.current_tick, 9)

    def test_simulation_controller(self):
        """Проверка методов и событий контроллера."""
        arch = ToroidalGridArchitecture(rows=6, cols=4, q_required=4, total_initial_requests=2)
        ctrl = SimulationController(arch, speed_ms=200)

        events_received = []
        ctrl.subscribe("state_change", lambda s: events_received.append(("state", s)))
        ctrl.subscribe("tick", lambda t: events_received.append(("tick", t)))

        ctrl.play()
        self.assertEqual(ctrl.state, PlayState.PLAYING)

        ctrl.pause()
        self.assertEqual(ctrl.state, PlayState.PAUSED)

        ctrl.step_forward()
        self.assertTrue(any(e[0] == "tick" for e in events_received))

        ctrl.fast_forward()
        self.assertEqual(ctrl.state, PlayState.FINISHED)
        self.assertTrue(arch.is_finished())

        ctrl.reset()
        self.assertEqual(ctrl.state, PlayState.STOPPED)
        self.assertEqual(arch.current_tick, 0)

    def test_funnel8_architecture_fault_tolerance(self):
        """
        Проверка Funnel8Architecture: 6 заявок при сбое на слое 1.
        Благодаря резервным узлам (3 и 7) все 6 заявок успешно завершают обслуживание (0 отказов).
        """
        arch = Funnel8Architecture(q_required=4, total_initial_requests=6)
        arch.toggle_neuron_fault(1, 0)
        arch.fast_forward_to_end()
        
        self.assertEqual(arch.rejected_count, 0)
        self.assertEqual(arch.completed_count, 6)
        self.assertEqual(len(arch.all_requests), 6)

    def test_chordal_funnel8_architecture_fault_tolerance(self):
        """
        Проверка ChordalFunnel8Architecture: 6 заявок при сбое нейрона [1, 0].
        Хордовые и смежные связи обеспечивают 100% прохождение всех 6 заявок.
        """
        arch = ChordalFunnel8Architecture(q_required=4, total_initial_requests=6)
        arch.toggle_neuron_fault(1, 0)
        arch.fast_forward_to_end()

        self.assertEqual(arch.rejected_count, 0)
        self.assertEqual(arch.completed_count, 6)
        self.assertEqual(len(arch.all_requests), 6)

    def test_backwards_compatibility_simulation_engine(self):
        """Проверка, что класс SimulationEngine доступен и работает как наследник."""
        engine = SimulationEngine(rows=6, cols=4, q_required=4, total_initial_requests=4)
        self.assertIsInstance(engine, BaseArchitecture)
        engine.step_forward()
        self.assertEqual(engine.current_tick, 1)

    def test_gui_headless_switching(self):
        """
        Проверка GUI в headless-режиме:
        - Инициализация главного окна.
        - Переключение выпадающего списка QComboBox между всеми 4 топологиями:
          тороид, ромб, Star-8 (4 входа), Star-8 (6 входов).
        - Выполнение шагов симуляции и перерисовка канваса без ошибок.
        """
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtWidgets import QApplication
        from gui import MainWindow

        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

        win = MainWindow()
        self.assertIsNotNone(win)
        self.assertEqual(win.combo_topo.count(), 9)

        # 1. Переключение на Пирамиду
        win.combo_topo.setCurrentIndex(1)
        self.assertEqual(win.ctrl.architecture.get_topology_info()["type"], "pyramidal")
        win.ctrl.step_forward()

        # 2. Переключение на Ромб
        win.combo_topo.setCurrentIndex(2)
        self.assertEqual(win.ctrl.architecture.get_topology_info()["type"], "rhomboid")
        win.ctrl.step_forward()

        # 3. Переключение на Star-8 (4 входа)
        win.combo_topo.setCurrentIndex(3)
        info4 = win.ctrl.architecture.get_topology_info()
        self.assertEqual(info4["type"], "star8")
        self.assertEqual(info4["num_inputs"], 4)
        self.assertEqual(info4["ray_depth"], 4)
        win.ctrl.step_forward()

        # 4. Переключение на Star-8 (6 входов)
        win.combo_topo.setCurrentIndex(4)
        info6 = win.ctrl.architecture.get_topology_info()
        self.assertEqual(info6["type"], "star8")
        self.assertEqual(info6["num_inputs"], 6)
        self.assertEqual(info6["ray_depth"], 4)
        win.ctrl.step_forward()

        # 5. Переключение на Соту (Honeycomb)
        win.combo_topo.setCurrentIndex(5)
        info_honey = win.ctrl.architecture.get_topology_info()
        self.assertEqual(info_honey["type"], "honeycomb")
        win.ctrl.step_forward()

        # 6. Переключение на Симметричную звезду C8 (Sym-Star8)
        win.combo_topo.setCurrentIndex(6)
        info_sym = win.ctrl.architecture.get_topology_info()
        self.assertEqual(info_sym["type"], "sym_star8")
        win.ctrl.step_forward()

        # 7. Переключение на Funnel-8
        win.combo_topo.setCurrentIndex(7)
        info_f8 = win.ctrl.architecture.get_topology_info()
        self.assertEqual(info_f8["type"], "funnel8")
        win.ctrl.step_forward()

        # 8. Переключение на Chordal-Funnel-8
        win.combo_topo.setCurrentIndex(8)
        info_cf8 = win.ctrl.architecture.get_topology_info()
        self.assertEqual(info_cf8["type"], "chordal_funnel8")
        win.ctrl.step_forward()

        # 9. Возврат на Тороид 6x6
        win.combo_topo.setCurrentIndex(0)
        self.assertEqual(win.combo_topo.itemText(0), "Тороидальная матрица 6x6")
        self.assertEqual(win.ctrl.architecture.get_topology_info()["type"], "toroidal_grid")
        self.assertEqual(win.ctrl.architecture.cols, 6)
        win.ctrl.step_forward()

    def test_analytics_recorder_persists_detailed_state(self):
        """SQLite stores run metadata, snapshots, events and every neuron state."""
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "analytics.sqlite"
            arch = ToroidalGridArchitecture(rows=3, cols=3, q_required=2, total_initial_requests=2)
            ctrl = SimulationController(arch)
            with AnalyticsRecorder(db_path) as recorder:
                run_id = recorder.start_run(arch)
                ctrl.step_forward()
                recorder.record_snapshot(arch)
                ctrl.toggle_fault(1, 0)
                recorder.record_snapshot(arch)

                summary = recorder.get_run_summary(run_id)
                self.assertEqual(summary["run_id"], run_id)
                self.assertGreaterEqual(summary["ticks"], 2)
                self.assertEqual(summary["neurons"], 9)
                self.assertGreaterEqual(summary["events"], 1)
                self.assertTrue(recorder.get_event_counts(run_id))
                self.assertEqual(len(recorder.get_neuron_heatmap(run_id)), 9)
                exported = recorder.export_csv(Path(tmp) / "csv", run_id)
                self.assertEqual(len(exported), 5)

            db = sqlite3.connect(db_path)
            try:
                self.assertGreater(db.execute("SELECT COUNT(*) FROM neuron_states").fetchone()[0], 0)
                self.assertGreater(db.execute("SELECT COUNT(*) FROM topology_edges").fetchone()[0], 0)
                self.assertGreater(db.execute("SELECT COUNT(*) FROM actions").fetchone()[0], 0)
            finally:
                db.close()

    def test_analytics_branch_keeps_history_indices(self):
        """Rewinding and changing a run leaves branch snapshots queryable."""
        with TemporaryDirectory() as tmp:
            arch = ToroidalGridArchitecture(rows=3, cols=3, q_required=2, total_initial_requests=3)
            ctrl = SimulationController(arch)
            with AnalyticsRecorder(Path(tmp) / "analytics.sqlite") as recorder:
                run_id = recorder.start_run(arch)
                for _ in range(4):
                    ctrl.step_forward()
                    recorder.record_snapshot(arch)
                ctrl.go_to_tick(2)
                recorder.record_snapshot(arch)
                ctrl.toggle_fault(1, 0)
                recorder.record_snapshot(arch)
                indices = [row["history_index"] for row in recorder.get_tick_metrics(run_id)]
                self.assertIn(2, indices)
                self.assertGreaterEqual(len(indices), 4)


    def test_neuron_invulnerability_during_request_processing(self):
        """
        Проверка принципа неуязвимости ОИН:
        Нейрон в процессе обработки заявки (BUSY) не может выйти из строя немедленно.
        Сбой переходит в отложенный режим (pending_fault) и вступает в силу строго
        после того, как заявка завершит обслуживание или уйдёт из системы по отказу.
        """
        arch = ToroidalGridArchitecture(rows=6, cols=6, q_required=4, total_initial_requests=1)
        ctrl = SimulationController(arch)

        # Такт 1: Заявка #1 входит в узел (0, 0)
        ctrl.step_forward()
        n00 = arch.get_neuron((0, 0))
        self.assertEqual(n00.state, NeuronState.BUSY)
        self.assertIsNotNone(n00.current_request)
        req = arch.all_requests[0]
        self.assertEqual(req.status, RequestStatus.IN_PROGRESS)

        # Инжектируем сбой в занятый узел (0, 0)
        is_active = arch.toggle_neuron_fault(0, 0)
        self.assertTrue(is_active)
        # Узел НЕ сбрасывает заявку и НЕ становится FAULTY немедленно!
        self.assertEqual(n00.state, NeuronState.BUSY)
        self.assertTrue(n00.pending_fault)
        self.assertTrue(n00.is_operational)
        self.assertEqual(req.status, RequestStatus.IN_PROGRESS)
        self.assertEqual(arch.rejected_count, 0)

        # Завершаем симуляцию
        ctrl.fast_forward()
        self.assertEqual(req.status, RequestStatus.COMPLETED)
        self.assertEqual(arch.completed_count, 1)
        self.assertEqual(arch.rejected_count, 0)

        # После завершения обслуживания заявки узел (0, 0) обязан перейти в FAULTY
        self.assertEqual(n00.state, NeuronState.FAULTY)
        self.assertFalse(n00.is_operational)
        self.assertFalse(n00.pending_fault)
        self.assertFalse(n00.can_accept())

    def test_neuron_invulnerability_cancellation(self):
        """Проверка отмены отложенного сбоя при повторном клике по занятому узлу."""
        arch = ToroidalGridArchitecture(rows=6, cols=6, q_required=4, total_initial_requests=1)
        ctrl = SimulationController(arch)

        ctrl.step_forward()
        n00 = arch.get_neuron((0, 0))
        self.assertEqual(n00.state, NeuronState.BUSY)

        # Клик 1: планирование сбоя
        arch.toggle_neuron_fault(0, 0)
        self.assertTrue(n00.pending_fault)

        # Клик 2: отмена отложенного сбоя
        arch.toggle_neuron_fault(0, 0)
        self.assertFalse(n00.pending_fault)
        self.assertEqual(n00.state, NeuronState.BUSY)

        ctrl.fast_forward()
        # По завершении узел свободен и исправен
        self.assertEqual(n00.state, NeuronState.FREE)
        self.assertTrue(n00.is_operational)

    def test_neuron_invulnerability_on_rejected_request(self):
        """Проверка перехода в сбой после ухода заявки из системы по отказу."""
        # Создаем узкую топологию, где неизбежен тупик на слое 1
        arch = ToroidalGridArchitecture(rows=6, cols=6, q_required=4, total_initial_requests=1)
        # Блокируем всех потенциальных кандидатов на слое 1
        for c in range(6):
            arch.toggle_neuron_fault(1, c)

        ctrl = SimulationController(arch)
        ctrl.step_forward()  # Заявка #1 входит в (0, 0)
        n00 = arch.get_neuron((0, 0))
        self.assertEqual(n00.state, NeuronState.BUSY)

        # Планируем сбой на входном узле (0, 0)
        arch.toggle_neuron_fault(0, 0)
        self.assertTrue(n00.pending_fault)

        # Следующий шаг: тупик на слое 1, заявка сбрасывается
        ctrl.step_forward()
        req = arch.all_requests[0]
        self.assertEqual(req.status, RequestStatus.REJECTED)

        # При сбросе заявки узел (0, 0) освобождается и переходит в FAULTY
        self.assertEqual(n00.state, NeuronState.FAULTY)
        self.assertFalse(n00.is_operational)
        self.assertFalse(n00.pending_fault)

    def test_random_fault_preserves_busy_request_and_records_event(self):
        """Случайный сбой занятого ОИН откладывается и попадает в текущий снимок."""
        arch = ToroidalGridArchitecture(rows=6, cols=6, q_required=4, total_initial_requests=1)
        ctrl = SimulationController(arch)
        ctrl.step_forward()

        changed = arch.randomize_faults(count=1, probability=1.0, seed=0)
        n00 = arch.get_neuron((0, 0))
        self.assertIn((0, 0), changed)
        self.assertEqual(n00.state, NeuronState.BUSY)
        self.assertTrue(n00.pending_fault)
        self.assertTrue(any(log["type"] == "FAULT" for log in arch.history[-1].new_logs))

        ctrl.fast_forward()
        self.assertEqual(arch.completed_count, 1)
        self.assertEqual(n00.state, NeuronState.FAULTY)
        self.assertFalse(n00.is_operational)

    def test_random_fault_truncates_future_history(self):
        """Случайный отказ после перемотки создаёт новую ветку истории."""
        arch = ToroidalGridArchitecture(rows=6, cols=6, q_required=4, total_initial_requests=2)
        arch.step_forward()
        arch.step_forward()
        arch.step_backward()
        self.assertEqual(len(arch.history), 3)

        arch.randomize_faults(count=1, probability=1.0, seed=0)
        self.assertEqual(len(arch.history), 2)
        self.assertEqual(arch.current_history_index, 1)

    def test_random_fault_zero_probability_is_noop(self):
        """Нулевая вероятность не меняет состояние и не обрезает историю."""
        arch = ToroidalGridArchitecture(rows=6, cols=6, q_required=4, total_initial_requests=2)
        arch.step_forward()
        arch.step_forward()
        arch.step_backward()
        before_len = len(arch.history)

        self.assertEqual(arch.randomize_faults(count=1, probability=0.0, seed=0), [])
        self.assertEqual(len(arch.history), before_len)
        self.assertEqual(arch.current_history_index, 1)

    def test_generation_ticks_custom(self):
        """Проверка генерации заявок строго на заданных тактах симуляции (замечание руководителя Югова А.А.)."""
        arch = ToroidalGridArchitecture(rows=6, cols=4, q_required=3, total_initial_requests=3, generation_ticks=[2, 5, 8])
        ctrl = SimulationController(arch)

        # Такт 1: не в generation_ticks -> заявок нет
        ctrl.step_forward()
        self.assertEqual(len(arch.all_requests), 0)

        # Такт 2: в generation_ticks -> Заявка #1 сгенерирована
        ctrl.step_forward()
        self.assertEqual(len(arch.all_requests), 1)
        self.assertEqual(arch.all_requests[0].id, 1)

        # Такт 3, 4: нет новых заявок
        ctrl.step_forward()
        self.assertEqual(len(arch.all_requests), 1)
        ctrl.step_forward()
        self.assertEqual(len(arch.all_requests), 1)

        # Такт 5: Заявка #2 сгенерирована
        ctrl.step_forward()
        self.assertEqual(len(arch.all_requests), 2)
        self.assertEqual(arch.all_requests[1].id, 2)

    def test_routing_mode_downward_without_return_torus(self):
        """
        Проверка режима переключения алгоритма: 'идти вниз при смещении' (без возврата в исходный столбец).
        При сбое нейрона заявка смещается в соседний столбец и продолжает двигаться вертикально вниз по новому столбцу.
        """
        # С возвратом (return_to_origin = True)
        arch_ret = ToroidalGridArchitecture(rows=6, cols=4, q_required=4, total_initial_requests=1, return_to_origin=True)
        arch_ret.toggle_neuron_fault(1, 0)
        ctrl_ret = SimulationController(arch_ret)
        ctrl_ret.step_forward()  # (0, 0)
        ctrl_ret.step_forward()  # Сбой (1, 0) -> уходит в (1, 3)
        self.assertEqual(arch_ret.all_requests[0].path[-1], (1, 3))
        ctrl_ret.step_forward()  # Режим с возвратом -> возвращается в (2, 0)
        self.assertEqual(arch_ret.all_requests[0].path[-1], (2, 0))

        # Без возврата (return_to_origin = False)
        arch_no_ret = ToroidalGridArchitecture(rows=6, cols=4, q_required=4, total_initial_requests=1, return_to_origin=False)
        arch_no_ret.toggle_neuron_fault(1, 0)
        ctrl_no_ret = SimulationController(arch_no_ret)
        ctrl_no_ret.step_forward()  # (0, 0)
        ctrl_no_ret.step_forward()  # Сбой (1, 0) -> уходит в (1, 3)
        self.assertEqual(arch_no_ret.all_requests[0].path[-1], (1, 3))
        ctrl_no_ret.step_forward()  # Режим БЕЗ возврата -> идет прямо вниз в новом столбце (2, 3)!
        self.assertEqual(arch_no_ret.all_requests[0].path[-1], (2, 3))

    def test_pyramidal_architecture_and_modes(self):
        """
        Проверка класса пирамидальной архитектуры (W=[4, 5, 6, 7, 8]):
        - Проверка топологии и количества узлов: 4+5+6+7+8 = 30 узлов.
        - Проверка построения конвейера заявок.
        - Проверка переключения режима маршрутизации (возврат vs прямой спуск).
        """
        pyramid = PyramidalArchitecture(q_required=4, total_initial_requests=1, return_to_origin=True)
        self.assertEqual(len(pyramid.all_neurons()), 30)
        info = pyramid.get_topology_info()
        self.assertEqual(info["type"], "pyramidal")
        self.assertEqual(len(info["nodes"]), 30)

        # Конвейер заявки в нормальном режиме
        ctrl = SimulationController(pyramid)
        ctrl.step_forward()  # Слой 0: (0, 0)
        req = pyramid.all_requests[0]
        self.assertEqual(req.path[-1], (0, 0))
        ctrl.step_forward()  # Слой 1: (1, 0)
        self.assertEqual(req.path[-1], (1, 0))
        ctrl.step_forward()  # Слой 2: (2, 0)
        self.assertEqual(req.path[-1], (2, 0))
        ctrl.step_forward()  # Слой 3: (3, 0)
        self.assertEqual(req.path[-1], (3, 0))
        self.assertTrue(req.is_pipeline_built())

        # Режим без возврата в пирамиде
        pyramid_no_ret = PyramidalArchitecture(q_required=4, total_initial_requests=1, return_to_origin=False)
        pyramid_no_ret.toggle_neuron_fault(1, 0)
        ctrl2 = SimulationController(pyramid_no_ret)
        ctrl2.step_forward()  # (0, 0)
        ctrl2.step_forward()  # (1, 0) сбой -> смещение вправо (1, 1) или влево (1, 4)
        c_layer1 = pyramid_no_ret.all_requests[0].path[-1][1]
        self.assertNotEqual(c_layer1, 0)
        ctrl2.step_forward()  # Слой 2: идет прямо вниз по проекции текущего смещения
        c_layer2 = pyramid_no_ret.all_requests[0].path[-1][1]
        expected_center = int(round(c_layer1 * (6 - 1) / max(1, 5 - 1)))
        self.assertEqual(c_layer2, expected_center)

    def test_rhomboid_architecture_modes(self):
        """Проверка переключения режимов маршрутизации в ромбовидной архитектуре."""
        rhombus_ret = RhomboidArchitecture(q_required=4, total_initial_requests=1, return_to_origin=True)
        self.assertTrue(rhombus_ret.return_to_origin)
        rhombus_ret.set_return_to_origin(False)
        self.assertFalse(rhombus_ret.return_to_origin)

        # Прогон симуляции ромба в режиме без возврата
        ctrl = SimulationController(rhombus_ret)
        ctrl.fast_forward()
        self.assertEqual(rhombus_ret.completed_count, 1)



if __name__ == "__main__":
    unittest.main()
