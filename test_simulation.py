"""
Модульные тесты для проверки НС АЦП (на базе unittest):
1. Абстрактный суперкласс BaseArchitecture.
2. Архитектура ToroidalGridArchitecture (тороид 6x4).
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
import sys
import unittest
from models import (
    BaseArchitecture, ToroidalGridArchitecture, RhomboidArchitecture,
    Star8Architecture, HoneycombPipelineArchitecture, SymmetricStar8Architecture,
    SimulationController, SimulationEngine,
    NeuronState, RequestStatus, PlayState, Request, Neuron
)


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
        self.assertEqual(win.combo_topo.count(), 6)

        # 1. Переключение на Ромб
        win.combo_topo.setCurrentIndex(1)
        self.assertEqual(win.ctrl.architecture.get_topology_info()["type"], "rhomboid")
        win.ctrl.step_forward()

        # 2. Переключение на Star-8 (4 входа)
        win.combo_topo.setCurrentIndex(2)
        info4 = win.ctrl.architecture.get_topology_info()
        self.assertEqual(info4["type"], "star8")
        self.assertEqual(info4["num_inputs"], 4)
        self.assertEqual(info4["ray_depth"], 4)
        win.ctrl.step_forward()

        # 3. Переключение на Star-8 (6 входов)
        win.combo_topo.setCurrentIndex(3)
        info6 = win.ctrl.architecture.get_topology_info()
        self.assertEqual(info6["type"], "star8")
        self.assertEqual(info6["num_inputs"], 6)
        self.assertEqual(info6["ray_depth"], 4)
        win.ctrl.step_forward()

        # 4. Переключение на Соту (Honeycomb)
        win.combo_topo.setCurrentIndex(4)
        info_honey = win.ctrl.architecture.get_topology_info()
        self.assertEqual(info_honey["type"], "honeycomb")
        win.ctrl.step_forward()

        # 5. Переключение на Симметричную звезду C8 (Sym-Star8)
        win.combo_topo.setCurrentIndex(5)
        info_sym = win.ctrl.architecture.get_topology_info()
        self.assertEqual(info_sym["type"], "sym_star8")
        win.ctrl.step_forward()

        # 6. Возврат на Тороид 6x6
        win.combo_topo.setCurrentIndex(0)
        self.assertEqual(win.combo_topo.itemText(0), "Тороидальная матрица 6x6")
        self.assertEqual(win.ctrl.architecture.get_topology_info()["type"], "toroidal_grid")
        self.assertEqual(win.ctrl.architecture.cols, 6)
        win.ctrl.step_forward()


if __name__ == "__main__":
    unittest.main()
