import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class BilliardBall:
    def __init__(self, mass, radius, position, velocity, color='gray'):
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        self.trail = [self.position.copy()]

    def update(self, dt):
        self.position += self.velocity * dt
        self.trail.append(self.position.copy())
        if len(self.trail) > 50:
            self.trail.pop(0)

    def distance_to(self, other):
        return np.linalg.norm(self.position - other.position)

    def is_colliding_with(self, other):
        return self.distance_to(other) <= (self.radius + other.radius)


class BilliardTable:
    def __init__(self, width=2.0, height=1.0, pocket_radius=0.1):
        self.width = width
        self.height = height
        self.pocket_radius = pocket_radius
        self.pockets = [
            (pocket_radius, pocket_radius),
            (width - pocket_radius, pocket_radius),
            (pocket_radius, height - pocket_radius),
            (width - pocket_radius, height - pocket_radius),
            (width / 2, pocket_radius),
            (width / 2, height - pocket_radius)
        ]


class ElasticCollisionSolver:
    def __init__(self, law_type='elastic', k=1000, exponent=1.0, damping=10.0):
        self.law_type = law_type  # 'elastic', 'hooke', 'hertz'
        self.k = k  # Уменьшил коэффициент для стабильности
        self.exponent = exponent
        self.damping = damping  # Увеличил демпфирование

    def wall_collision(self, ball, table):
        if self.law_type == 'elastic':
            self._elastic_wall_collision(ball, table)
        else:
            self._deformable_wall_collision(ball, table)

    def ball_collision(self, ball1, ball2):
        if self.law_type == 'elastic':
            self._elastic_ball_collision(ball1, ball2)
        else:
            self._deformable_ball_collision(ball1, ball2)

    def _elastic_wall_collision(self, ball, table):
        """Абсолютно упругое отталкивание от стенок"""
        collision_occurred = False

        if ball.position[0] - ball.radius <= 0:
            ball.position[0] = ball.radius
            ball.velocity[0] = -ball.velocity[0]
            collision_occurred = True
        elif ball.position[0] + ball.radius >= table.width:
            ball.position[0] = table.width - ball.radius
            ball.velocity[0] = -ball.velocity[0]
            collision_occurred = True

        if ball.position[1] - ball.radius <= 0:
            ball.position[1] = ball.radius
            ball.velocity[1] = -ball.velocity[1]
            collision_occurred = True
        elif ball.position[1] + ball.radius >= table.height:
            ball.position[1] = table.height - ball.radius
            ball.velocity[1] = -ball.velocity[1]
            collision_occurred = True

        return collision_occurred

    def _deformable_wall_collision(self, ball, table):
        """Отталкивание от стенок с учётом деформации"""
        force = np.zeros(2)
        collision_occurred = False

        # Проверяем столкновения со всеми стенками
        # Левая стенка
        if ball.position[0] - ball.radius < 0:
            overlap = ball.radius - ball.position[0]
            if overlap > 0:
                if self.law_type == 'hooke':
                    force_magnitude = self.k * (overlap ** self.exponent)
                else:  # hertz
                    force_magnitude = self.k * (overlap ** 1.5)

                # Демпфирование - только если шар движется в стенку
                if ball.velocity[0] < 0:
                    damping_force = self.damping * abs(ball.velocity[0])
                else:
                    damping_force = 0

                total_force = force_magnitude + damping_force
                force[0] += total_force
                collision_occurred = True

        # Правая стенка
        if ball.position[0] + ball.radius > table.width:
            overlap = ball.position[0] + ball.radius - table.width
            if overlap > 0:
                if self.law_type == 'hooke':
                    force_magnitude = self.k * (overlap ** self.exponent)
                else:  # hertz
                    force_magnitude = self.k * (overlap ** 1.5)

                # Демпфирование - только если шар движется в стенку
                if ball.velocity[0] > 0:
                    damping_force = self.damping * abs(ball.velocity[0])
                else:
                    damping_force = 0

                total_force = force_magnitude + damping_force
                force[0] -= total_force
                collision_occurred = True

        # Нижняя стенка
        if ball.position[1] - ball.radius < 0:
            overlap = ball.radius - ball.position[1]
            if overlap > 0:
                if self.law_type == 'hooke':
                    force_magnitude = self.k * (overlap ** self.exponent)
                else:  # hertz
                    force_magnitude = self.k * (overlap ** 1.5)

                # Демпфирование - только если шар движется в стенку
                if ball.velocity[1] < 0:
                    damping_force = self.damping * abs(ball.velocity[1])
                else:
                    damping_force = 0

                total_force = force_magnitude + damping_force
                force[1] += total_force
                collision_occurred = True

        # Верхняя стенка
        if ball.position[1] + ball.radius > table.height:
            overlap = ball.position[1] + ball.radius - table.height
            if overlap > 0:
                if self.law_type == 'hooke':
                    force_magnitude = self.k * (overlap ** self.exponent)
                else:  # hertz
                    force_magnitude = self.k * (overlap ** 1.5)

                # Демпфирование - только если шар движется в стенку
                if ball.velocity[1] > 0:
                    damping_force = self.damping * abs(ball.velocity[1])
                else:
                    damping_force = 0

                total_force = force_magnitude + damping_force
                force[1] -= total_force
                collision_occurred = True

        # Применяем силу (F = ma)
        if np.linalg.norm(force) > 0:
            acceleration = force / ball.mass
            ball.velocity -= acceleration * 0.001  # Уменьшил шаг интегрирования

        return collision_occurred

    def _elastic_ball_collision(self, ball1, ball2):
        """Абсолютно упругое столкновение шаров"""
        if not ball1.is_colliding_with(ball2):
            return

        collision_vector = ball2.position - ball1.position
        distance = np.linalg.norm(collision_vector)

        if distance == 0:
            return

        collision_normal = collision_vector / distance

        v1n = np.dot(ball1.velocity, collision_normal)
        v2n = np.dot(ball2.velocity, collision_normal)

        # Только если шары сближаются
        if v1n - v2n <= 0:
            return

        m1, m2 = ball1.mass, ball2.mass
        v1n_new = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2)
        v2n_new = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2)

        ball1.velocity += (v1n_new - v1n) * collision_normal
        ball2.velocity += (v2n_new - v2n) * collision_normal

        # Разделяем шары, чтобы избежать залипания
        overlap = ball1.radius + ball2.radius - distance
        if overlap > 0:
            separation = overlap * 0.5
            ball1.position -= separation * collision_normal
            ball2.position += separation * collision_normal

    def _deformable_ball_collision(self, ball1, ball2):
        """Столкновение шаров с учётом деформации"""
        if not ball1.is_colliding_with(ball2):
            return

        r_vec = ball2.position - ball1.position
        distance = np.linalg.norm(r_vec)

        if distance == 0:
            return

        overlap = ball1.radius + ball2.radius - distance

        if overlap <= 0:
            return

        normal = r_vec / distance

        # Проверяем, что шары сближаются
        relative_velocity = np.dot(ball2.velocity - ball1.velocity, normal)
        if relative_velocity > 0:  # Шары удаляются друг от друга
            return

        # Сила по закону Гука или Герца
        if self.law_type == 'hooke':
            force_magnitude = self.k * (overlap ** self.exponent)
        else:  # hertz
            force_magnitude = self.k * (overlap ** 1.5)

        # Демпфирование
        damping_force = self.damping * abs(relative_velocity)

        total_force = (force_magnitude + damping_force) * normal

        # Применяем силу (F = ma)
        ball1.velocity += (total_force / ball1.mass) * 0.001  # Уменьшил шаг
        ball2.velocity -= (total_force / ball2.mass) * 0.001

        # Легкое разделение шаров для стабильности
        separation = overlap * 0.05  # Уменьшил коэффициент разделения
        ball1.position -= separation * normal
        ball2.position += separation * normal


class BilliardGame:
    def __init__(self, law_type='elastic', k=1000, exponent=1.0, damping=10.0):
        self.table = BilliardTable()
        self.balls = []
        self.collision_solver = ElasticCollisionSolver(law_type, k, exponent, damping)
        self.time = 0
        self.dt = 0.01
        # Для сохранения истории
        self.history = {
            'time': [],
            'positions': [],
            'velocities': [],
            'energy': [],
            'momentum': []
        }

    def setup_game(self, ball1_params, ball2_params):
        self.balls = []
        self.history = {'time': [], 'positions': [], 'velocities': [], 'energy': [], 'momentum': []}

        # Биток
        cue_ball = BilliardBall(
            mass=ball1_params['mass'],
            radius=ball1_params['radius'],
            position=ball1_params['position'],
            velocity=ball1_params['velocity'],
            color=ball1_params['color']
        )
        self.add_ball(cue_ball)

        # Целевой шар
        target_ball = BilliardBall(
            mass=ball2_params['mass'],
            radius=ball2_params['radius'],
            position=ball2_params['position'],
            velocity=ball2_params['velocity'],
            color=ball2_params['color']
        )
        self.add_ball(target_ball)

    def add_ball(self, ball):
        self.balls.append(ball)

    def calculate_energy(self):
        """Вычисление полной кинетической энергии системы"""
        total_energy = 0
        for ball in self.balls:
            total_energy += 0.5 * ball.mass * np.dot(ball.velocity, ball.velocity)
        return total_energy

    def calculate_momentum(self):
        """Вычисление полного импульса системы"""
        total_momentum = np.zeros(2)
        for ball in self.balls:
            total_momentum += ball.mass * ball.velocity
        return total_momentum

    def update(self):
        # Обновляем позиции
        for ball in self.balls:
            ball.update(self.dt)

        # Обрабатываем столкновения со стенками
        for ball in self.balls:
            self.collision_solver.wall_collision(ball, self.table)

        # Обрабатываем столкновения между шарами
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                self.collision_solver.ball_collision(self.balls[i], self.balls[j])

        # Сохраняем данные
        self.history['time'].append(self.time)
        self.history['positions'].append([ball.position.copy() for ball in self.balls])
        self.history['velocities'].append([ball.velocity.copy() for ball in self.balls])
        self.history['energy'].append(self.calculate_energy())
        self.history['momentum'].append(self.calculate_momentum())

        self.time += self.dt


def get_user_input():
    """Получение параметров от пользователя"""
    print("=== НАСТРОЙКА ПАРАМЕТРОВ БИЛЬЯРДА ===")

    # Выбор закона физики
    print("\nВыберите закон физики:")
    print("1 - Абсолютно упругие столкновения (рекомендуется)")
    print("2 - Закон Гука (F ∼ -Δx)")
    print("3 - Закон Герца (F ∼ -Δx³/²)")

    law_choice = input("Ваш выбор (1-3): ").strip()
    law_types = {'1': 'elastic', '2': 'hooke', '3': 'hertz'}
    law_type = law_types.get(law_choice, 'elastic')

    # Более стабильные параметры по умолчанию
    k = 1000 if law_type != 'elastic' else 0
    exponent = 1.0
    damping = 10.0 if law_type != 'elastic' else 0

    if law_type != 'elastic':
        try:
            k = float(input(f"Коэффициент упругости k (по умолчанию {k}): ") or str(k))
            if law_type == 'hooke':
                exponent = float(input("Показатель степени (по умолчанию 1.0): ") or "1.0")
            damping = float(input(f"Коэффициент демпфирования (по умолчанию {damping}): ") or str(damping))
        except ValueError:
            print("Использую значения по умолчанию")

    # Параметры первого шара (битка)
    print("\n=== ПАРАМЕТРЫ ПЕРВОГО ШАРА (БИТОК) ===")
    ball1_params = {}

    ball1_params['mass'] = float(input("Масса (по умолчанию 1.0): ") or "1.0")
    ball1_params['radius'] = float(input("Радиус (по умолчанию 0.05): ") or "0.05")

    print("Начальная позиция (x y):")
    pos_input = input("По умолчанию 0.3 0.5: ") or "0.3 0.5"
    ball1_params['position'] = list(map(float, pos_input.split()))

    print("Начальная скорость (vx vy):")
    vel_input = input("По умолчанию 2.0 0.2: ") or "2.0 0.2"  # Уменьшил скорость
    ball1_params['velocity'] = list(map(float, vel_input.split()))

    color_input = input("Цвет (по умолчанию gray): ") or "gray"
    ball1_params['color'] = color_input.lower()

    # Параметры второго шара
    print("\n=== ПАРАМЕТРЫ ВТОРОГО ШАРА ===")
    ball2_params = {}

    ball2_params['mass'] = float(input("Масса (по умолчанию 1.0): ") or "1.0")
    ball2_params['radius'] = float(input("Радиус (по умолчанию 0.05): ") or "0.05")

    print("Начальная позиция (x y):")
    pos_input = input("По умолчанию 1.5 0.5: ") or "1.5 0.5"
    ball2_params['position'] = list(map(float, pos_input.split()))

    print("Начальная скорость (vx vy):")
    vel_input = input("По умолчанию 0.0 0.0: ") or "0.0 0.0"
    ball2_params['velocity'] = list(map(float, vel_input.split()))

    color_input = input("Цвет (по умолчанию red): ") or "red"
    ball2_params['color'] = color_input.lower()

    return law_type, k, exponent, damping, ball1_params, ball2_params


def analyze_and_plot_results(game):
    """Анализ результатов и построение графиков"""
    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ СИМУЛЯЦИИ")
    print("=" * 60)

    # Извлекаем данные из истории
    time = np.array(game.history['time'])
    energy = np.array(game.history['energy'])
    momentum = np.array(game.history['momentum'])
    positions = np.array(game.history['positions'])

    # Вычисляем основные характеристики
    initial_energy = energy[0]
    final_energy = energy[-1]
    energy_change = abs(final_energy - initial_energy) / initial_energy * 100

    initial_momentum = np.linalg.norm(momentum[0])
    final_momentum = np.linalg.norm(momentum[-1])
    momentum_change = abs(final_momentum - initial_momentum) / initial_momentum * 100

    # Вывод вычисленных значений
    print(f"\nЭНЕРГЕТИЧЕСКИЕ ХАРАКТЕРИСТИКИ:")
    print(f"Начальная энергия системы: {initial_energy:.6f} Дж")
    print(f"Конечная энергия системы: {final_energy:.6f} Дж")
    print(f"Изменение энергии: {energy_change:.4f}%")
    print(f"Сохранение энергии: {100 - energy_change:.4f}%")

    print(f"\nИМПУЛЬСНЫЕ ХАРАКТЕРИСТИКИ:")
    print(f"Начальный импульс системы: {initial_momentum:.6f} кг·м/с")
    print(f"Конечный импульс системы: {final_momentum:.6f} кг·м/с")
    print(f"Изменение импульса: {momentum_change:.4f}%")
    print(f"Сохранение импульса: {100 - momentum_change:.4f}%")

    # Анализ столкновений
    velocities = np.array(game.history['velocities'])
    ball1_v = velocities[:, 0]  # Скорости первого шара
    ball2_v = velocities[:, 1]  # Скорости второго шара

    # Находим момент столкновения (когда скорости резко меняются)
    ball1_speed_change = np.linalg.norm(np.diff(ball1_v, axis=0), axis=1)
    collision_time_idx = np.argmax(ball1_speed_change) + 1 if len(ball1_speed_change) > 0 else 0

    if collision_time_idx > 0 and collision_time_idx < len(time):
        collision_time = time[collision_time_idx]
        print(f"\nМомент столкновения: {collision_time:.3f} с")

        # Скорости до и после столкновения
        v1_before = ball1_v[collision_time_idx - 1]
        v2_before = ball2_v[collision_time_idx - 1]
        v1_after = ball1_v[collision_time_idx]
        v2_after = ball2_v[collision_time_idx]

        print(f"Скорость шара 1 до столкновения: ({v1_before[0]:.3f}, {v1_before[1]:.3f}) м/с")
        print(f"Скорость шара 2 до столкновения: ({v2_before[0]:.3f}, {v2_before[1]:.3f}) м/с")
        print(f"Скорость шара 1 после столкновения: ({v1_after[0]:.3f}, {v1_after[1]:.3f}) м/с")
        print(f"Скорость шара 2 после столкновения: ({v2_after[0]:.3f}, {v2_after[1]:.3f}) м/с")

    # Построение графиков
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # График 1: Траектории шаров
    ball1_pos = positions[:, 0]
    ball2_pos = positions[:, 1]

    ax1.plot(ball1_pos[:, 0], ball1_pos[:, 1], 'b-', label='Шар 1', linewidth=2, alpha=0.7)
    ax1.plot(ball2_pos[:, 0], ball2_pos[:, 1], 'r-', label='Шар 2', linewidth=2, alpha=0.7)
    ax1.scatter(ball1_pos[0, 0], ball1_pos[0, 1], c='blue', s=100, marker='o', label='Начало шар 1')
    ax1.scatter(ball2_pos[0, 0], ball2_pos[0, 1], c='red', s=100, marker='o', label='Начало шар 2')
    ax1.scatter(ball1_pos[-1, 0], ball1_pos[-1, 1], c='blue', s=100, marker='s', label='Конец шар 1')
    ax1.scatter(ball2_pos[-1, 0], ball2_pos[-1, 1], c='red', s=100, marker='s', label='Конец шар 2')

    ax1.set_xlim(0, game.table.width)
    ax1.set_ylim(0, game.table.height)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X координата (м)')
    ax1.set_ylabel('Y координата (м)')
    ax1.set_title('Траектории движения шаров')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Сохранение энергии
    ax2.plot(time, energy, 'g-', linewidth=2)
    ax2.axhline(y=initial_energy, color='r', linestyle='--', alpha=0.7, label='Начальная энергия')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Энергия (Дж)')
    ax2.set_title('Сохранение кинетической энергии')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # График 3: Сохранение импульса
    momentum_magnitude = np.linalg.norm(momentum, axis=1)
    ax3.plot(time, momentum_magnitude, 'purple', linewidth=2)
    ax3.axhline(y=initial_momentum, color='r', linestyle='--', alpha=0.7, label='Начальный импульс')
    ax3.set_xlabel('Время (с)')
    ax3.set_ylabel('Импульс (кг·м/с)')
    ax3.set_title('Сохранение импульса системы')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Скорости шаров
    ball1_speed = np.linalg.norm(ball1_v, axis=1)
    ball2_speed = np.linalg.norm(ball2_v, axis=1)

    ax4.plot(time, ball1_speed, 'b-', label='Шар 1', linewidth=2)
    ax4.plot(time, ball2_speed, 'r-', label='Шар 2', linewidth=2)
    ax4.set_xlabel('Время (с)')
    ax4.set_ylabel('Скорость (м/с)')
    ax4.set_title('Изменение скоростей шаров')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Дополнительный анализ
    print(f"\nДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ:")
    print(f"Общее время симуляции: {time[-1]:.2f} с")
    print(f"Количество временных шагов: {len(time)}")
    print(f"Максимальная скорость шара 1: {np.max(ball1_speed):.3f} м/с")
    print(f"Максимальная скорость шара 2: {np.max(ball2_speed):.3f} м/с")
    print(f"Средняя энергия системы: {np.mean(energy):.6f} Дж")
    print(f"Стандартное отклонение энергии: {np.std(energy):.6f} Дж")


def main():
    # Получаем параметры от пользователя
    law_type, k, exponent, damping, ball1_params, ball2_params = get_user_input()

    # Создаём игру с выбранными параметрами
    game = BilliardGame(law_type, k, exponent, damping)
    game.setup_game(ball1_params, ball2_params)

    # Настраиваем график для анимации
    fig, ax = plt.subplots(figsize=(12, 6))
    frames_to_simulate = 500

    def animate(frame):
        ax.clear()

        # Настраиваем график
        ax.set_xlim(-0.1, game.table.width + 0.1)
        ax.set_ylim(-0.1, game.table.height + 0.1)
        ax.set_aspect('equal')

        law_names = {
            'elastic': 'Абсолютно упругие столкновения',
            'hooke': f'Закон Гука (k={k}, n={exponent})',
            'hertz': f'Закон Герца (k={k})'
        }
        ax.set_title(f'Бильярд: {law_names[law_type]} - Время: {game.time:.2f} с')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)

        # Рисуем лузы
        for pocket in game.table.pockets:
            circle = plt.Circle(pocket, game.table.pocket_radius, color='black', alpha=0.5)
            ax.add_patch(circle)

        # Рисуем границы стола
        table_rect = plt.Rectangle((0, 0), game.table.width, game.table.height,
                                   fill=False, edgecolor='brown', linewidth=3)
        ax.add_patch(table_rect)

        # Обновляем симуляцию
        game.update()

        # Рисуем шары
        for ball in game.balls:
            circle = plt.Circle(ball.position, ball.radius, color=ball.color,
                                edgecolor='black', linewidth=2)
            ax.add_patch(circle)

            # Рисуем след
            if len(ball.trail) > 1:
                trail = np.array(ball.trail)
                ax.plot(trail[:, 0], trail[:, 1], color=ball.color, alpha=0.5, linewidth=1)

        # Отображаем информацию
        energy = game.calculate_energy()
        momentum = game.calculate_momentum()

        ax.text(0.02, 0.98, f'Время: {game.time:.2f} с',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.text(0.02, 0.92, f'Энергия: {energy:.3f} Дж',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.text(0.02, 0.86, f'Импульс: ({momentum[0]:.2f}, {momentum[1]:.2f})',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    print(f"\nЗапуск симуляции с параметрами:")
    print(f"Закон: {law_type}")
    print(f"Шар 1: масса={ball1_params['mass']}, радиус={ball1_params['radius']}, цвет={ball1_params['color']}")
    print(f"Шар 2: масса={ball2_params['mass']}, радиус={ball2_params['radius']}, цвет={ball2_params['color']}")
    print("\nРекомендация: для стабильной работы используйте абсолютно упругие столкновения (вариант 1)")

    # Запускаем анимацию
    print("\nЗапуск анимации...")
    anim = FuncAnimation(fig, animate, frames=frames_to_simulate, interval=50, repeat=False)
    plt.show()

    # После завершения анимации запускаем анализ результатов
    print("\nАнимация завершена. Запуск анализа результатов...")
    analyze_and_plot_results(game)


if __name__ == "__main__":
    main()