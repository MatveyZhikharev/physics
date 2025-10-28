from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def _safe_time_grid(t_max: float, dt: float) -> np.ndarray:
    if t_max <= 0 or dt <= 0:
        return np.array([0.0, max(t_max, 0.0)], dtype=float)
    n = int(np.floor(t_max / dt))
    t = dt * np.arange(n + 1, dtype=float)
    if t[-1] < t_max:
        t = np.append(t, float(t_max))
    t = np.unique(np.round(t, 15))
    return t


def reflect_velocity(v: np.ndarray, n_unit: np.ndarray) -> np.ndarray:
    n = n_unit / np.linalg.norm(n_unit)
    return v - 2.0 * np.dot(v, n) * n


def collide_elastic_two_spheres(
        m1: float, m2: float,
        x1: np.ndarray, v1: np.ndarray,
        x2: np.ndarray, v2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    dx = x1 - x2
    dist2 = float(np.dot(dx, dx))
    if dist2 == 0.0:
        return v1.copy(), v2.copy()
    factor1 = (2.0 * m2 / (m1 + m2)) * float(np.dot(v1 - v2, dx)) / dist2
    factor2 = (2.0 * m1 / (m1 + m2)) * float(np.dot(v2 - v1, -dx)) / dist2
    v1p = v1 - factor1 * dx
    v2p = v2 - factor2 * (-dx)
    return v1p, v2p


@dataclass
class Ball:
    m: float
    R: float
    x: np.ndarray
    v: np.ndarray


def contact_force(delta: float, n_unit: np.ndarray, k: float, p: float) -> np.ndarray:
    if delta <= 0.0:
        return np.zeros_like(n_unit)
    return (k * (delta ** p)) * n_unit


def potential_energy(delta: float, k: float, p: float) -> float:
    if delta <= 0.0:
        return 0.0
    return k * (delta ** (p + 1.0)) / (p + 1.0)


def potential_energy_vec(delta_vec: np.ndarray, k: float, p: float) -> np.ndarray:
    U = np.zeros_like(delta_vec)
    mask = delta_vec > 0.0
    U[mask] = k * delta_vec[mask] ** (p + 1.0) / (p + 1.0)
    return U


def two_ball_ode(
        t: float, y: np.ndarray, m1: float, m2: float, R1: float, R2: float, k: float, p: float
) -> np.ndarray:
    x1 = y[0:2]
    v1 = y[2:4]
    x2 = y[4:6]
    v2 = y[6:8]
    r = x1 - x2
    dist = float(np.linalg.norm(r))
    delta = (R1 + R2) - dist
    if dist > 0.0:
        n = r / dist
    else:
        n = np.array([1.0, 0.0])
    F = contact_force(delta, n, k, p)
    a1 = -F / m1
    a2 = F / m2
    return np.hstack([v1, a1, v2, a2])


def simulate_two_balls(
        b1: Ball, b2: Ball,
        k: float = 5e5, p: float = 1.5,
        t_max: float = 0.01, dt: float = 2e-6,
        rtol: float = 1e-8, atol: float = 1e-10,
        method: str = "RK45"
):
    y0 = np.hstack([b1.x, b1.v, b2.x, b2.v])
    t_eval = _safe_time_grid(t_max, dt)
    sol = solve_ivp(
        two_ball_ode, (0.0, t_max), y0,
        t_eval=t_eval, rtol=rtol, atol=atol, max_step=dt, method=method,
        args=(b1.m, b2.m, b1.R, b2.R, k, p)
    )
    Y = sol.y.T
    x1 = Y[:, 0:2]
    v1 = Y[:, 2:4]
    x2 = Y[:, 4:6]
    v2 = Y[:, 6:8]
    r = x1 - x2
    dist = np.linalg.norm(r, axis=1)
    delta = np.maximum((b1.R + b2.R) - dist, 0.0)
    KE = 0.5 * b1.m * np.sum(v1 ** 2, axis=1) + 0.5 * b2.m * np.sum(v2 ** 2, axis=1)
    U = potential_energy_vec(delta, k, p)
    E = KE + U
    P = b1.m * v1 + b2.m * v2
    return sol.t, x1, v1, x2, v2, delta, E, P


def wall_contact_force(
        x: np.ndarray, R: float, n_unit: np.ndarray, x0_on_plane: np.ndarray, k: float, p: float
) -> np.ndarray:
    n = n_unit / np.linalg.norm(n_unit)
    s = float(np.dot(x - x0_on_plane, n))
    delta = R - s
    if delta <= 0.0:
        return np.zeros_like(n)
    return (k * (delta ** p)) * n


def simulate_ball_wall(
        m: float = 0.17, R: float = 0.0285,
        x: np.ndarray = np.array([0.10, 0.0]),
        v: np.ndarray = np.array([-3.0, 0.4]),
        n_unit: np.ndarray = np.array([1.0, 0.0]),
        x0_on_plane: np.ndarray = np.array([0.0, 0.0]),
        k: float = 1e6, p: float = 1.5,
        t_max: float = 0.01, dt: float = 2e-6,
        rtol: float = 1e-8, atol: float = 1e-10,
        method: str = "RK45"
):
    def ode(t, y):
        x_ = y[:2]
        v_ = y[2:]
        F = wall_contact_force(x_, R, n_unit, x0_on_plane, k, p)
        a = F / m
        return np.hstack([v_, a])

    y0 = np.hstack([x, v])
    t_eval = _safe_time_grid(t_max, dt)
    sol = solve_ivp(
        ode, (0.0, t_max), y0,
        t_eval=t_eval, rtol=rtol, atol=atol, max_step=dt, method=method
    )
    X = sol.y[:2, :].T
    V = sol.y[2:, :].T
    KE = 0.5 * m * np.sum(V ** 2, axis=1)
    return sol.t, X, V, KE


def relative_energy_drift(E: np.ndarray) -> float:
    e0 = max(float(E[0]), 1e-16)
    return (float(np.max(E)) - float(np.min(E))) / e0


def momentum_drift(P: np.ndarray) -> float:
    return float(np.linalg.norm(P[-1] - P[0]))


def run_autotests(verbose: bool = True) -> None:
    tests = [
        (0.17, 0.17, [-0.06, 0.0], [2.5, 0.0], [0.06, 0.0], [0.0, 0.0], 5e5, 0.006, 2e-4, 2e-6),
        (0.20, 0.10, [-0.06, -0.005], [2.2, 0.3], [0.06, 0.004], [0.0, 0.0], 8e5, 0.008, 2e-4, 5e-6)
    ]

    for i, (m1, m2, x1, v1, x2, v2, k, t_max, tol_E, tol_P) in enumerate(tests, 1):
        b1 = Ball(m1, 0.0285, np.array(x1), np.array(v1))
        b2 = Ball(m2, 0.0285, np.array(x2), np.array(v2))
        t, x1_sim, v1_sim, x2_sim, v2_sim, delta, E, P = simulate_two_balls(b1, b2, k=k, t_max=t_max)
        energy_drift = relative_energy_drift(E)
        momentum_drift_val = momentum_drift(P)
        assert energy_drift < tol_E, f"Тест{i} энергия: {energy_drift:.2e} >= {tol_E:.2e}"
        assert momentum_drift_val < tol_P, f"Тест{i} импульс: {momentum_drift_val:.2e} >= {tol_P:.2e}"
        if verbose:
            print(f"[OK] Тест{i}: энергия/импульс в норме (дрейф: {energy_drift:.2e})")

    v_in = np.array([-1.0, 0.2])
    t, X, V, KE = simulate_ball_wall(x=np.array([0.05, 0.0]), v=v_in, k=1e6, p=1.0, t_max=0.05)
    assert X[-1, 0] > X[0, 0], "Шар не отлетел от стенки"
    assert abs(V[-1, 1] - v_in[1]) < 0.05, "Тангенциальная компонента изменилась"
    assert np.sign(V[-1, 0]) != np.sign(v_in[0]), "Нормальная компонента не изменила знак"
    if verbose:
        print("[OK] Тест3: отражение о стенку")
        print("Все автотесты пройдены успешно.")


def validate_float_input(prompt, default, min_val=None, max_val=None):
    while True:
        try:
            value = input(prompt).strip()
            if not value:
                value = default
            else:
                value = float(value)

            if min_val is not None and value < min_val:
                print(f"Ошибка: значение должно быть не меньше {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Ошибка: значение должно быть не больше {max_val}")
                continue

            return value
        except ValueError:
            print("Ошибка: введите корректное число")


def validate_vector_input(prompt, default, min_val=None, max_val=None):
    while True:
        try:
            value = input(prompt).strip()
            if not value:
                values = default
            else:
                values = list(map(float, value.split()))
                if len(values) != 2:
                    print("Ошибка: введите два числа через пробел")
                    continue

            if min_val is not None:
                if any(v < min_val for v in values):
                    print(f"Ошибка: все значения должны быть не меньше {min_val}")
                    continue
            if max_val is not None:
                if any(v > max_val for v in values):
                    print(f"Ошибка: все значения должны быть не больше {max_val}")
                    continue

            return values
        except ValueError:
            print("Ошибка: введите корректные числа")


def get_user_input():
    print("\n" + "=" * 50)
    print("МОДЕЛИРОВАНИЕ СТОЛКНОВЕНИЙ БИЛЬЯРДНЫХ ШАРОВ")
    print("=" * 50)

    print("\nВыберите тип моделирования:")
    print("1. Столкновение двух шаров")
    print("2. Отражение шара от стенки")

    while True:
        choice = input("Ваш выбор (1 или 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Ошибка: введите 1 или 2")

    print("\n--- Общие параметры ---")
    m = validate_float_input("Масса шара (кг) [0.17]: ", 0.17, min_val=0.001, max_val=10.0)
    R = validate_float_input("Радиус шара (м) [0.0285]: ", 0.0285, min_val=0.001, max_val=0.04)

    if choice == "1":
        print("\n--- Параметры первого шара ---")
        x1 = validate_vector_input("Начальная позиция (x y) [-0.06 0.0]: ", [-0.06, 0.0], min_val=-1.0, max_val=1.0)
        v1 = validate_vector_input("Начальная скорость (vx vy) [2.5 0.0]: ", [2.5, 0.0], min_val=-50.0, max_val=50.0)

        print("\n--- Параметры второго шара ---")
        x2 = validate_vector_input("Начальная позиция (x y) [0.06 0.0]: ", [0.06, 0.0], min_val=-1.0, max_val=1.0)
        v2 = validate_vector_input("Начальная скорость (vx vy) [0.0 0.0]: ", [0.0, 0.0], min_val=-50.0, max_val=50.0)

        dist = np.linalg.norm(np.array(x1) - np.array(x2))
        if dist < 2 * R:
            print(f"Предупреждение: шары изначально перекрываются (расстояние {dist:.3f} < {2 * R:.3f})")

        return {
            'type': 'two_balls',
            'm': m, 'R': R,
            'ball1': {'x': np.array(x1), 'v': np.array(v1)},
            'ball2': {'x': np.array(x2), 'v': np.array(v2)}
        }

    else:
        print("\n--- Параметры шара ---")
        x = validate_vector_input("Начальная позиция (x y) [0.05 0.0]: ", [0.05, 0.0], min_val=-1.0, max_val=1.0)
        v = validate_vector_input("Начальная скорость (vx vy) [-1.0 0.2]: ", [-1.0, 0.2], min_val=-50.0, max_val=50.0)

        print("\n--- Параметры стенки ---")
        wall_norm = validate_vector_input("Нормаль стенки (nx ny) [1.0 0.0]: ", [1.0, 0.0], min_val=-1.0, max_val=1.0)
        wall_pos = validate_float_input("Позиция стенки по x [0.0]: ", 0.0, min_val=-1.0, max_val=1.0)

        wall_norm = np.array(wall_norm)
        wall_norm = wall_norm / np.linalg.norm(wall_norm)

        return {
            'type': 'ball_wall',
            'm': m, 'R': R,
            'ball': {'x': np.array(x), 'v': np.array(v)},
            'wall': {'normal': wall_norm, 'position': wall_pos}
        }


def get_simulation_params():
    print("\n--- Параметры симуляции ---")

    print("Выберите закон контакта:")
    print("1. Закон Гука (F ~ Δx)")
    print("2. Закон Герца (F ~ Δx^(3/2))")

    while True:
        law_choice = input("Ваш выбор (1 или 2): ").strip()
        if law_choice in ['1', '2']:
            break
        print("Ошибка: введите 1 или 2")

    p = 1.0 if law_choice == "1" else 1.5

    default_k = 5e5 if p == 1.5 else 1e6
    k = validate_float_input(f"Коэффициент жесткости [{default_k:.0e}]: ", default_k, min_val=1e3, max_val=1e9)

    t_max = validate_float_input("Время моделирования (с) [0.01]: ", 0.01, min_val=0.001, max_val=0.1)

    return {'p': p, 'k': k, 't_max': t_max}


def plot_billiard_table(results, params, sim_params):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_traj, ax_energy = axes

    colors = {
        'table': '#2E8B57', 'ball1': '#1f77b4', 'ball2': '#ff7f0e',
        'wall': '#8B4513', 'collision': '#d62728', 'text_bg': 'lightblue', 'params_bg': 'lightyellow'
    }

    table = plt.Rectangle((-0.1, -0.1), 0.2, 0.2, fill=True,
                          facecolor=colors['table'], alpha=0.2, linewidth=3, edgecolor='black')
    ax_traj.add_patch(table)

    if params['type'] == 'two_balls':
        t, x1, v1, x2, v2, delta, E, P = results

        ax_traj.plot(x1[:, 0], x1[:, 1], '-', color=colors['ball1'], linewidth=2, label='Шар 1')
        ax_traj.plot(x2[:, 0], x2[:, 1], '-', color=colors['ball2'], linewidth=2, label='Шар 2')

        ax_traj.plot(x1[0, 0], x1[0, 1], 'o', color=colors['ball1'], markersize=8,
                     markeredgecolor='black', markeredgewidth=1, label='Начало шар 1')
        ax_traj.plot(x2[0, 0], x2[0, 1], 'o', color=colors['ball2'], markersize=8,
                     markeredgecolor='black', markeredgewidth=1, label='Начало шар 2')
        ax_traj.plot(x1[-1, 0], x1[-1, 1], 's', color=colors['ball1'], markersize=6,
                     markeredgecolor='black', markeredgewidth=1, label='Конец шар 1')
        ax_traj.plot(x2[-1, 0], x2[-1, 1], 's', color=colors['ball2'], markersize=6,
                     markeredgecolor='black', markeredgewidth=1, label='Конец шар 2')

        collision_idx = np.argmax(delta > 0)
        if 0 < collision_idx < len(x1):
            cx = (x1[collision_idx, 0] + x2[collision_idx, 0]) / 2
            cy = (x1[collision_idx, 1] + x2[collision_idx, 1]) / 2
            ax_traj.plot(cx, cy, '*', color=colors['collision'], markersize=10,
                         markeredgecolor='black', markeredgewidth=1, label='Столкновение')

        KE_ball1 = 0.5 * params['m'] * np.sum(v1 ** 2, axis=1)
        KE_ball2 = 0.5 * params['m'] * np.sum(v2 ** 2, axis=1)
        KE_total = KE_ball1 + KE_ball2
        U = potential_energy_vec(delta, sim_params['k'], sim_params['p'])

        ax_energy.plot(t * 1000, KE_total, 'g-', linewidth=2, label='Кинетическая')
        ax_energy.plot(t * 1000, U, 'r-', linewidth=2, label='Потенциальная')
        ax_energy.plot(t * 1000, E, 'b-', linewidth=2, label='Полная')

        if 0 < collision_idx < len(t):
            ax_energy.axvline(x=t[collision_idx] * 1000, color='red', linestyle='--',
                              alpha=0.7, label='Столкновение')

        energy_drift = relative_energy_drift(E)
        momentum_drift_val = momentum_drift(P)
        info_text = f"""РЕЗУЛЬТАТЫ СТОЛКНОВЕНИЯ:

ЭНЕРГИЯ:
Начальная: {E[0]:.4f} Дж
Конечная: {E[-1]:.4f} Дж
Дрейф: {energy_drift:.2e}

ИМПУЛЬС:
Изменение: {momentum_drift_val:.2e}

СКОРОСТИ:
Шар 1: {np.linalg.norm(v1[-1]):.2f} м/с
Шар 2: {np.linalg.norm(v2[-1]):.2f} м/с"""

    else:
        t, X, V, KE = results

        ax_traj.plot(X[:, 0], X[:, 1], '-', color=colors['ball1'], linewidth=2, label='Траектория')
        ax_traj.plot(X[0, 0], X[0, 1], 'o', color=colors['ball1'], markersize=8,
                     markeredgecolor='black', markeredgewidth=1, label='Начало')
        ax_traj.plot(X[-1, 0], X[-1, 1], 's', color=colors['ball1'], markersize=6,
                     markeredgecolor='black', markeredgewidth=1, label='Конец')

        wall_norm = params['wall']['normal']
        wall_pos = params['wall']['position']
        if abs(wall_norm[0]) > 0.5:
            ax_traj.axvline(x=wall_pos, color=colors['wall'], linewidth=3, label='Стенка')
        else:
            ax_traj.axhline(y=wall_pos, color=colors['wall'], linewidth=3, label='Стенка')

        ax_energy.plot(t * 1000, KE, 'b-', linewidth=2, label='Кинетическая')

        energy_drift = (np.max(KE) - np.min(KE)) / np.max(KE)
        v_final = np.linalg.norm(V[-1])
        energy_loss = 100 * (KE[0] - KE[-1]) / KE[0] if KE[0] > 0 else 0

        info_text = f"""РЕЗУЛЬТАТЫ ОТРАЖЕНИЯ:

ЭНЕРГИЯ:
Начальная: {KE[0]:.4f} Дж
Конечная: {KE[-1]:.4f} Дж  
Потери: {energy_loss:.1f}%

СКОРОСТЬ:
Конечная: {v_final:.2f} м/с"""

    ax_traj.set_xlim(-0.1, 0.1)
    ax_traj.set_ylim(-0.1, 0.1)
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlabel('X (м)')
    ax_traj.set_ylabel('Y (м)')

    law_name = "Гука" if sim_params['p'] == 1.0 else "Герца"
    ax_traj.set_title(f'ТРАЕКТОРИИ ({law_name})')
    ax_traj.legend()

    ax_energy.set_xlabel('Время (мс)')
    ax_energy.set_ylabel('Энергия (Дж)')
    ax_energy.set_title('ЭНЕРГИЯ')
    ax_energy.legend()
    ax_energy.grid(True, alpha=0.3)

    ax_traj.text(0.02, 0.98, info_text, transform=ax_traj.transAxes, fontsize=8,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor=colors['text_bg'], alpha=0.8))

    sim_info = f"""ПАРАМЕТРЫ:
Закон: {law_name}
k = {sim_params['k']:.0e} Н/м
t = {sim_params['t_max']} с
m = {params['m']} кг
R = {params['R']} м"""

    ax_energy.text(0.98, 0.98, sim_info, transform=ax_energy.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor=colors['params_bg'], alpha=0.8))

    plt.tight_layout()
    plt.show()


def main():
    print("Запуск автотестов...")
    run_autotests(verbose=True)
    print("\nАвтотесты завершены успешно!")

    while True:
        try:
            print("\n" + "=" * 50)
            print("НАСТРОЙКА ПАРАМЕТРОВ МОДЕЛИРОВАНИЯ")
            print("=" * 50)

            params = get_user_input()
            sim_params = get_simulation_params()

            print("\nЗапуск моделирования...")

            if params['type'] == 'two_balls':
                b1 = Ball(params['m'], params['R'], params['ball1']['x'], params['ball1']['v'])
                b2 = Ball(params['m'], params['R'], params['ball2']['x'], params['ball2']['v'])
                results = simulate_two_balls(b1, b2, k=sim_params['k'], p=sim_params['p'], t_max=sim_params['t_max'])
            else:
                results = simulate_ball_wall(
                    m=params['m'], R=params['R'], x=params['ball']['x'], v=params['ball']['v'],
                    n_unit=params['wall']['normal'], x0_on_plane=np.array([params['wall']['position'], 0.0]),
                    k=sim_params['k'], p=sim_params['p'], t_max=sim_params['t_max']
                )

            print("Моделирование завершено. Построение графика...")
            plot_billiard_table(results, params, sim_params)

            cont = input("\nПровести еще одно моделирование? (y/n): ").strip().lower()
            if cont not in ['y', 'д']:
                print("Выход из программы.")
                break

        except KeyboardInterrupt:
            print("\n\nПрограмма прервана пользователем.")
            break
        except Exception as e:
            print(f"\nОшибка: {e}")
            print("Попробуйте еще раз.")


if __name__ == "__main__":
    main()