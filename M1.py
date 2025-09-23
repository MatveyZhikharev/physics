import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def stone_motion(y, t, g, k, resistance_type):
    """
    Система дифференциальных уравнений движения камня
    y[0] = x (координата по горизонтали)
    y[1] = y (координата по вертикали)
    y[2] = vx (скорость по горизонтали)
    y[3] = vy (скорость по вертикали)
    """
    x, y_pos, vx, vy = y
    v = math.sqrt(vx ** 2 + vy ** 2)

    # Сила сопротивления
    if resistance_type == 1:  # Вязкое трение (F ∼ v)
        Fx = -k * vx
        Fy = -k * vy
    else:  # Лобовое сопротивление (F ∼ v²)
        if v > 0:
            Fx = -k * v * vx
            Fy = -k * v * vy
        else:
            Fx, Fy = 0, 0

    # Уравнения движения
    dxdt = vx
    dydt = vy
    dvxdt = Fx
    dvydt = -g + Fy

    return [dxdt, dydt, dvxdt, dvydt]


def calculate_trajectory(angle, velocity, coeff, resistance_type, g=9.81, t_max=20, dt=0.01):
    """
    Расчет траектории движения камня
    """
    # Начальные условия
    vx0 = velocity * math.cos(angle)
    vy0 = velocity * math.sin(angle)
    y0 = [0, 0, vx0, vy0]  # x, y, vx, vy

    # Временной интервал
    t = np.arange(0, t_max, dt)

    # Решение системы ОДУ
    solution = odeint(stone_motion, y0, t, args=(g, coeff, resistance_type))

    # Находим точку падения (y = 0)
    landing_index = None
    for i in range(1, len(solution)):
        if solution[i, 1] < 0:  # y становится отрицательным
            landing_index = i
            break

    if landing_index is not None:
        # Интерполяция для точного определения точки падения
        y_prev = solution[landing_index - 1, 1]
        y_curr = solution[landing_index, 1]
        t_prev = t[landing_index - 1]
        t_curr = t[landing_index]

        # Линейная интерполяция
        t_landing = t_prev - y_prev * (t_curr - t_prev) / (y_curr - y_prev)
        x_landing = np.interp(t_landing, t[:landing_index + 1], solution[:landing_index + 1, 0])
    else:
        t_landing = t_max
        x_landing = solution[-1, 0]

    return solution, t, x_landing, t_landing


def theoretical_no_resistance(angle, velocity, g=9.81):
    """
    Теоретическое решение для движения без сопротивления воздуха
    """
    vx0 = velocity * math.cos(angle)
    vy0 = velocity * math.sin(angle)

    # Время полета
    t_flight = 2 * vy0 / g

    # Дальность полета
    x_max = vx0 * t_flight

    # Максимальная высота
    y_max = (vy0 ** 2) / (2 * g)

    return x_max, y_max, t_flight


def plot_trajectory(solution, x_landing, angle, velocity, coeff, resistance_type):
    """
    Построение графика траектории
    """
    landing_index = np.argmax(solution[:, 1] < 0)
    if landing_index > 0:
        x_plot = solution[:landing_index, 0]
        y_plot = solution[:landing_index, 1]
    else:
        x_plot = solution[:, 0]
        y_plot = solution[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Численное решение')
    plt.plot(x_landing, 0, 'ro', markersize=8, label='Точка падения')

    # Теоретическая траектория для случая без сопротивления
    if coeff == 0:
        t_theor = np.linspace(0, 2 * velocity * math.sin(angle) / 9.81, 100)
        x_theor = velocity * math.cos(angle) * t_theor
        y_theor = velocity * math.sin(angle) * t_theor - 0.5 * 9.81 * t_theor ** 2
        plt.plot(x_theor, y_theor, 'g--', linewidth=1, label='Теоретическая траектория (без сопротивления)')

    plt.xlabel('Расстояние, м')
    plt.ylabel('Высота, м')
    plt.title(f'Траектория движения камня (угол: {math.degrees(angle):.1f}°, скорость: {velocity} м/с)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.show()


def analyze_parameters(angle, velocity, coeff, resistance_type):
    """
    Анализ влияния параметров на траекторию
    """
    print("\n" + "=" * 50)
    print("АНАЛИЗ ВЛИЯНИЯ ПАРАМЕТРОВ НА ТРАЕКТОРИЮ")
    print("=" * 50)

    # Анализ влияния угла
    print("\n1. Влияние угла броска:")
    angles = [math.radians(30), math.radians(45), math.radians(60)]
    for ang in angles:
        solution, t, x_landing, t_landing = calculate_trajectory(ang, velocity, coeff, resistance_type)
        max_height = np.max(solution[:, 1])
        print(f"Угол {math.degrees(ang):.0f}°: дальность = {x_landing:.2f} м, высота = {max_height:.2f} м")

    # Анализ влияния скорости
    print("\n2. Влияние начальной скорости:")
    velocities = [velocity / 2, velocity, velocity * 1.5]
    for vel in velocities:
        solution, t, x_landing, t_landing = calculate_trajectory(angle, vel, coeff, resistance_type)
        max_height = np.max(solution[:, 1])
        print(f"Скорость {vel} м/с: дальность = {x_landing:.2f} м, высота = {max_height:.2f} м")

    # Анализ влияния коэффициента сопротивления
    print("\n3. Влияние коэффициента сопротивления:")
    coeffs = [0, coeff / 2, coeff, coeff * 2]
    for c in coeffs:
        solution, t, x_landing, t_landing = calculate_trajectory(angle, velocity, c, resistance_type)
        max_height = np.max(solution[:, 1])
        resistance_name = "вязкое трение" if resistance_type == 1 else "лобовое сопротивление"
        print(f"Коэффициент {c} ({resistance_name}): дальность = {x_landing:.2f} м, высота = {max_height:.2f} м")


def get_motion_params():
    """
    Получение параметров движения от пользователя (дополненная версия)
    """
    try:
        ANGLE = ((int(input("Введите угол: ")) % 360 + 360) % 360) / 180 * math.pi
        VELOCITY = int(input("Введите начальную скорость: "))
        COEFF = float(input("Введите коэффициент сопротивления: "))
        RESISTANCE_TYPE = int(
            input("Введите модель сопротивления\n\t(1): вязкое трение\n\t(2): лобовое сопротивление\n"))

        if VELOCITY <= 0:
            raise ValueError("Скорость должна быть положительной")
        if COEFF < 0:
            raise ValueError("Коэффициент сопротивления не может быть отрицательным")
        if RESISTANCE_TYPE not in [1, 2]:
            raise ValueError("Неверный выбор модели сопротивления")

        return ANGLE, VELOCITY, COEFF, RESISTANCE_TYPE

    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        return get_motion_params()


def main():
    """
    Основная функция программы (дополненная версия)
    """
    # Получаем параметры от пользователя (используем вашу функцию)
    ANGLE, VELOCITY, COEFF, RESISTANCE_TYPE = get_motion_params()

    # Расчет траектории
    solution, t, x_landing, t_landing = calculate_trajectory(ANGLE, VELOCITY, COEFF, RESISTANCE_TYPE)

    # Вывод результатов
    max_height = np.max(solution[:, 1])

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ РАСЧЕТА")
    print("=" * 50)
    print(f"Угол броска: {math.degrees(ANGLE):.1f}°")
    print(f"Начальная скорость: {VELOCITY} м/с")
    print(f"Коэффициент сопротивления: {COEFF}")
    print(f"Модель сопротивления: {'Вязкое трение' if RESISTANCE_TYPE == 1 else 'Лобовое сопротивление'}")
    print(f"Максимальная высота: {max_height:.2f} м")
    print(f"Дальность полета: {x_landing:.2f} м")
    print(f"Время полета: {t_landing:.2f} с")

    # Сравнение с теоретическим решением (без сопротивления)
    if COEFF == 0:
        x_theor, y_theor, t_theor = theoretical_no_resistance(ANGLE, VELOCITY)
        print(f"\nТеоретическое решение (без сопротивления):")
        print(f"Дальность: {x_theor:.2f} м")
        print(f"Максимальная высота: {y_theor:.2f} м")
        print(f"Время полета: {t_theor:.2f} с")

        # Сравнение погрешности
        error_x = abs(x_landing - x_theor) / x_theor * 100
        error_y = abs(max_height - y_theor) / y_theor * 100
        error_t = abs(t_landing - t_theor) / t_theor * 100
        print(f"\nПогрешность численного решения:")
        print(f"По дальности: {error_x:.2f}%")
        print(f"По высоте: {error_y:.2f}%")
        print(f"По времени: {error_t:.2f}%")

    # Построение графика траектории
    plot_trajectory(solution, x_landing, ANGLE, VELOCITY, COEFF, RESISTANCE_TYPE)

    # Дополнительный анализ параметров
    analyze_parameters(ANGLE, VELOCITY, COEFF, RESISTANCE_TYPE)

    # Сравнение двух моделей сопротивления
    if COEFF > 0:
        print("\n" + "=" * 50)
        print("СРАВНЕНИЕ МОДЕЛЕЙ СОПРОТИВЛЕНИЯ")
        print("=" * 50)

        # Расчет для вязкого трения
        solution_viscous, t1, x_landing_viscous, t_landing_viscous = calculate_trajectory(ANGLE, VELOCITY, COEFF, 1)
        max_height_viscous = np.max(solution_viscous[:, 1])

        # Расчет для лобового сопротивления
        solution_quadratic, t2, x_landing_quadratic, t_landing_quadratic = calculate_trajectory(ANGLE, VELOCITY, COEFF,
                                                                                                2)
        max_height_quadratic = np.max(solution_quadratic[:, 1])

        print(f"Вязкое трение: дальность = {x_landing_viscous:.2f} м, высота = {max_height_viscous:.2f} м")
        print(f"Лобовое сопротивление: дальность = {x_landing_quadratic:.2f} м, высота = {max_height_quadratic:.2f} м")

        # Построение сравнительного графика
        plt.figure(figsize=(10, 6))

        # Траектория для вязкого трения
        landing_idx_viscous = np.argmax(solution_viscous[:, 1] < 0)
        if landing_idx_viscous > 0:
            x_viscous = solution_viscous[:landing_idx_viscous, 0]
            y_viscous = solution_viscous[:landing_idx_viscous, 1]
            plt.plot(x_viscous, y_viscous, 'b-', linewidth=2, label='Вязкое трение')

        # Траектория для лобового сопротивления
        landing_idx_quadratic = np.argmax(solution_quadratic[:, 1] < 0)
        if landing_idx_quadratic > 0:
            x_quadratic = solution_quadratic[:landing_idx_quadratic, 0]
            y_quadratic = solution_quadratic[:landing_idx_quadratic, 1]
            plt.plot(x_quadratic, y_quadratic, 'r-', linewidth=2, label='Лобовое сопротивление')

        # Траектория без сопротивления для сравнения
        if COEFF > 0:
            solution_no_resist, t3, x_landing_no_resist, t_landing_no_resist = calculate_trajectory(ANGLE, VELOCITY, 0,
                                                                                                    RESISTANCE_TYPE)
            landing_idx_no_resist = np.argmax(solution_no_resist[:, 1] < 0)
            if landing_idx_no_resist > 0:
                x_no_resist = solution_no_resist[:landing_idx_no_resist, 0]
                y_no_resist = solution_no_resist[:landing_idx_no_resist, 1]
                plt.plot(x_no_resist, y_no_resist, 'g--', linewidth=1, label='Без сопротивления')

        plt.xlabel('Расстояние, м')
        plt.ylabel('Высота, м')
        plt.title('Сравнение моделей сопротивления воздуха')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.show()


if __name__ == '__main__':
    main()