import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def stone_motion(params, g, k, m, resistance_type):
    """
    Система дифференциальных уравнений движения камня
    params[0] = x (координата по горизонтали)
    params[1] = y (координата по вертикали)
    params[2] = vx (скорость по горизонтали)
    params[3] = vy (скорость по вертикали)
    """
    x, y_pos, vx, vy = params
    v = math.sqrt(vx ** 2 + vy ** 2)

    # Сила сопротивления
    if resistance_type == 1:  # Вязкое трение
        Fx = -k * vx
        Fy = -k * vy
    else:  # Лобовое сопротивление
        if v > 0:
            Fx = -k * v * vx
            Fy = -k * v * vy
        else:
            Fx, Fy = 0, 0

    # Уравнения движения: F = ma => a = F/m
    # По горизонтали: только сила сопротвиления
    # По вертикали: сила тяжести + сила сопротивления
    dxdt = vx
    dydt = vy
    dvxdt = Fx / m
    dvydt = -g + Fy / m

    return [dxdt, dydt, dvxdt, dvydt]


def calculate_trajectory(angle, velocity, coeff, mass, resistance_type, g=9.81, t_max=20, dt=0.01):
    """
    Расчет траектории движения камня
    """
    # Начальные условия
    vx0 = velocity * math.cos(angle)
    vy0 = velocity * math.sin(angle)
    params = [0, 0, vx0, vy0]  # x, y, vx, vy

    # Временной интервал
    t = np.arange(0, t_max, dt)

    # Решение системы ОДУ
    solution = odeint(stone_motion, params, args=(g, coeff, mass, resistance_type))

    # Находим точку падения (y = 0)
    landing_index = None
    for i in range(1, len(solution)):
        if solution[i, 1] < 0:
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
    Решение для движения без сопротивления воздуха
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


def plot_trajectory(solution, x_landing, angle, velocity, coeff, mass, resistance_type):
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
    t_theor = np.linspace(0, 2 * velocity * math.sin(angle) / 9.81, 100)
    x_theor = velocity * math.cos(angle) * t_theor
    y_theor = velocity * math.sin(angle) * t_theor - 0.5 * 9.81 * t_theor ** 2
    plt.plot(x_theor, y_theor, 'g--', linewidth=1, label='Теоретическая траектория (без сопротивления)')

    plt.xlabel('Расстояние, м')
    plt.ylabel('Высота, м')

    resistance_name = 'вязкое трение' if resistance_type == 1 else 'лобовое сопротивление'
    title = f'Траектория движения камня\n'
    title += f'Угол: {math.degrees(angle):.1f}°, Скорость: {velocity} м/с, Масса: {mass} кг\n'
    title += f'Сопротивление: {resistance_name} (k={coeff})'

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.show()


def get_motion_params():
    """
    Получение параметров движения от пользователя
    """
    try:
        print('=' * 50)
        print('ПАРАМЕТРЫ ДВИЖЕНИЯ КАМНЯ')
        print('=' * 50)

        angle_deg = int(input('Введите угол броска (градусы): '))
        ANGLE = ((angle_deg % 360 + 360) % 360) / 180 * math.pi

        VELOCITY = float(input('Введите начальную скорость (м/с): '))
        COEFF = float(input('Введите коэффициент сопротивления: '))
        MASS = float(input('Введите массу камня (кг): '))

        print('1 - Вязкое трение (F = -k*v)')
        print('2 - Лобовое сопротивление (F = -k*v²)')
        RESISTANCE_TYPE = int(input('\t(1) - Вязкое трение (F = -k*v)\n'
                                    '\t(2)'
                                    'Выберите модель сопротивления (1 или 2): '))

        if VELOCITY <= 0:
            raise ValueError('Скорость должна быть положительной')
        if COEFF < 0:
            raise ValueError('Коэффициент сопротивления не может быть отрицательным')
        if MASS <= 0:
            raise ValueError('Масса должна быть положительной')
        if RESISTANCE_TYPE not in [1, 2]:
            raise ValueError('Неверный выбор модели сопротивления')

        return ANGLE, VELOCITY, COEFF, MASS, RESISTANCE_TYPE

    except ValueError as e:
        print(f'Ошибка ввода: {e}')
        return get_motion_params()


def main():
    """
    Основная функция программы
    """
    # Получаем параметры от пользователя
    ANGLE, VELOCITY, COEFF, MASS, RESISTANCE_TYPE = get_motion_params()

    # Расчет траектории
    solution, t, x_landing, t_landing = calculate_trajectory(ANGLE, VELOCITY, COEFF, MASS, RESISTANCE_TYPE)

    # Вывод результатов
    max_height = np.max(solution[:, 1])
    max_height_time = t[np.argmax(solution[:, 1])]

    print('\n' + '=' * 50)
    print('РЕЗУЛЬТАТЫ РАСЧЕТА')
    print('=' * 50)
    print(f'Угол броска: {math.degrees(ANGLE):.1f}°'
          f'Начальная скорость: {VELOCITY} м/с'
          f'Коэффициент сопротивления: {COEFF}'
          f'Масса камня: {MASS} кг'
          f'Модель сопротивления: {'Вязкое трение (F = -k*v)' if RESISTANCE_TYPE == 1 else 'Лобовое сопротивление (F = -k*v²)'}'
          f'Максимальная высота: {max_height:.2f} м (время: {max_height_time:.2f} с)'
          f'Дальность полета: {x_landing:.2f} м'
          f'Время полета: {t_landing:.2f} с')

    # Сравнение с решением без сопротивления
    x_theor, y_theor, t_theor = theoretical_no_resistance(ANGLE, VELOCITY)
    print(f'\nТеоретическое решение (без сопротивления):'
          f'Дальность: {x_theor:.2f} м'
          f'Максимальная высота: {y_theor:.2f} м'
          f'Время полета: {t_theor:.2f} с')

    # Сравнение влияния сопротивления
    if COEFF > 0:
        reduction_x = (x_theor - x_landing) / x_theor * 100
        reduction_y = (y_theor - max_height) / y_theor * 100
        reduction_t = (t_theor - t_landing) / t_theor * 100

        print(f'\nВлияние сопротивления воздуха:')
        print(f'Уменьшение дальности: {reduction_x:.1f}%'
              f'Уменьшение высоты: {reduction_y:.1f}%'
              f'Уменьшение времени полета: {reduction_t:.1f}%')

    # Построение графика траектории
    plot_trajectory(solution, x_landing, ANGLE, VELOCITY, COEFF, MASS, RESISTANCE_TYPE)


if __name__ == '__main__':
    main()
