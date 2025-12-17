import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_param(prompt, default_value, value_type=float):
    """
    Запрашивает у пользователя параметр.
    Если ввод пустой (Enter), возвращает default_value.
    Иначе пытается преобразовать ввод в value_type.
    """
    user_input = input(f"{prompt} [по умолчанию {default_value}]: ").strip()
    if not user_input:
        return default_value
    try:
        return value_type(user_input)
    except ValueError:
        print(f"Ошибка: некорректное число. Используется значение по умолчанию: {default_value}")
        return default_value


# === ПАРАМЕТРЫ СЕТКИ И ВРЕМЕНИ (глобальные для простоты) ===
a = 0.1   # Расстояние между грузами (для визуализации и теории)
dt = 0.005  # Шаг по времени


class ChainSimulation:
    def __init__(self, N, m, k, gamma, mode):
        self.N = N
        self.m = m
        self.k = k
        self.gamma = gamma
        self.mode = mode

        self.x = np.zeros(N)  # Смещения
        self.v = np.zeros(N)  # Скорости
        self.t = 0.0

        # Теоретическая скорость звука
        self.c_theory = a * np.sqrt(k / m)
        print(f"Расчётная скорость волны: {self.c_theory:.4f} м/с")

        # Частота для резонанса (3-я мода):
        # ω_n = 2 * sqrt(k/m) * sin(n * π / (2*(N+1)))
        self.drive_freq = 2 * np.sqrt(k / m) * np.sin(3 * np.pi / (2 * (N + 1)))
        if self.mode == 'standing':
            print(f"Частота возбуждения (3-я мода): {self.drive_freq:.4f} рад/с")

    def apply_boundary_conditions(self, t):
        if self.mode == 'pulse':
            # Левый край: плавный импульс
            if t < 0.5:
                self.x[0] = 1.0 * np.exp(-100 * (t - 0.25) ** 2)
            else:
                self.x[0] = 0.0
            # Правый край условно "фиксируем"
            # self.x[-1] = 0.0

        elif self.mode == 'standing':
            # Оба края закреплены
            self.x[0] = 0.0
            self.x[-1] = 0.0

    def get_forces(self, x, v, t):
        F = np.zeros(self.N)

        # Упругие силы: k * (x_{i+1} - 2x_i + x_{i-1})
        F[1:-1] = self.k * (x[2:] - 2 * x[1:-1] + x[:-2])

        # 2. Правый край (свободный конец):
        # сила только от левой пружины
        F[-1] = -self.k * (x[-1] - x[-2])

        # 3. Левый край:
        # сила от правой пружины
        F[0] = self.k * (x[1] - x[0])


        # Вязкое трение
        F -= self.gamma * v

        # Вынуждающая сила (только для режима стоячих волн)
        if self.mode == 'standing':
            drive_idx = max(1, int(self.N * 0.1))
            force_amp = 50.0
            F[drive_idx] += force_amp * np.cos(self.drive_freq * t)

        return F

    def step(self):
        # Velocity Verlet
        F = self.get_forces(self.x, self.v, self.t)
        self.v += 0.5 * (F / self.m) * dt

        self.x += self.v * dt
        self.t += dt

        self.apply_boundary_conditions(self.t)

        F_new = self.get_forces(self.x, self.v, self.t)
        self.v += 0.5 * (F_new / self.m) * dt


# === ТЕСТЫ ===

def measure_wave_speed(sim: ChainSimulation, probe_i1=None, probe_i2=None,
                       t_max=5.0):
    """
    Запускает режим бегущей волны и по времени прихода максимума
    на двух зондовых массах оценивает численную скорость волны.
    """
    if sim.mode != 'pulse':
        raise ValueError("measure_wave_speed: sim.mode должен быть 'pulse'")

    if probe_i1 is None:
        probe_i1 = int(sim.N * 0.2)
    if probe_i2 is None:
        probe_i2 = int(sim.N * 0.5)

    t_hist = []
    x1_hist = []
    x2_hist = []

    while sim.t < t_max:
        sim.step()
        t_hist.append(sim.t)
        x1_hist.append(sim.x[probe_i1])
        x2_hist.append(sim.x[probe_i2])

    t_hist = np.array(t_hist)
    x1_hist = np.array(x1_hist)
    x2_hist = np.array(x2_hist)

    # индексы максимумов по модулю
    idx1 = np.argmax(np.abs(x1_hist))
    idx2 = np.argmax(np.abs(x2_hist))
    t1 = t_hist[idx1]
    t2 = t_hist[idx2]

    dx = (probe_i2 - probe_i1) * a
    c_num = dx / (t2 - t1)

    return c_num, (t1, t2)


def measure_resonance_frequency(sim: ChainSimulation,
                                drive_idx=None,
                                t_relax=50.0,
                                t_obs=50.0):
    """
    Запускает режим стоячей волны, собирает временной ряд смещения
    выбранного узла и по FFT оценивает основную частоту колебаний.
    """
    if sim.mode != 'standing':
        raise ValueError("measure_resonance_frequency: sim.mode должен быть 'standing'")

    if drive_idx is None:
        drive_idx = max(1, int(sim.N * 0.1))

    # Разогрев до установившегося режима
    while sim.t < t_relax:
        sim.step()

    xs = []
    ts = []
    t_start = sim.t
    while sim.t < t_start + t_obs:
        sim.step()
        xs.append(sim.x[drive_idx])
        ts.append(sim.t)

    xs = np.array(xs)
    ts = np.array(ts)

    # шаг по времени в выборке
    dt_samp = np.mean(np.diff(ts))

    # спектр
    xs_centered = xs - xs.mean()
    Xf = np.fft.rfft(xs_centered)
    freqs = np.fft.rfftfreq(len(xs_centered), d=dt_samp)

    # доминирующая частота (кроме нуля)
    idx = np.argmax(np.abs(Xf[1:])) + 1
    f_dom = freqs[idx]
    omega_dom = 2 * np.pi * f_dom

    return omega_dom


if __name__ == "__main__":
    # === БЛОК НАСТРОЙКИ ===
    print("=== Настройка параметров модели ===")
    print("Нажмите Enter, чтобы оставить значение по умолчанию.\n")

    N = get_param("Число грузов (N)", 100, int)
    m = get_param("Масса груза (m, кг)", 1.0)
    k = get_param("Жесткость пружины (k, Н/м)", 1000.0)
    gamma = get_param("Коэффициент трения (gamma)", 0.05)

    # --- Тест 1: скорость бегущей волны ---
    print("\n=== Тест 1: проверка скорости волны ===")
    sim_test1 = ChainSimulation(N, m, k, gamma, mode='pulse')
    c_theory = sim_test1.c_theory
    c_num, (t1, t2) = measure_wave_speed(sim_test1, t_max=5.0)

    rel_err_c = abs(c_num - c_theory) / c_theory
    print(f"Теоретическая скорость:  {c_theory:.4f}")
    print(f"Численная скорость:     {c_num:.4f}")
    print(f"Времена прихода пика:   t1={t1:.4f}, t2={t2:.4f}")
    print(f"Относительная ошибка:   {rel_err_c:.2%}")

    # --- Тест 2: частота стоячей волны (3-я мода) ---
    print("\n=== Тест 2: проверка частоты стоячей волны (3-я мода) ===")
    sim_test2 = ChainSimulation(N, m, k, gamma, mode='standing')
    omega_theory = sim_test2.drive_freq
    omega_num = measure_resonance_frequency(sim_test2, t_relax=50.0, t_obs=50.0)
    rel_err_w = abs(omega_num - omega_theory) / omega_theory

    print(f"Теоретическая ω_3:      {omega_theory:.4f}")
    print(f"Численная доминир. ω:   {omega_num:.4f}")
    print(f"Относительная ошибка:   {rel_err_w:.2%}")

    # --- Выбор режима визуализации ---
    print("\nВыберите режим визуализации:")
    print("1 - Бегущая волна (одиночный импульс)")
    print("2 - Стоячая волна (вынужденные колебания/резонанс)")
    mode_choice = input("Ваш выбор [1]: ").strip()
    mode = 'standing' if mode_choice == '2' else 'pulse'
    print(f"\nЗапуск симуляции: {mode}...\n")

    # Инициализация симуляции с выбранным режимом
    sim = ChainSimulation(N, m, k, gamma, mode)

    # === ВИЗУАЛИЗАЦИЯ ===
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes(xlim=(0, N), ylim=(-2.5, 2.5))
    line, = ax.plot([], [], 'o-', lw=1, markersize=3 if N > 50 else 6)
    title = ax.set_title('')
    ax.set_xlabel('Номер груза')
    ax.set_ylabel('Смещение')
    ax.grid(True, alpha=0.3)

    param_str = f"N={N}, m={m}, k={k}, γ={gamma}"
    ax.text(0.02, 0.95, param_str, transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8))

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        for _ in range(10):
            sim.step()

        line.set_data(np.arange(N), sim.x)
        mode_desc = "Бегущая волна" if sim.mode == 'pulse' else "Стоячая волна"
        title.set_text(f'{mode_desc}, t={sim.t:.2f} с')
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
    print("Окно с графиком открыто.")
    plt.show()
