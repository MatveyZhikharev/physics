import numpy as np
import matplotlib.pyplot as plt

# --- 1. КОНСТАНТЫ ---
G = 6.67430e-11
M_SUN = 1.989e30
M_EARTH = 5.972e24
R_EARTH = 6.371e6
M_MARS = 6.417e23
R_MARS = 3.3895e6

ATMOSPHERE = {
    'Earth': {'rho0': 1.225, 'H': 8500.0},
    'Mars':  {'rho0': 0.020, 'H': 11100.0}
}

# --- 2. КЛАСС КОРАБЛЯ ---
class Spacecraft:
    def __init__(self, dry_mass, fuel_mass, thrust, isp, length, radius, cd):
        # Параметры массы и двигателя
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass
        self.thrust_max = thrust
        self.isp = isp
        
        # Геометрия (для задания C)
        self.length = length
        self.radius = radius
        self.cd = cd
        self.area_frontal = np.pi * radius**2
        self.area_side = length * 2 * radius
        
        # Кинематика (Линейная)
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])
        
        # Кинематика (Угловая - для задания C)
        self.angle = np.pi / 2 # Начальный угол (90 град, носом вверх)
        self.omega = 0.0
        self.rcs_torque = 10000.0 # Н*м (мощность маневровых двигателей)
        
        # История для графиков
        self.history = {'t': [], 'h': [], 'v': [], 'angle': [], 'stage': []}
        self.total_time = 0.0

    @property
    def total_mass(self):
        return self.dry_mass + self.fuel_mass

    @property
    def moment_of_inertia(self):
        # I = 1/12 * m * (3r^2 + L^2) + m*(L/2)^2 (если вращение вокруг центра)
        # Упрощенно для цилиндра:
        return (1/12) * self.total_mass * (3*self.radius**2 + self.length**2)

    def log(self, stage_name, altitude):
        self.history['t'].append(self.total_time)
        self.history['h'].append(altitude)
        self.history['v'].append(np.linalg.norm(self.vel))
        self.history['angle'].append(self.angle)
        self.history['stage'].append(stage_name)

    # Функция управления вращением (PD-регулятор)
    def update_orientation(self, target_angle, dt):
        error = target_angle - self.angle
        # Нормализация угла к [-pi, pi]
        error = (error + np.pi) % (2 * np.pi) - np.pi
        
        kp = 50000.0  # Жесткость
        kd = 200000.0 # Демпфирование
        
        torque = kp * error - kd * self.omega
        torque = np.clip(torque, -self.rcs_torque, self.rcs_torque)
        
        alpha = torque / self.moment_of_inertia
        self.omega += alpha * dt
        self.angle += self.omega * dt
        
        # Возвращаем True, если стабилизировались
        return abs(error) < 0.05

# --- 3. ФИЗИКА ---

def get_rho(h, body):
    if h < 0: return ATMOSPHERE[body]['rho0']
    p = ATMOSPHERE[body]
    return p['rho0'] * np.exp(-h / p['H'])

def get_gravity(mass, pos, M_body):
    r = np.linalg.norm(pos)
    if r == 0: return np.array([0.0, 0.0])
    return -G * M_body * mass / r**3 * pos

def get_drag_simple(ship, rho):
    """Для этапа 1: считаем, что ракета летит носом вперед идеально"""
    v_mag = np.linalg.norm(ship.vel)
    if v_mag == 0: return np.array([0.0, 0.0])
    f = 0.5 * ship.cd * rho * ship.area_frontal * v_mag**2
    return -f * (ship.vel / v_mag)

def get_drag_advanced(ship, rho):
    """Для этапа 3: зависит от угла атаки"""
    v_mag = np.linalg.norm(ship.vel)
    if v_mag == 0: return np.array([0.0, 0.0])
    
    # Угол скорости
    v_ang = np.arctan2(ship.vel[1], ship.vel[0])
    # Угол атаки
    alpha = ship.angle - v_ang
    
    # Проекция площади
    area_eff = ship.area_frontal * abs(np.cos(alpha)) + ship.area_side * abs(np.sin(alpha))
    
    f = 0.5 * ship.cd * rho * area_eff * v_mag**2
    return -f * (ship.vel / v_mag)


# --- 4. ЭТАПЫ ПОЛЁТА ---

def stage_1_launch(ship):
    print("--- ЗАПУСК: ЭТАП 1 (Земля) ---")
    dt = 0.1
    t = 0
    
    ship.pos = np.array([0.0, R_EARTH]) # Старт с полюса (условно)
    ship.angle = np.pi / 2 # Носом вверх
    
    target_orbit = 200000 # 200 км
    
    while ship.fuel_mass > 0:
        r = np.linalg.norm(ship.pos)
        alt = r - R_EARTH
        v_mag = np.linalg.norm(ship.vel)
        
        # Силы
        fg = get_gravity(ship.total_mass, ship.pos, M_EARTH)
        fd = get_drag_simple(ship, get_rho(alt, 'Earth')) # Пункт A: Атмосфера
        
        # Управление вектором тяги (Gravity Turn)
        # Чем выше, тем сильнее наклоняем угол
        target_angle = np.pi/2 - (np.pi/2) * (alt / 150000) # Плавно кладем на горизонт
        if target_angle < 0: target_angle = 0 # Горизонтально
        
        # Мгновенный поворот для упрощения на взлете (предполагаем идеальные рули)
        ship.angle = target_angle 
        
        thrust_vec = np.array([np.cos(ship.angle), np.sin(ship.angle)])
        ft = ship.thrust_max * thrust_vec
        
        # Интеграция
        acc = (fg + fd + ft) / ship.total_mass
        ship.vel += acc * dt
        ship.pos += ship.vel * dt
        
        # Расход
        dm = ship.thrust_max / (ship.isp * 9.81) * dt
        ship.fuel_mass -= dm
        
        t += dt
        ship.total_time += dt
        if int(t) % 10 == 0: ship.log('Launch', alt)
        
        # Условие выхода на орбиту (грубое)
        if alt > target_orbit and v_mag > 7500:
            print(f"Орбита достигнута! H={alt/1000:.1f}км, V={v_mag:.1f} м/с")
            break
            
    return ship

def stage_2_transfer(ship):
    print("--- ПЕРЕЛЁТ: ЭТАП 2 (Гелиоцентрический) ---")
    # Смена координат: Переходим в систему Солнца
    # r1 (Земля) -> r2 (Марс)
    r1 = 149.6e9
    r2 = 227.9e9
    
    ship.pos = np.array([r1, 0.0])
    
    # Считаем Delta-V
    v_start = np.sqrt(G * M_SUN / r1) # Скорость Земли
    v_needed = v_start * np.sqrt(2*r2 / (r1+r2)) # Скорость для эллипса
    
    print(f"Приращение скорости (Delta-V): {v_needed - v_start:.1f} м/с")
    
    # Применяем импульс
    ship.vel = np.array([0.0, v_needed])
    
    dt = 3600 * 12 # Шаг полдня
    t = 0
    
    while True:
        r = np.linalg.norm(ship.pos)
        
        # Только гравитация Солнца
        fg = get_gravity(ship.total_mass, ship.pos, M_SUN)
        
        acc = fg / ship.total_mass
        ship.vel += acc * dt
        ship.pos += ship.vel * dt
        
        t += dt
        ship.total_time += dt
        
        # Логируем реже
        if int(t) % (3600*24*5) == 0: 
            ship.log('Transfer', r) # Тут высота - это расстояние от Солнца
            
        if r >= r2:
            print(f"Прибытие к Марсу через {t/(3600*24):.1f} дней")
            break
    
    return ship

def stage_3_landing(ship):
    print("--- ПОСАДКА: ЭТАП 3 (С разворотом) ---")
    
    # Пересборка корабля для посадки (сбрасываем ступени, меняем конфиг)
    # Остается посадочный модуль
    ship.dry_mass = 2000
    ship.fuel_mass = 1500
    ship.thrust_max = 50000 
    
    # Переход в систему Марса
    ship.pos = np.array([0.0, R_MARS + 100000]) # 100 км
    ship.vel = np.array([3500.0, -200.0]) # Орбитальная скорость + снижение
    ship.angle = np.arctan2(ship.vel[1], ship.vel[0]) # Летим носом по курсу
    
    dt = 0.05
    t = 0
    landed = False
    
    while not landed:
        r = np.linalg.norm(ship.pos)
        alt = r - R_MARS
        v_mag = np.linalg.norm(ship.vel)
        
        # 1. Логика ориентации (Задание C)
        # Если высоко - летим боком (тормозим корпусом)
        # Если низко - разворачиваемся двигателем вперед
        if alt > 20000:
            v_ang = np.arctan2(ship.vel[1], ship.vel[0])
            target_angle = v_ang + np.pi/2 # Боком
        else:
            target_angle = np.arctan2(-ship.vel[1], -ship.vel[0]) # Двигателем вперед
            
        is_stable = ship.update_orientation(target_angle, dt)
        
        # 2. Силы
        fg = get_gravity(ship.total_mass, ship.pos, M_MARS)
        fd = get_drag_advanced(ship, get_rho(alt, 'Mars')) # С учетом угла атаки
        
        ft = np.array([0.0, 0.0])
        
        # 3. Двигатель (Suicide Burn)
        if alt < 8000 and ship.fuel_mass > 0:
            # Включаем только если стабилизировались
            if is_stable: 
                throttle = 1.0
                if v_mag < 50: throttle = 0.6
                
                thrust_vec = np.array([np.cos(ship.angle), np.sin(ship.angle)])
                ft = throttle * ship.thrust_max * thrust_vec
                
                dm = (np.linalg.norm(ft) / (ship.isp * 9.81)) * dt
                ship.fuel_mass -= dm
        
        # Интеграция
        acc = (fg + fd + ft) / ship.total_mass
        ship.vel += acc * dt
        ship.pos += ship.vel * dt
        
        t += dt
        ship.total_time += dt
        if int(t/dt) % 20 == 0: ship.log('Landing', alt)
        
        if alt <= 0:
            print(f"КАСАНИЕ. Скорость: {v_mag:.2f} м/с. Топливо: {ship.fuel_mass:.1f} кг")
            landed = True
            
    return ship

# --- 5. ЗАПУСК И ГРАФИКИ ---

def main():
    # Создаем ракету (Длина 30м, Радиус 2м)
    rocket = Spacecraft(dry_mass=5000, fuel_mass=60000, thrust=900000, isp=320, 
                        length=30, radius=2, cd=0.6)
    
    # Цепочка выполнения
    rocket = stage_1_launch(rocket)
    rocket = stage_2_transfer(rocket)
    rocket = stage_3_landing(rocket)
    
    # Визуализация
    hist = rocket.history
    
    # Разделяем данные по этапам для красивых графиков
    t = np.array(hist['t'])
    h = np.array(hist['h'])
    v = np.array(hist['v'])
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. Высота (Launch & Landing) - исключаем трансфер, так как там масштаб другой
    mask_launch = [s == 'Launch' for s in hist['stage']]
    mask_land = [s == 'Landing' for s in hist['stage']]
    
    axs[0].plot(t[mask_launch], h[mask_launch]/1000, 'b', label='Взлёт (Земля)')
    axs[0].set_ylabel('Высота (км)')
    axs[0].set_title('Этап 1: Выход на орбиту')
    axs[0].grid(True)
    axs[0].legend()
    
    # 2. Посадка (детально)
    t_land = t[mask_land] - t[mask_launch][-1] - (t[mask_land][0] - t[mask_launch][-1]) # сброс времени для графика
    axs[1].plot(t_land, h[mask_land]/1000, 'r', label='Высота')
    axs[1].plot(t_land, v[mask_land]/10, 'g--', label='Скорость (x0.1)')
    axs[1].set_title('Этап 3: Посадка на Марс')
    axs[1].set_ylabel('Высота (км)')
    axs[1].set_xlabel('Время посадки (с)')
    axs[1].legend()
    axs[1].grid(True)
    
    # 3. Угол ориентации при посадке
    angles = np.array(hist['angle'])[mask_land]
    axs[2].plot(t_land, np.degrees(angles), 'purple')
    axs[2].set_title('Ориентация корабля (Пункт C)')
    axs[2].set_ylabel('Угол (градусы)')
    axs[2].set_xlabel('Время посадки (с)')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
