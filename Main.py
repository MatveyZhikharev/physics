import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve
import matplotlib.pyplot as plt

EPS0 = 8.8541878128e-12
K = 1.0 / (4.0 * np.pi * EPS0)


# ---------- Ввод с клавиатуры ----------
def ask_float(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return float(s) if s else float(default)

def ask_int(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return int(s) if s else int(default)

def ask_vec3(prompt, default_xyz):
    s = input(
        f"{prompt} (x y z) [{default_xyz[0]} {default_xyz[1]} {default_xyz[2]}]: "
    ).strip()
    if not s:
        return tuple(float(v) for v in default_xyz)
    parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("Нужно ввести ровно 3 числа: x y z")
    return tuple(float(v) for v in parts)


# ---------- Геометрия / дискретизация ----------
def fibonacci_sphere_points(n):
    """Квазиравномерные точки на единичной сфере."""
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = (np.pi * (3.0 - np.sqrt(5.0))) * i  # golden angle
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def build_two_spheres(R1=0.06, c1=(0.0, 0.0, 0.0), N1=500,
                      R2=0.06, c2=(0.18, 0.0, 0.0), N2=500):
    u1 = fibonacci_sphere_points(N1)
    u2 = fibonacci_sphere_points(N2)

    c1 = np.array(c1, dtype=float)
    c2 = np.array(c2, dtype=float)

    p1 = c1[None, :] + R1 * u1
    p2 = c2[None, :] + R2 * u2

    a1 = 4.0 * np.pi * R1**2 / N1
    a2 = 4.0 * np.pi * R2**2 / N2

    pos = np.vstack([p1, p2])  # (M,3)
    area = np.hstack([np.full(N1, a1), np.full(N2, a2)])
    eid = np.hstack([np.zeros(N1, dtype=int), np.ones(N2, dtype=int)])  # 0 или 1

    s1 = (R1, c1)
    s2 = (R2, c2)
    return pos, area, eid, s1, s2


# ---------- Метод моментов ----------
def assemble_A(pos, area):
    """
    A_ij = K / r_ij (i!=j)
    A_ii: приближение самовлияния через эквивалентный диск площади area[i]:
          A_ii ~ K * 2 / a_eff, a_eff = sqrt(area/pi)
    """
    d = pos[:, None, :] - pos[None, :, :]
    rij = norm(d, axis=2)

    A = K / np.where(rij > 0, rij, 1.0)  # диагональ временно не важна
    np.fill_diagonal(A, 0.0)

    a_eff = np.sqrt(area / np.pi)
    np.fill_diagonal(A, K * 2.0 / a_eff)
    return A

def solve_charges(pos, area, eid, V=1000.0):
    A = assemble_A(pos, area)
    b = np.where(eid == 0, +V / 2.0, -V / 2.0)
    q = solve(A, b)
    return q

def total_charge(q, eid, electrode_id):
    return q[eid == electrode_id].sum()


# ---------- Поля и потенциал ----------
def field_at_points(r_pts, src_pos, src_q):
    """
    E(r) = K * sum q_j (r-rj)/|r-rj|^3
    r_pts: (P,3)
    """
    P = r_pts.shape[0]
    E = np.zeros((P, 3), dtype=float)

    chunk = 400
    for s in range(0, src_pos.shape[0], chunk):
        rp = src_pos[s:s + chunk]   # (C,3)
        rq = src_q[s:s + chunk]     # (C,)

        dr = r_pts[:, None, :] - rp[None, :, :]   # (P,C,3)
        r = norm(dr, axis=2)                      # (P,C)
        r = np.maximum(r, 1e-6)

        coeff = rq[None, :] / (r**3)              # (P,C)
        E += K * (dr * coeff[:, :, None]).sum(axis=1)

    return E

def potential_at_points(r_pts, src_pos, src_q):
    """phi(r) = K * sum q_j / |r-rj|"""
    P = r_pts.shape[0]
    phi = np.zeros(P, dtype=float)

    chunk = 400
    for s in range(0, src_pos.shape[0], chunk):
        rp = src_pos[s:s + chunk]
        rq = src_q[s:s + chunk]

        dr = r_pts[:, None, :] - rp[None, :, :]
        r = norm(dr, axis=2)
        r = np.maximum(r, 1e-6)

        phi += K * (rq[None, :] / r).sum(axis=1)

    return phi

def inside_any_sphere(xyz, spheres):
    mask = np.zeros(xyz.shape[0], dtype=bool)
    for R, c in spheres:
        mask |= (norm(xyz - c[None, :], axis=1) <= R * 1.001)
    return mask


def main():
    print("Метод моментов (электростатика): 2 сферы")
    print("Подсказка: для вложенных сфер сделайте центры одинаковыми.\n")

    V = ask_float("Разность потенциалов V, В", 1000.0)

    R1 = ask_float("R1, м", 0.06)
    c1 = ask_vec3("Центр сферы 1", (0.0, 0.0, 0.0))
    N1 = ask_int("Число элементов N1", 500)

    R2 = ask_float("R2, м", 0.06)
    c2 = ask_vec3("Центр сферы 2", (0.18, 0.0, 0.0))
    N2 = ask_int("Число элементов N2", 500)

    x_min = ask_float("x_min, м", -0.15)
    x_max = ask_float("x_max, м", 0.33)
    z_min = ask_float("z_min, м", -0.22)
    z_max = ask_float("z_max, м", 0.22)
    n_grid = ask_int("Размер сетки n_grid (например 220)", 220)

    pos, area, eid, s1, s2 = build_two_spheres(R1=R1, c1=c1, N1=N1, R2=R2, c2=c2, N2=N2)
    q = solve_charges(pos, area, eid, V=V)

    Q1 = total_charge(q, eid, 0)
    Q2 = total_charge(q, eid, 1)
    C = abs(Q1) / V

    print("\n--- Результаты ---")
    print(f"Q1 = {Q1:.6e} Кл")
    print(f"Q2 = {Q2:.6e} Кл")
    print(f"Q1 + Q2 = {(Q1 + Q2):.6e} Кл")
    print(f"C ≈ {C:.6e} Ф")

    # Сечение y=0: строим φ(x,z) и линии поля
    x = np.linspace(x_min, x_max, n_grid)
    z = np.linspace(z_min, z_max, n_grid)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)

    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    mask_in = inside_any_sphere(pts, spheres=[s1, s2])

    pts_eval = pts.copy()
    pts_eval[mask_in] = np.array([1e9, 1e9, 1e9])  # чтобы не считать поле внутри проводника

    E = field_at_points(pts_eval, pos, q)
    Ex = E[:, 0].reshape(X.shape)
    Ez = E[:, 2].reshape(X.shape)

    phi = potential_at_points(pts_eval, pos, q).reshape(X.shape)

    Ex = np.where(mask_in.reshape(X.shape), np.nan, Ex)
    Ez = np.where(mask_in.reshape(X.shape), np.nan, Ez)
    phi = np.where(mask_in.reshape(X.shape), np.nan, phi)

    plt.figure(figsize=(9, 5))
    plt.contourf(X, Z, phi, levels=40, cmap="coolwarm")
    plt.colorbar(label="Potential φ, V")

    plt.gca().set_aspect('equal', 'box') # это для правильного масштаба
    plt.streamplot(X, Z, Ex, Ez, density=1.6, color="k", linewidth=0.7)

    for (R, c) in [s1, s2]:
        t = np.linspace(0, 2 * np.pi, 400)
        plt.plot(c[0] + R * np.cos(t), c[2] + R * np.sin(t), "k", lw=2)

    plt.title("Equipotentials and field lines (slice y=0)")
    plt.xlabel("x, m")
    plt.ylabel("z, m")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
