import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

# ─────────────────────────────────────────────
#  Physics helpers
# ─────────────────────────────────────────────

def thin_lens_refract(y, angle_in, f):
    return angle_in - y / f

def intersect_sphere(ray_origin, ray_dir, center, R):
    ox, oz = ray_origin[0] - center[0], ray_origin[1] - center[1]
    dx, dz = ray_dir
    a = dx*dx + dz*dz
    b = 2*(ox*dx + oz*dz)
    c = ox*ox + oz*oz - R*R
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2*a)
    t2 = (-b + sq) / (2*a)
    candidates = [t for t in [t1, t2] if t > 1e-9]
    return min(candidates) if candidates else None

def refract_at_sphere(pos, direction, center, n1, n2):
    normal = np.array(pos) - np.array(center)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return None
    normal = normal / norm_len
    if np.dot(normal, direction) > 0:
        normal = -normal
    cos_i = -np.dot(normal, direction)
    sin_i2 = max(0.0, 1.0 - cos_i**2)
    sin_t2 = (n1/n2)**2 * sin_i2
    if sin_t2 > 1.0:
        return None
    cos_t = np.sqrt(1.0 - sin_t2)
    refracted = (n1/n2)*np.array(direction) + (n1/n2*cos_i - cos_t)*normal
    nlen = np.linalg.norm(refracted)
    return refracted / nlen if nlen > 1e-12 else None

# ─────────────────────────────────────────────
#  Thin-lens ray trace
# ─────────────────────────────────────────────

def trace_thin_lens_system(obj_height, obj_dist, f1, f2, tube_length, n_rays=9):
    z_obj = 0.0
    z_L1  = obj_dist
    z_L2  = z_L1 + tube_length

    object_points = [obj_height, 0.0, -obj_height]
    colors = ['red', 'green', 'blue']
    all_paths = []
    rays_after_L2 = {op: [] for op in object_points}

    for op, col in zip(object_points, colors):
        angles = np.linspace(-0.25, 0.25, n_rays)
        for angle in angles:
            path = [(z_obj, op)]
            y = op
            m = np.tan(angle)

            dz = z_L1 - z_obj
            y1 = y + m * dz
            path.append((z_L1, y1))
            m = thin_lens_refract(y1, m, f1)

            dz = z_L2 - z_L1
            y2 = y1 + m * dz
            path.append((z_L2, y2))
            m = thin_lens_refract(y2, m, f2)
            rays_after_L2[op].append((y2, m))

            dz_extra = abs(f2) * 3
            y_end = y2 + m * dz_extra
            path.append((z_L2 + dz_extra, y_end))
            all_paths.append((path, col, op))

    try:
        v1 = 1.0 / (1.0 / f1 - 1.0 / obj_dist)
    except ZeroDivisionError:
        v1 = np.inf

    m1 = -v1 / obj_dist if np.isfinite(v1) and obj_dist != 0 else np.inf
    z_int = z_L1 + v1 if np.isfinite(v1) else np.inf
    h_int = obj_height * m1 if np.isfinite(m1) else np.nan

    if np.isfinite(z_int):
        s2 = z_L2 - z_int
        try:
            v2_formula = 1.0 / (1.0 / f2 - 1.0 / s2)
        except ZeroDivisionError:
            v2_formula = np.inf
    else:
        s2 = np.inf
        v2_formula = np.inf

    image_z = None
    image_h = None
    image_type = 'none'

    rays = [(y2, m) for (y2, m) in rays_after_L2[obj_height] if abs(m) > 1e-9]
    zs, hs = [], []
    for i in range(len(rays)):
        y2a, ma = rays[i]
        for j in range(i + 1, len(rays)):
            y2b, mb = rays[j]
            if abs(ma - mb) < 1e-9:
                continue
            dz_cross = (y2b - y2a) / (ma - mb)
            z_cross = z_L2 + dz_cross
            h_cross = y2a + ma * dz_cross
            if np.isfinite(z_cross) and np.isfinite(h_cross):
                zs.append(z_cross)
                hs.append(h_cross)

    if zs:
        med = float(np.median(zs))
        spread = max(0.15 * abs(f2), 0.2)
        keep = [k for k, z in enumerate(zs) if abs(z - med) < spread]
        if keep:
            zs = [zs[k] for k in keep]
            hs = [hs[k] for k in keep]
        image_z = float(np.mean(zs))
        image_h = float(np.mean(hs))
        image_type = 'real' if image_z > z_L2 else 'virtual'

    v2 = image_z - z_L2 if image_z is not None else np.nan
    m2 = image_h / h_int if (image_h is not None and np.isfinite(h_int) and abs(h_int) > 1e-12) else np.nan
    m_total = image_h / obj_height if (image_h is not None and abs(obj_height) > 1e-12) else np.nan

    return all_paths, {
        'v1': v1, 'v2': v2, 'v2_formula': v2_formula,
        'm1': m1, 'm2': m2, 'm_total': m_total,
        'z_L1': z_L1, 'z_L2': z_L2,
        'image_z': image_z, 'image_h': image_h, 'image_type': image_type,
        'z_int': z_int, 'h_int': h_int,
        'obj_h': obj_height, 'f1': f1, 'f2': f2,
        'tube': tube_length, 'obj_dist': obj_dist, 's2': s2
    }


def thick_lens_efl(R1, R2, d, n):
    R2c = -R2   # Cartesian sign: second surface curves away from incoming light
    P1  = (n - 1) / R1
    P2  = (1 - n) / R2c        # = (n-1)/R2
    P   = P1 + P2 - (d / n) * P1 * P2
    return 1.0 / P if abs(P) > 1e-12 else np.inf

def trace_thick_lens(obj_height, obj_dist, R1, R2, thickness, n_lens, n_rays=9):
    z_front = obj_dist
    z_back  = z_front + thickness

    c_front_z = z_front + R1
    c_back_z  = z_back  - R2

    efl = thick_lens_efl(R1, R2, thickness, n_lens)

    object_points = [obj_height, 0.0, -obj_height]
    colors = ['red', 'green', 'blue']
    all_paths = []
    image_zs = []

    for op, col in zip(object_points, colors):
        angles = np.linspace(-0.18, 0.18, n_rays)
        for angle in angles:
            path = [(0.0, op)]
            pos   = np.array([op,            0.0])
            d_vec = np.array([np.sin(angle), np.cos(angle)])
            d_vec /= np.linalg.norm(d_vec)

            c_front_2d = np.array([0.0, c_front_z])   # [x, z]
            t = intersect_sphere(pos[::-1], d_vec[::-1], c_front_2d[::-1], R1)
            if t is None:
                path.append((z_front + 1, op)); all_paths.append((path, col, op)); continue
            hit1 = pos + t * d_vec
            path.append((hit1[1], hit1[0]))   # append (z, x)

            # refract — pass [z,x] coords to refract_at_sphere
            c_f_zx = np.array([c_front_z, 0.0])
            hit1_zx = hit1[::-1]
            d_zx    = d_vec[::-1]
            new_d_zx = refract_at_sphere(hit1_zx, d_zx, c_f_zx, 1.0, n_lens)
            if new_d_zx is None:
                all_paths.append((path, col, op)); continue
            d_vec = new_d_zx[::-1]   # back to [x, z]

            # ── hit back surface ──
            pos2 = hit1.copy()
            c_back_2d = np.array([0.0, c_back_z])   # [x, z]
            t2 = intersect_sphere(pos2[::-1], d_vec[::-1], c_back_2d[::-1], R2)
            if t2 is None:
                path.append((pos2[1]+5, pos2[0])); all_paths.append((path, col, op)); continue
            hit2 = pos2 + t2 * d_vec
            path.append((hit2[1], hit2[0]))

            c_b_zx  = np.array([c_back_z, 0.0])
            hit2_zx = hit2[::-1]
            d_zx    = d_vec[::-1]
            new_d_zx = refract_at_sphere(hit2_zx, d_zx, c_b_zx, n_lens, 1.0)
            if new_d_zx is None:
                all_paths.append((path, col, op)); continue
            d_vec = new_d_zx[::-1]

            # ── propagate to image region ──
            pos3  = hit2.copy()
            t_ext = max(abs(efl) * 3.5, 1.0)
            hit3  = pos3 + t_ext * d_vec
            path.append((hit3[1], hit3[0]))
            all_paths.append((path, col, op))

            # find focus: z where x=0 for paraxial rays
            if abs(op) < 1e-6 and abs(angle) < 0.06 and abs(d_vec[0]) > 1e-9:
                t_cross = -pos3[0] / d_vec[0]
                if t_cross > 0:
                    image_zs.append(pos3[1] + t_cross * d_vec[1])

    img_z = float(np.median(image_zs)) if image_zs else None

    return all_paths, {
        'efl': efl, 'R1': R1, 'R2': R2,
        'd': thickness, 'n': n_lens,
        'obj_dist': obj_dist, 'obj_h': obj_height,
        'img_z': img_z,
        'z_front': z_front, 'z_back': z_back
    }

# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────

def draw_thin_lens(ax, z, half_size=1.2, label=''):
    ax.annotate('', xy=(z, half_size*0.95), xytext=(z, -half_size*0.95),
                arrowprops=dict(arrowstyle='<|-|>', color='steelblue', lw=2, mutation_scale=15))
    if label:
        ax.text(z, half_size + 0.12, label, ha='center', fontsize=8, color='steelblue')

def draw_thick_lens_boundary(ax, z_front, z_back, R1, R2, height=1.2):
    ys = np.linspace(-height, height, 80)
    # front surface: arc of sphere with center at z_front+R1
    zs_f = z_front + R1 - np.sqrt(np.maximum(R1**2 - ys**2, 0))
    # back  surface: arc of sphere with center at z_back-R2
    zs_b = z_back  - R2 + np.sqrt(np.maximum(R2**2 - ys**2, 0))
    ax.fill_betweenx(ys, zs_f, zs_b, alpha=0.15, color='lightblue')
    ax.plot(zs_f, ys, 'b-', lw=1.5)
    ax.plot(zs_b, ys, 'b-', lw=1.5)

def draw_arrow(ax, z, h, color):
    if abs(h) > 1e-9:
        ax.annotate('', xy=(z, h), xytext=(z, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
    else:
        ax.plot([z], [0], 'o', color=color, ms=5)

# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────

fig = plt.figure(figsize=(16, 10), facecolor='white')
fig.canvas.manager.set_window_title('Ray Tracing — Optical Instruments')

ax_main = fig.add_axes([0.05, 0.42, 0.65, 0.54], facecolor='white')
ax_img  = fig.add_axes([0.72, 0.52, 0.26, 0.44], facecolor='white')
ax_info = fig.add_axes([0.72, 0.42, 0.26, 0.08], facecolor='white')

sl_y, dsl = 0.33, 0.055
sl_kw = dict(facecolor='#f0f0f0')

ax_f1   = fig.add_axes([0.08, sl_y,         0.35, 0.025], **sl_kw)
ax_f2   = fig.add_axes([0.08, sl_y-dsl,     0.35, 0.025], **sl_kw)
ax_tube = fig.add_axes([0.08, sl_y-2*dsl,   0.35, 0.025], **sl_kw)
ax_obj  = fig.add_axes([0.08, sl_y-3*dsl,   0.35, 0.025], **sl_kw)
ax_objh = fig.add_axes([0.08, sl_y-4*dsl,   0.35, 0.025], **sl_kw)

ax_R1   = fig.add_axes([0.55, sl_y,         0.35, 0.025], **sl_kw)
ax_R2   = fig.add_axes([0.55, sl_y-dsl,     0.35, 0.025], **sl_kw)
ax_d    = fig.add_axes([0.55, sl_y-2*dsl,   0.35, 0.025], **sl_kw)
ax_nn   = fig.add_axes([0.55, sl_y-3*dsl,   0.35, 0.025], **sl_kw)

ax_mode  = fig.add_axes([0.50, 0.01, 0.12, 0.12], facecolor='#f0f0f0')
ax_reset = fig.add_axes([0.88, 0.04, 0.08, 0.04], facecolor='#f0f0f0')

s_f1   = Slider(ax_f1,   'f₁ obj (cm)',   0.5,  5.0,  valinit=1.0,  color='#1976d2')
s_f2   = Slider(ax_f2,   'f₂ eye (cm)',   1.0,  8.0,  valinit=3.0,  color='#1976d2')
s_tube = Slider(ax_tube, 'tube L (cm)',   5.0,  25.0, valinit=16.0, color='#1976d2')
s_obj  = Slider(ax_obj,  'obj dist (cm)', 0.6,  4.0,  valinit=1.3,  color='#1976d2')
s_objh = Slider(ax_objh, 'obj height',    0.05, 1.0,  valinit=0.3,  color='#1976d2')

s_R1   = Slider(ax_R1,   'R₁ (cm)',       1.0,  8.0,  valinit=3.0,  color='#ff6f00')
s_R2   = Slider(ax_R2,   'R₂ (cm)',       1.0,  8.0,  valinit=3.0,  color='#ff6f00')
s_d    = Slider(ax_d,    'thickness d',   0.01, 3.0,  valinit=0.3,  color='#ff6f00')
s_nn   = Slider(ax_nn,   'n glass',       1.3,  2.0,  valinit=1.52, color='#ff6f00')

radio    = RadioButtons(ax_mode, ('Microscope\n(thin)', 'Thick lens'), activecolor='#1976d2')
btn_reset = Button(ax_reset, 'Reset', color='#f0f0f0', hovercolor='#1976d2')

for ax in [ax_main, ax_img, ax_info]:
    for sp in ax.spines.values(): sp.set_color('#333')
    ax.tick_params(colors='#333')

def update(_):
    mode = radio.value_selected
    ax_main.cla(); ax_img.cla(); ax_info.cla()

    for ax in [ax_main, ax_img, ax_info]:
        ax.set_facecolor('white')
        for sp in ax.spines.values(): sp.set_color('#333')
        ax.tick_params(colors='#333', labelsize=8)
        ax.xaxis.label.set_color('#333')
        ax.yaxis.label.set_color('#333')

    if 'thin' in mode.lower() or 'micro' in mode.lower():
        f1, f2 = s_f1.val, s_f2.val
        tube, odist, oh = s_tube.val, s_obj.val, s_objh.val

        paths, info = trace_thin_lens_system(oh, odist, f1, f2, tube, n_rays=7)
        for path, col, op in paths:
            zs = [p[0] for p in path]; ys = [p[1] for p in path]
            ax_main.plot(zs, ys, color=col, alpha=0.5, lw=0.8)

        hs = oh * 1.5 + 0.6
        draw_thin_lens(ax_main, info['z_L1'], hs, label=f'L₁ f={f1:.1f}')
        draw_thin_lens(ax_main, info['z_L2'], hs, label=f'L₂ f={f2:.1f}')

        z_end = info['z_L2'] + abs(f2)*3.5
        ax_main.axhline(0, color='#aaa', lw=0.8, ls='--')
        ax_main.axvline(info['z_L1'], color='steelblue', lw=0.5, ls=':')
        ax_main.axvline(info['z_L2'], color='steelblue', lw=0.5, ls=':')
        draw_arrow(ax_main, 0, oh, '#2e7d32')

        z_int = info['z_int']
        h_int = info['h_int']
        if np.isfinite(z_int) and 0 < z_int < z_end:
            ax_main.axvline(z_int, color='#ff6f00', lw=0.7, ls='--', alpha=0.8)
            draw_arrow(ax_main, z_int, h_int, '#ff6f00')
            ax_main.text(z_int, -hs*0.85, 'img₁', color='#ff6f00', fontsize=7, ha='center')

        if info['image_z'] is not None and np.isfinite(info['image_z']):
            iz, ih = info['image_z'], info['image_h']
            if 0 < iz < z_end:
                ls = '--' if info['image_type'] == 'real' else ':'
                label = 'img₂' if info['image_type'] == 'real' else 'img₂ virt'
                ax_main.axvline(iz, color='#c62828', lw=1, ls=ls, alpha=0.9)
                draw_arrow(ax_main, iz, ih, '#c62828')
                ax_main.text(iz, -hs*0.85, label, color='#c62828', fontsize=7, ha='center')

        ax_main.set_xlim(-0.5, z_end); ax_main.set_ylim(-hs, hs)
        ax_main.set_xlabel('z (cm)'); ax_main.set_ylabel('height (cm)')
        ax_main.set_title('Microscope — Thin Lens Ray Tracing', color='#000', fontsize=11, weight='bold')
        ax_main.grid(True, alpha=0.2)

        if (info['image_z'] is not None and np.isfinite(info['image_z'])
                and info['image_h'] is not None and np.isfinite(info['image_h'])):
            ih = info['image_h']

            scale = 0.42 / max(abs(oh), abs(ih), 1e-6)
            ax_img.annotate('', xy=(0.25, oh * scale), xytext=(0.25, 0),
                            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))
            ax_img.annotate('', xy=(0.75, ih * scale), xytext=(0.75, 0),
                            arrowprops=dict(arrowstyle='->', color='#c62828', lw=2))

            ax_img.set_xlim(0,1); ax_img.set_ylim(-0.6, 0.6)
            ax_img.axhline(0, color='#aaa', lw=0.8)
            ax_img.text(0.25, -0.5, 'Object',  color='#2e7d32', ha='center', fontsize=9)
            ax_img.text(0.75, -0.5, f'{info["image_type"].capitalize()} image\n×{info["m_total"]:.2f}', color='#c62828', ha='center', fontsize=9)
        ax_img.set_title('Object vs Image (normalized)', color='#000', fontsize=10, weight='bold')
        ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.grid(True, alpha=0.2)

        mt = info['m_total']
        ax_info.axis('off')
        err1 = abs(1/info['v1'] - 1/f1 + 1/odist) if np.isfinite(info['v1']) else np.nan
        err2 = abs(1/info['v2_formula'] - 1/f2 + 1/info['s2']) if np.isfinite(info['v2_formula']) and np.isfinite(info['s2']) else np.nan
        txt = (f"img₁: v₁={info['v1']:.2f}  m₁={info['m1']:.2f}  err₁={err1:.2e}\n"
               f"img₂ ({info['image_type']}): v₂(ray)={info['v2']:.2f}  v₂(form)={info['v2_formula']:.2f}\n"
               f"m₂={info['m2']:.2f}  M_total={mt:.2f}  ({'INVERTED' if mt<0 else 'UPRIGHT'})  err₂={err2:.2e}")
        ax_info.text(0.02, 0.5, txt, color='#1976d2', fontsize=8.5,
                     va='center', family='monospace', transform=ax_info.transAxes)

    else:
        R1, R2 = s_R1.val, s_R2.val
        d, nn  = s_d.val, s_nn.val
        odist, oh = s_obj.val, s_objh.val

        # Pass R1, R2 as positive — thick_lens takes care of sign convention internally
        paths, info = trace_thick_lens(oh, odist, R1, R2, d, nn, n_rays=9)

        for path, col, op in paths:
            zs = [p[0] for p in path]; ys = [p[1] for p in path]
            ax_main.plot(zs, ys, color=col, alpha=0.5, lw=0.8)

        hs = oh*1.5 + 0.7
        draw_thick_lens_boundary(ax_main, info['z_front'], info['z_back'],
                                 R1, R2, height=oh*1.4+0.4)

        z_end = info['z_back'] + abs(info['efl'])*2.5 + 1
        ax_main.axhline(0, color='#aaa', lw=0.8, ls='--')
        draw_arrow(ax_main, 0, oh, '#2e7d32')

        if info['img_z'] is not None:
            ax_main.axvline(info['img_z'], color='#c62828', lw=1, ls='--', alpha=0.8)
            ax_main.text(info['img_z'], -(oh*1.4+0.4)*0.85, 'image (ray trace)',
                         color='#c62828', fontsize=7, ha='center')

        # Thin-lens prediction (object distance from principal plane ≈ front vertex)
        efl = info['efl']
        try:
            v_thin = 1.0 / (1.0/efl - 1.0/(-odist))  # u = -odist
        except:
            v_thin = np.inf
        z_thin_img = info['z_front'] + v_thin
        if np.isfinite(z_thin_img) and z_thin_img < z_end + 5:
            ax_main.axvline(z_thin_img, color='#ff6f00', lw=1, ls=':', alpha=0.8)
            ax_main.text(z_thin_img, hs*0.82, 'thin\npred', color='#ff6f00', fontsize=7, ha='center')

        ax_main.set_xlim(-0.5, z_end); ax_main.set_ylim(-hs, hs)
        ax_main.set_xlabel('z (cm)'); ax_main.set_ylabel('height (cm)')
        ax_main.set_title('Thick Spherical Lens — Ray Tracing', color='#000', fontsize=11, weight='bold')
        ax_main.grid(True, alpha=0.2)

        ax_img.axis('off')
        ax_img.text(0.5, 0.65, 'Thick biconvex lens', color='#ff6f00',
                    ha='center', fontsize=11, transform=ax_img.transAxes, weight='bold')
        ax_img.text(0.5, 0.45, f'EFL = {efl:.3f} cm', color='#000',
                    ha='center', fontsize=11, transform=ax_img.transAxes)
        ax_img.text(0.5, 0.28,
                    f'R₁={R1:.2f}  R₂={R2:.2f}\nn={nn:.2f}   d={d:.2f} cm',
                    color='#333', ha='center', fontsize=9, transform=ax_img.transAxes)

        ax_info.axis('off')
        if info['img_z'] is not None:
            v_real = info['img_z'] - info['z_front']
            dev = v_real - v_thin if np.isfinite(v_thin) else float('nan')
            txt = (f"EFL={efl:.3f} cm\n"
                   f"Thin-lens pred:  v={v_thin:.3f} cm\n"
                   f"Ray-trace focus: v≈{v_real:.3f} cm   Δ={dev:+.4f} cm")
        else:
            txt = f"EFL={efl:.3f} cm\n(rays didn't converge)"
        ax_info.text(0.02, 0.5, txt, color='#ff6f00', fontsize=8.5,
                     va='center', family='monospace', transform=ax_info.transAxes)

    fig.canvas.draw_idle()

for s in [s_f1, s_f2, s_tube, s_obj, s_objh, s_R1, s_R2, s_d, s_nn]:
    s.on_changed(update)
radio.on_clicked(update)

def reset(event):
    for s in [s_f1, s_f2, s_tube, s_obj, s_objh, s_R1, s_R2, s_d, s_nn]:
        s.reset()
btn_reset.on_clicked(reset)

fig.text(0.08, 0.375, '── Тонкие линзы (микроскоп) ──', color='#1976d2',
         fontsize=9, style='italic', weight='bold')
fig.text(0.55, 0.375, '── Толстая линза ──', color='#ff6f00',
         fontsize=9, style='italic', weight='bold')

update(None)
plt.show()
