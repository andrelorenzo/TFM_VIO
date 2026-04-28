import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

G = -9.81


def skew(w):
    wx, wy, wz = w
    return np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0],
    ])


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    W = skew(w)
    A = np.sin(th) / th
    B = (1.0 - np.cos(th)) / (th * th)
    return np.eye(3) + A * W + B * (W @ W)


@dataclass
class Segment:
    duration: float
    a_body: np.ndarray
    omega_body: np.ndarray


@dataclass
class CaseSpec:
    name: str
    kind: str
    a_forward: float = 0.0
    yaw_rate: float = 0.0
    split: tuple = (0.3, 0.4)
    segments: list | None = None


@dataclass
class ImuNoiseModel:
    bg: np.ndarray
    ba: np.ndarray
    ng: np.ndarray
    kg: np.ndarray
    na: np.ndarray
    ka: np.ndarray
    source_path: str = ''


def _read_vec3_from_fs(fs, key, default=None):
    if default is None:
        default = np.zeros(3, dtype=float)

    node = fs.getNode(key)
    if node.empty():
        return np.array(default, dtype=float)

    mat = node.mat()
    if mat is None:
        return np.array(default, dtype=float)

    arr = np.array(mat, dtype=float).reshape(-1)
    if arr.size < 3:
        return np.array(default, dtype=float)
    return arr[:3].astype(float)


def load_imu_model_from_yaml(yaml_path):
    if cv2 is None:
        raise RuntimeError('OpenCV Python module (cv2) is required to read OpenCV YAML config files')

    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f'Could not open config YAML: {yaml_path}')

    allangx = _read_vec3_from_fs(fs, 'imu.allangx')
    allangy = _read_vec3_from_fs(fs, 'imu.allangy')
    allangz = _read_vec3_from_fs(fs, 'imu.allangz')
    allanax = _read_vec3_from_fs(fs, 'imu.allanax')
    allanay = _read_vec3_from_fs(fs, 'imu.allanay')
    allanaz = _read_vec3_from_fs(fs, 'imu.allanaz')

    model = ImuNoiseModel(
        bg=_read_vec3_from_fs(fs, 'imu.bg'),
        ba=_read_vec3_from_fs(fs, 'imu.ba'),
        ng=np.array([allangx[0], allangy[0], allangz[0]], dtype=float),
        kg=np.array([allangx[2], allangy[2], allangz[2]], dtype=float),
        na=np.array([allanax[0], allanay[0], allanaz[0]], dtype=float),
        ka=np.array([allanax[2], allanay[2], allanaz[2]], dtype=float),
        source_path=str(yaml_path),
    )
    fs.release()
    return model


def resolve_fs(fs_arg, yaml_path):
    if fs_arg is not None:
        return float(fs_arg)

    if yaml_path is not None and cv2 is not None:
        fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            node = fs.getNode('imu.gyrfps')
            if not node.empty():
                value = float(node.real())
                fs.release()
                if value > 0.0:
                    return value
            fs.release()

    return 200.0


def inject_imu_noise(gyro, accel, fs, model: ImuNoiseModel, seed=0,
                     use_bias=True, use_white_noise=True, use_random_walk=True):
    dt = 1.0 / fs
    rng = np.random.default_rng(seed)

    gyro_meas = gyro.copy()
    accel_meas = accel.copy()

    bg_rw = np.zeros(3, dtype=float)
    ba_rw = np.zeros(3, dtype=float)

    white_g_std = model.ng / np.sqrt(dt) if use_white_noise else np.zeros(3, dtype=float)
    white_a_std = model.na / np.sqrt(dt) if use_white_noise else np.zeros(3, dtype=float)
    rw_g_step = model.kg * np.sqrt(dt) if use_random_walk else np.zeros(3, dtype=float)
    rw_a_step = model.ka * np.sqrt(dt) if use_random_walk else np.zeros(3, dtype=float)
    static_bg = model.bg if use_bias else np.zeros(3, dtype=float)
    static_ba = model.ba if use_bias else np.zeros(3, dtype=float)

    for k in range(len(gyro_meas)):
        if use_random_walk:
            bg_rw += rw_g_step * rng.normal(size=3)
            ba_rw += rw_a_step * rng.normal(size=3)

        gyro_meas[k] += static_bg + bg_rw + white_g_std * rng.normal(size=3)
        accel_meas[k] += static_ba + ba_rw + white_a_std * rng.normal(size=3)

    return gyro_meas, accel_meas


def hold_segment(duration):
    return Segment(duration=duration, a_body=np.zeros(3), omega_body=np.zeros(3))



def line_segment(duration, a_forward=0.0):
    return Segment(
        duration=duration,
        a_body=np.array([a_forward, 0.0, 0.0], dtype=float),
        omega_body=np.zeros(3),
    )



def yaw_turn_segment(duration, yaw_rate, a_forward=0.0):
    return Segment(
        duration=duration,
        a_body=np.array([a_forward, 0.0, 0.0], dtype=float),
        omega_body=np.array([0.0, yaw_rate, 0.0], dtype=float),
    )


def motion_segment(duration, a_body, omega_body):
    return Segment(
        duration=duration,
        a_body=np.array(a_body, dtype=float),
        omega_body=np.array(omega_body, dtype=float),
    )


def build_long_complex_segments():
    # ~60 s total. Rich 6DoF excitation with varying translational and angular motion
    # in all axes, while keeping magnitudes moderate enough to remain readable.
    pattern = [
        (3.0, [0.80, 0.20, -0.10], [0.18, 0.10, -0.12]),
        (2.0, [0.00, 0.00,  0.00], [0.05, 0.22,  0.18]),
        (4.0, [-0.40, 0.55, 0.25], [-0.12, 0.16, 0.20]),
        (3.0, [0.25, -0.35, 0.60], [0.14, -0.20, 0.08]),
        (2.5, [0.10, 0.00, -0.45], [-0.18, 0.06, -0.22]),
        (2.5, [0.55, -0.25, 0.05], [0.10, -0.15, 0.24]),
        (3.5, [-0.30, -0.20, 0.35], [0.22, 0.08, -0.10]),
        (2.5, [0.00, 0.40, -0.30], [-0.08, -0.24, 0.16]),
    ]

    segments = []
    for _ in range(3):  # 3 * 20 s = 60 s
        for duration, a_body, omega_body in pattern:
            segments.append(motion_segment(duration, a_body, omega_body))
    return segments



def generate_real_imu(segments, fs=200.0, g=9.81, p0=None, v0=None, R0=None, substeps=20):
    if p0 is None:
        p0 = np.zeros(3)
    if v0 is None:
        v0 = np.zeros(3)
    if R0 is None:
        R0 = np.eye(3)

    dt = 1.0 / fs
    g_v = np.array([0.0, -g, 0.0], dtype=float)

    Ns = [max(1, int(round(seg.duration * fs))) for seg in segments]
    N = sum(Ns)  # Number of integration intervals

    t = np.arange(N + 1, dtype=float) * dt
    gyro = np.zeros((N + 1, 3), dtype=float)
    accel = np.zeros((N + 1, 3), dtype=float)

    p_w = np.zeros((N + 1, 3), dtype=float)
    v_w = np.zeros((N + 1, 3), dtype=float)
    R_wb = np.zeros((N + 1, 3, 3), dtype=float)

    p_w[0] = p0
    v_w[0] = v0
    R_wb[0] = R0

    k = 0
    for seg, nseg in zip(segments, Ns):
        for _ in range(nseg):
            omega_b = seg.omega_body.copy()
            a_b_cmd = seg.a_body.copy()

            Rk = R_wb[k]
            Rbw = Rk.T

            a_w = Rk @ a_b_cmd
            f_b = Rbw @ (a_w - g_v)

            gyro[k] = omega_b
            accel[k] = f_b

            p_curr = p_w[k].copy()
            v_curr = v_w[k].copy()
            R_curr = Rk.copy()
            h = dt / max(1, int(substeps))

            for _ in range(max(1, int(substeps))):
                # Midpoint orientation gives a much more faithful trajectory than
                # using only the orientation at the start of the whole IMU interval.
                R_mid = R_curr @ exp_so3(omega_b * (0.5 * h))
                a_w_mid = R_mid @ a_b_cmd
                p_curr = p_curr + v_curr * h + 0.5 * a_w_mid * h * h
                v_curr = v_curr + a_w_mid * h
                R_curr = R_curr @ exp_so3(omega_b * h)

            v_w[k + 1] = v_curr
            p_w[k + 1] = p_curr
            R_wb[k + 1] = R_curr
            k += 1

    if N > 0:
        gyro[N] = gyro[N - 1]
        accel[N] = accel[N - 1]

    return t, gyro, accel, p_w, v_w, R_wb



def save_imu_csv(filename, t, gyro, accel):
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ts', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
        for i in range(len(t)):
            writer.writerow([
                f'{1e3 * t[i]:.9f}',
                f'{gyro[i,0]:.9f}', f'{gyro[i,1]:.9f}', f'{gyro[i,2]:.9f}',
                f'{accel[i,0]:.9f}', f'{accel[i,1]:.9f}', f'{accel[i,2]:.9f}',
            ])



def compute_expected_preintegration(t, p_w, v_w, R_wb, g=9.81):
    if len(t) == 0:
        dt_total = 0.0
    else:
        dt_total = t[-1] - t[0]


    R_i = R_wb[0]
    R_j = R_wb[-1]
    p_i = p_w[0]
    p_j = p_w[-1]
    v_i = v_w[0]
    v_j = v_w[-1]

    dR = R_i.T @ R_j
    dv = R_i.T @ (v_j - v_i)
    dp = R_i.T @ (p_j - p_i - v_i * dt_total)

    return dR, dv, dp, dt_total



def save_expected_txt(filename, case_name, dR, dv, dp, dt_total, sample_count):
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        f.write(f'case: {case_name}\n')
        f.write(f'samples: {sample_count}\n')
        f.write(f'dt_total: {dt_total:.9f}\n')
        f.write('\n')
        f.write('Definition used:\n')
        f.write('dR = R_i^T R_j\n')
        f.write('dv = R_i^T (v_j - v_i)\n')
        f.write('dp = R_i^T (p_j - p_i - v_i * dt)\n')
        f.write('with g = [0, -9.81, 0]\n')
        f.write('\n')
        f.write('dR:\n')
        for row in dR:
            f.write(' '.join(f'{x:.9f}' for x in row) + '\n')
        f.write('\n')
        f.write('dv:\n')
        f.write(' '.join(f'{x:.9f}' for x in dv) + '\n')
        f.write('\n')
        f.write('dp:\n')
        f.write(' '.join(f'{x:.9f}' for x in dp) + '\n')


def save_noise_model_txt(filename, model: ImuNoiseModel, use_bias, use_white_noise, use_random_walk, seed):
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        f.write('Synthetic IMU noise model\n')
        f.write(f'source_yaml: {model.source_path}\n')
        f.write(f'seed: {seed}\n')
        f.write(f'use_bias: {int(use_bias)}\n')
        f.write(f'use_white_noise: {int(use_white_noise)}\n')
        f.write(f'use_random_walk: {int(use_random_walk)}\n')
        f.write('\n')
        f.write('bg:\n')
        f.write(' '.join(f'{x:.12e}' for x in model.bg) + '\n')
        f.write('ba:\n')
        f.write(' '.join(f'{x:.12e}' for x in model.ba) + '\n')
        f.write('Ng:\n')
        f.write(' '.join(f'{x:.12e}' for x in model.ng) + '\n')
        f.write('Kg:\n')
        f.write(' '.join(f'{x:.12e}' for x in model.kg) + '\n')
        f.write('Na:\n')
        f.write(' '.join(f'{x:.12e}' for x in model.na) + '\n')
        f.write('Ka:\n')
        f.write(' '.join(f'{x:.12e}' for x in model.ka) + '\n')


CASE_SPECS = [
    CaseSpec(name='case1_zero', kind='zero'),
    CaseSpec(name='case2_const_accel', kind='const_accel', a_forward=1.0),
    CaseSpec(name='case3_const_yaw', kind='const_yaw', yaw_rate=0.8),
    CaseSpec(
        name='case4_straight_turn_straight',
        kind='straight_turn_straight',
        a_forward=1.0,
        yaw_rate=1.2,
        split=(0.3, 0.4),
    ),
    CaseSpec(
        name='case5_full_6dof',
        kind='segments',
        segments=[
            ('motion', 0.40,  0.80,  0.20,  0.10,   0.30,  0.20, -0.10),
            ('motion', 0.50, -0.30,  0.60,  0.20,  -0.20,  0.35,  0.25),
            ('motion', 0.45,  0.15, -0.40,  0.55,   0.25, -0.30,  0.20),
            ('motion', 0.65,  0.00,  0.20, -0.35,  -0.15,  0.10, -0.40),
        ],
    ),
    CaseSpec(
        name='case6_long_complex_6dof',
        kind='long_complex_6dof',
    ),
    CaseSpec(
        name='custom_demo',
        kind='segments',
        segments=[
            ('line', 0.5, 1.0, 0.0),
            ('turn', 1.0, 0.2, 0.8),
            ('line', 0.5, 0.0, 0.0),
        ],
    ),
]

CASE_MAP = {spec.name: spec for spec in CASE_SPECS}



def case_names():
    return list(CASE_MAP.keys())



def build_segments_from_spec(spec: CaseSpec, fs=200.0, n=20):
    dt = 1.0 / fs
    total_duration = n * dt

    if spec.kind == 'zero':
        return [hold_segment(total_duration)]

    if spec.kind == 'const_accel':
        return [line_segment(total_duration, a_forward=spec.a_forward)]

    if spec.kind == 'const_yaw':
        return [yaw_turn_segment(total_duration, yaw_rate=spec.yaw_rate, a_forward=0.0)]

    if spec.kind == 'straight_turn_straight':
        frac1, frac2 = spec.split
        n1 = int(round(n * frac1))
        n2 = int(round(n * frac2))
        n3 = n - n1 - n2
        return [
            line_segment(n1 * dt, a_forward=spec.a_forward),
            yaw_turn_segment(n2 * dt, yaw_rate=spec.yaw_rate, a_forward=spec.a_forward),
            line_segment(n3 * dt, a_forward=spec.a_forward),
        ]

    if spec.kind == 'long_complex_6dof':
        return build_long_complex_segments()

    if spec.kind == 'segments':
        segments = []
        for seg in spec.segments or []:
            seg_type = seg[0]
            if seg_type == 'line':
                _, duration, a_forward, yaw_rate = seg
                segments.append(line_segment(duration, a_forward=a_forward))
            elif seg_type == 'turn':
                _, duration, a_forward, yaw_rate = seg
                segments.append(yaw_turn_segment(duration, yaw_rate=yaw_rate, a_forward=a_forward))
            elif seg_type == 'hold':
                _, duration, a_forward, yaw_rate = seg
                segments.append(hold_segment(duration))
            elif seg_type == 'motion':
                _, duration, ax, ay, az, wx, wy, wz = seg
                segments.append(motion_segment(duration, [ax, ay, az], [wx, wy, wz]))
            else:
                raise ValueError(f'Unknown segment type in spec {spec.name}: {seg_type}')
        return segments

    raise ValueError(f'Unknown case kind: {spec.kind}')



def build_case(case_name, fs=200.0, n=20):
    if case_name not in CASE_MAP:
        raise ValueError(f'Unknown case: {case_name}')
    segments = build_segments_from_spec(CASE_MAP[case_name], fs=fs, n=n)
    return generate_real_imu(segments, fs=fs, g=G)



def resolve_cases(case_arg: str):
    if case_arg == 'all':
        return case_names()
    if case_arg not in CASE_MAP:
        raise ValueError(f'Unknown case: {case_arg}')
    return [case_arg]



def output_path_for_case(out_arg: str, case_name: str, multiple: bool) -> Path:
    out_path = Path(out_arg)
    if multiple or out_path.suffix == '' or out_arg.endswith('/'):
        return out_path / f'{case_name}.csv'
    return out_path



def expected_path_for_csv(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + '_expected.txt')


def noise_path_for_csv(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + '_noise_model.txt')



def generate_and_save(case_list, fs, n, out_arg):
    saved = []
    multiple = len(case_list) > 1
    for case_name in case_list:
        t, gyro, accel, p_w, v_w, R_wb = build_case(case_name, fs=fs, n=n)
        out_path = output_path_for_case(out_arg, case_name, multiple)
        save_imu_csv(out_path, t, gyro, accel)
        dR, dv, dp, dt_total = compute_expected_preintegration(t, p_w, v_w, R_wb, g=G)
        txt_path = expected_path_for_csv(out_path)
        save_expected_txt(txt_path, case_name, dR, dv, dp, dt_total, len(t))
        saved.append((case_name, out_path, txt_path, len(t)))
    return saved


def generate_and_save_with_optional_noise(case_list, fs, n, out_arg, imu_model=None,
                                          use_bias=True, use_white_noise=True,
                                          use_random_walk=True, seed=0):
    saved = []
    multiple = len(case_list) > 1
    for case_name in case_list:
        t, gyro_clean, accel_clean, p_w, v_w, R_wb = build_case(case_name, fs=fs, n=n)
        gyro_out = gyro_clean
        accel_out = accel_clean

        if imu_model is not None:
            gyro_out, accel_out = inject_imu_noise(
                gyro_clean, accel_clean, fs, imu_model, seed=seed,
                use_bias=use_bias, use_white_noise=use_white_noise,
                use_random_walk=use_random_walk,
            )

        out_path = output_path_for_case(out_arg, case_name, multiple)
        save_imu_csv(out_path, t, gyro_out, accel_out)
        dR, dv, dp, dt_total = compute_expected_preintegration(t, p_w, v_w, R_wb, g=G)
        txt_path = expected_path_for_csv(out_path)
        save_expected_txt(txt_path, case_name, dR, dv, dp, dt_total, len(t))

        noise_txt = None
        if imu_model is not None:
            noise_txt = noise_path_for_csv(out_path)
            save_noise_model_txt(
                noise_txt, imu_model, use_bias=use_bias,
                use_white_noise=use_white_noise,
                use_random_walk=use_random_walk, seed=seed,
            )

        saved.append((case_name, out_path, txt_path, noise_txt, len(t)))
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate IMU CSV(s) with header ts,gx,gy,gz,ax,ay,az (ts in ms) and expected dR, dv, dp text files'
    )
    parser.add_argument(
        '--case',
        type=str,
        default='all',
        choices=case_names() + ['all'],
        help='Case to generate, or all to generate every case in CASE_SPECS',
    )
    parser.add_argument('--fs', type=float, default=None)
    parser.add_argument('--n', type=int, default=20, help='Number of IMU intervals to generate')
    parser.add_argument('--out', type=str, default='imu_output.csv')
    parser.add_argument('--config-yaml', type=str, default=None, help='OpenCV YAML config to read imu.bg/ba/allan* from')
    parser.add_argument('--seed', type=int, default=0, help='Random seed used for synthetic IMU noise')
    parser.add_argument('--inject-bias', action='store_true', help='Inject static biases imu.bg and imu.ba from config')
    parser.add_argument('--inject-noise', action='store_true', help='Inject white noise from Allan N terms in config')
    parser.add_argument('--inject-rw', action='store_true', help='Inject bias random walk from Allan K terms in config')
    args = parser.parse_args()

    imu_model = None
    if args.config_yaml is not None:
        imu_model = load_imu_model_from_yaml(args.config_yaml)

    inject_any = args.inject_bias or args.inject_noise or args.inject_rw
    if imu_model is not None and not inject_any:
        use_bias = True
        use_white_noise = True
        use_random_walk = True
    else:
        use_bias = args.inject_bias
        use_white_noise = args.inject_noise
        use_random_walk = args.inject_rw

    fs = resolve_fs(args.fs, args.config_yaml)
    selected_cases = resolve_cases(args.case)
    results = generate_and_save_with_optional_noise(
        selected_cases, fs=fs, n=args.n, out_arg=args.out,
        imu_model=imu_model if (use_bias or use_white_noise or use_random_walk) else None,
        use_bias=use_bias, use_white_noise=use_white_noise,
        use_random_walk=use_random_walk, seed=args.seed,
    )

    for case_name, out_path, txt_path, noise_txt, sample_count in results:
        print(f'Saved {sample_count} samples for {case_name} to {out_path}')
        print(f'Saved expected dR/dv/dp for {case_name} to {txt_path}')
        if noise_txt is not None:
            print(f'Saved IMU noise model for {case_name} to {noise_txt}')
