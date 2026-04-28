import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
PI = np.pi
G_REF = (0.0, -9.770184612169, 0.0)  # m/s^2

# ========================================
# DEFAULTS / FALLBACKS
# ========================================
DEFAULT_CSV_PATH = "input.csv"

DEFAULT_GX_COL = "gx"
DEFAULT_GY_COL = "gy"
DEFAULT_GZ_COL = "gz"
DEFAULT_AX_COL = "ax"
DEFAULT_AY_COL = "ay"
DEFAULT_AZ_COL = "az"

DEFAULT_USE_TIMESTAMP = True
DEFAULT_TIMESTAMP_COL = "ts"
DEFAULT_FS = 100.0  # Hz

DEFAULT_OUTPUT_DIR = ""   # si está vacío, se autogenera
DEFAULT_SHOW_PLOTS = False
DEFAULT_SAVE_PLOTS = True
DEFAULT_MAX_CLUSTERS = 100

DEFAULT_TXT_FILENAME = "allan_bias_results.txt"


def allan_variance(data: np.ndarray, f: float, max_clusters: int = 100):
    """
    Compute Allan variance.

    Parameters
    ----------
    data : np.ndarray
        1D signal array. Units:
        - gyro: rad/s
        - accel: m/s^2
    f : float
        Sample rate in Hz
    max_clusters : int
        Maximum number of cluster sizes

    Returns
    -------
    tau : np.ndarray
        Averaging times [s]
    allan_var : np.ndarray
        Allan variance
    """
    tau0 = 1.0 / f
    data_integral = np.cumsum(data) * tau0
    n = len(data_integral)

    if n < 10:
        raise ValueError("Muy pocas muestras para calcular Allan variance.")

    m_max = int(2 ** np.floor(np.log2(n / 2)))
    if m_max < 1:
        raise ValueError("No hay suficientes datos para formar clusters de Allan.")

    m = np.logspace(np.log10(1), np.log10(m_max), num=max_clusters)
    m = np.ceil(m)
    m = np.unique(m).astype(int)

    tau = m * tau0

    allan_var = np.zeros(len(m), dtype=np.float64)
    for i, mi in enumerate(m):
        allan_var[i] = np.sum(
            (
                data_integral[2 * mi:n]
                - 2 * data_integral[mi:n - mi]
                + data_integral[0:n - 2 * mi]
            ) ** 2
        )

    allan_var /= (2 * tau ** 2) * (n - 2 * m)
    return tau, allan_var


def extract_allan_constants(tau: np.ndarray, allan_dev: np.ndarray):
    """
    Extract Allan deviation coefficients:
    - N: white noise / random walk coefficient
    - B: bias instability
    - K: rate random walk coefficient
    """
    valid = np.isfinite(tau) & np.isfinite(allan_dev) & (tau > 0) & (allan_dev > 0)
    tau = tau[valid]
    allan_dev = allan_dev[valid]

    if len(tau) < 5:
        raise ValueError("No hay suficientes puntos válidos en Allan deviation.")

    log_tau = np.log10(tau)
    log_allan_dev = np.log10(allan_dev)
    dlog_allan_dev = np.diff(log_allan_dev) / np.diff(log_tau)

    # White noise / random walk
    slope_n = -0.5
    idx_n = np.argmin(np.abs(dlog_allan_dev - slope_n))
    intercept_n = log_allan_dev[idx_n] - slope_n * log_tau[idx_n]
    N = 10 ** intercept_n

    # Rate random walk
    slope_k = 0.5
    idx_k = np.argmin(np.abs(dlog_allan_dev - slope_k))
    intercept_k = log_allan_dev[idx_k] - slope_k * log_tau[idx_k]
    log_k = slope_k * np.log10(3.0) + intercept_k
    K = 10 ** log_k

    # Bias instability
    slope_b = 0.0
    idx_b = np.argmin(np.abs(dlog_allan_dev - slope_b))
    intercept_b = log_allan_dev[idx_b]
    scf_b = np.sqrt(2.0 * np.log(2.0) / PI)
    log_b = intercept_b - np.log10(scf_b)
    B = 10 ** log_b

    return {
        "N": float(N),
        "B": float(B),
        "K": float(K),
        "tau_N_s": float(tau[idx_n]),
        "tau_B_s": float(tau[idx_b]),
        "tau_K_s": float(tau[idx_k]),
        "slope_N_found": float(dlog_allan_dev[idx_n]),
        "slope_B_found": float(dlog_allan_dev[idx_b]),
        "slope_K_found": float(dlog_allan_dev[idx_k]),
        "allan_dev_min": float(np.min(allan_dev)),
        "tau_at_allan_dev_min_s": float(tau[np.argmin(allan_dev)]),
        "idx_n": int(idx_n),
        "idx_b": int(idx_b),
        "idx_k": int(idx_k),
    }

@dataclass
class ManualFitResult:
    mode: str
    tau1: float
    tau2: float
    sigma1: float
    sigma2: float
    slope_measured: float
    intercept_forced: float
    value: float


def fit_forced_line_loglog(tau_sel: np.ndarray, allan_sel: np.ndarray, forced_slope: float):
    """
    Ajusta en log-log una recta con pendiente fija:
        log10(sigma) = forced_slope * log10(tau) + b
    y devuelve el intercepto b.
    """
    x = np.log10(tau_sel)
    y = np.log10(allan_sel)

    # mejor intercepto LS con pendiente fija
    b = np.mean(y - forced_slope * x)
    return b


def compute_manual_constant(tau_sel: np.ndarray, allan_sel: np.ndarray, mode: str) -> ManualFitResult:
    """
    mode:
      - 'N' => pendiente -0.5
      - 'B' => pendiente  0.0
      - 'K' => pendiente +0.5
    """
    if len(tau_sel) < 2:
        raise ValueError("Selecciona al menos 2 puntos.")

    x1 = np.log10(tau_sel[0])
    x2 = np.log10(tau_sel[-1])
    y1 = np.log10(allan_sel[0])
    y2 = np.log10(allan_sel[-1])

    slope_measured = (y2 - y1) / (x2 - x1)

    if mode == "N":
        forced_slope = -0.5
        b = fit_forced_line_loglog(tau_sel, allan_sel, forced_slope)
        value = 10 ** b

    elif mode == "B":
        forced_slope = 0.0
        b = fit_forced_line_loglog(tau_sel, allan_sel, forced_slope)
        scf_b = np.sqrt(2.0 * np.log(2.0) / PI)
        value = (10 ** b) / scf_b

    elif mode == "K":
        forced_slope = 0.5
        b = fit_forced_line_loglog(tau_sel, allan_sel, forced_slope)
        # sigma = K * sqrt(tau/3)
        # log10(sigma) = 0.5 log10(tau) + log10(K/sqrt(3))
        # => log10(K) = b + 0.5 log10(3)
        value = 10 ** (b + 0.5 * np.log10(3.0))

    else:
        raise ValueError("mode debe ser 'N', 'B' o 'K'")

    return ManualFitResult(
        mode=mode,
        tau1=float(tau_sel[0]),
        tau2=float(tau_sel[-1]),
        sigma1=float(allan_sel[0]),
        sigma2=float(allan_sel[-1]),
        slope_measured=float(slope_measured),
        intercept_forced=float(b),
        value=float(value),
    )

def estimate_fs_from_timestamp_ms(timestamp_ms: np.ndarray):
    """
    Estima fs a partir de una columna timestamp en milisegundos.
    """
    timestamp_ms = np.asarray(timestamp_ms, dtype=np.float64)

    if len(timestamp_ms) < 2:
        raise ValueError("No hay suficientes timestamps para estimar la frecuencia.")

    dt_ms = np.diff(timestamp_ms)
    dt_ms = dt_ms[np.isfinite(dt_ms) & (dt_ms > 0)]

    if len(dt_ms) == 0:
        raise ValueError("Los timestamps no permiten calcular un dt válido.")

    dt_s = np.median(dt_ms) / 1000.0
    fs = 1.0 / dt_s
    return fs, dt_s


def safe_numeric_column(df: pd.DataFrame, col_name: str) -> np.ndarray:
    if col_name not in df.columns:
        raise ValueError(
            f"La columna '{col_name}' no existe en el CSV. "
            f"Disponibles: {', '.join(df.columns.astype(str))}"
        )

    data = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) < 10:
        raise ValueError(f"No hay suficientes muestras válidas en la columna {col_name}.")
    return data


def compute_biases(signals: dict):
    """
    Bias estimation:
    - Gyro static: true angular rate = 0 rad/s
    - Accel horizontal:
      ax_true = G_REF[0]
      ay_true = G_REF[1]
      az_true = G_REF[2]
    """
    gyro_bias = {
        "gx_bias": float(np.mean(signals["gx"])),
        "gy_bias": float(np.mean(signals["gy"])),
        "gz_bias": float(np.mean(signals["gz"])),
    }

    accel_bias = {
        "ax_bias": float(np.mean(signals["ax"]) - G_REF[0]),
        "ay_bias": float(np.mean(signals["ay"]) - G_REF[1]),
        "az_bias": float(np.mean(signals["az"]) - G_REF[2]),
    }

    return gyro_bias, accel_bias


def compute_axis_results(signals: dict, fs: float, max_clusters: int):
    axis_results = {}
    for axis_name, signal in signals.items():
        tau, allan_var = allan_variance(signal, fs, max_clusters=max_clusters)
        allan_dev = np.sqrt(allan_var)
        constants = extract_allan_constants(tau, allan_dev)

        axis_results[axis_name] = {
            "mean": float(np.mean(signal)),
            "std": float(np.std(signal)),
            "tau": tau,
            "allan_var": allan_var,
            "allan_dev": allan_dev,
            "constants": constants,
        }
    return axis_results
def interactive_manual_fit(tau: np.ndarray, allan_dev: np.ndarray, axis_name: str, units: str):
    """
    Herramienta interactiva para ajustar visualmente las rectas teóricas:
      - N: pendiente -1/2
      - B: pendiente  0
      - K: pendiente +1/2

    Controles:
      n / b / k      -> seleccionar tipo de recta
      left / right   -> mover horizontalmente
      up / down      -> mover verticalmente
      pageup/down    -> mover verticalmente más rápido
      enter          -> guardar valor actual
      c              -> borrar resultados guardados
      q              -> salir
    """
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title(
        f"Manual Allan fit - {axis_name}\n"
        "n/b/k: modo | ←/→: mover horizontal | ↑/↓: mover vertical | Enter: guardar | c: limpiar | q: salir"
    )

    ax.plot(tau, allan_dev, label="Allan deviation")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ [s]")
    ax.set_ylabel(f"$\\sigma(\\tau)$ [{units}]")
    ax.grid(True, which="both", ls="-", color="0.65")

    log_tau = np.log10(tau)
    log_sigma = np.log10(allan_dev)

    # punto inicial razonable: cerca del centro de la curva
    idx0 = len(tau) // 2
    anchor_x = log_tau[idx0]
    anchor_y = log_sigma[idx0]

    current_mode = ["N"]  # mutable para closures
    saved_results = {"N": None, "B": None, "K": None}

    line_artist = [None]
    text_artist = [None]
    saved_artists = []

    def mode_slope(mode):
        if mode == "N":
            return -0.5
        elif mode == "B":
            return 0.0
        elif mode == "K":
            return 0.5
        else:
            raise ValueError("Modo inválido")

    def build_line(mode, x_anchor, y_anchor, xvals):
        m = mode_slope(mode)
        yvals = m * (xvals - x_anchor) + y_anchor
        return yvals

    def compute_value_from_anchor(mode, x_anchor, y_anchor):
        # y = log10(sigma), x = log10(tau)
        # N: sigma = N / sqrt(tau)        => log10(N) = y + 0.5 x
        # B: sigma = B * scf_b           => log10(B) = y - log10(scf_b)
        # K: sigma = K * sqrt(tau/3)     => log10(K) = y - 0.5 x + 0.5 log10(3)
        scf_b = np.sqrt(2.0 * np.log(2.0) / PI)

        if mode == "N":
            val = 10 ** (y_anchor + 0.5 * x_anchor)
        elif mode == "B":
            val = 10 ** (y_anchor - np.log10(scf_b))
        elif mode == "K":
            val = 10 ** (y_anchor - 0.5 * x_anchor + 0.5 * np.log10(3.0))
        else:
            raise ValueError("Modo inválido")
        return float(val)

    def redraw():
        mode = current_mode[0]

        if line_artist[0] is not None:
            line_artist[0].remove()
            line_artist[0] = None

        if text_artist[0] is not None:
            text_artist[0].remove()
            text_artist[0] = None

        yline = build_line(mode, anchor_x, anchor_y, log_tau)
        sigma_line = 10 ** yline

        value = compute_value_from_anchor(mode, anchor_x, anchor_y)

        line_artist[0], = ax.plot(tau, sigma_line, "--", linewidth=2.0, label=f"Fit {mode}")

        label = (
            f"Modo: {mode}\n"
            f"tau_anchor = {10**anchor_x:.6e} s\n"
            f"sigma_anchor = {10**anchor_y:.6e}\n"
            f"{mode} = {value:.6e}"
        )

        text_artist[0] = ax.text(
            0.02, 0.98, label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        fig.canvas.draw_idle()

    def save_current_result():
        mode = current_mode[0]
        value = compute_value_from_anchor(mode, anchor_x, anchor_y)

        result = ManualFitResult(
            mode=mode,
            tau1=float(10 ** log_tau[0]),
            tau2=float(10 ** log_tau[-1]),
            sigma1=float(10 ** build_line(mode, anchor_x, anchor_y, np.array([log_tau[0]]))[0]),
            sigma2=float(10 ** build_line(mode, anchor_x, anchor_y, np.array([log_tau[-1]]))[0]),
            slope_measured=float(mode_slope(mode)),
            intercept_forced=float(anchor_y - mode_slope(mode) * anchor_x),
            value=float(value),
        )
        saved_results[mode] = result

        yline = build_line(mode, anchor_x, anchor_y, log_tau)
        sigma_line = 10 ** yline
        artist, = ax.plot(tau, sigma_line, ":", linewidth=1.5)
        saved_artists.append(artist)

        print("\n----------------------------------------")
        print(f"[MANUAL {mode}] {axis_name}")
        print(f"tau_anchor = {10**anchor_x:.6e} s")
        print(f"sigma_anchor = {10**anchor_y:.6e}")
        print(f"{mode} = {value:.12e}")
        print("----------------------------------------\n")

    def clear_saved():
        saved_results["N"] = None
        saved_results["B"] = None
        saved_results["K"] = None

        for a in saved_artists:
            a.remove()
        saved_artists.clear()

        fig.canvas.draw_idle()

    def snap_anchor_to_curve():
        nonlocal anchor_y
        idx = int(np.argmin(np.abs(log_tau - anchor_x)))
        anchor_y = log_sigma[idx]

    def on_key(event):
        nonlocal anchor_x, anchor_y

        x_step = 0.03
        y_step = 0.03
        y_step_big = 0.08

        if event.key == "n":
            current_mode[0] = "N"
            snap_anchor_to_curve()
            redraw()

        elif event.key == "b":
            current_mode[0] = "B"
            snap_anchor_to_curve()
            redraw()

        elif event.key == "k":
            current_mode[0] = "K"
            snap_anchor_to_curve()
            redraw()

        elif event.key == "left":
            anchor_x = max(np.min(log_tau), anchor_x - x_step)
            snap_anchor_to_curve()
            redraw()

        elif event.key == "right":
            anchor_x = min(np.max(log_tau), anchor_x + x_step)
            snap_anchor_to_curve()
            redraw()

        elif event.key == "up":
            anchor_y += y_step
            redraw()

        elif event.key == "down":
            anchor_y -= y_step
            redraw()

        elif event.key == "pageup":
            anchor_y += y_step_big
            redraw()

        elif event.key == "pagedown":
            anchor_y -= y_step_big
            redraw()

        elif event.key == "enter":
            save_current_result()
            redraw()

        elif event.key == "c":
            clear_saved()
            redraw()

        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.tight_layout()
    plt.show()

    return saved_results

def plot_allan_result(
    tau: np.ndarray,
    allan_dev: np.ndarray,
    constants: dict,
    axis_name: str,
    units: str,
    output_path: Path | None = None,
    show_plot: bool = False,
):
    """
    Ploteo estilo ejemplo del usuario:
    - Allan deviation
    - Línea N con pendiente -1/2
    - Línea K con pendiente +1/2
    - Línea B horizontal
    """
    N = constants["N"]
    B = constants["B"]
    K = constants["K"]

    scf_b = np.sqrt(2.0 * np.log(2.0) / PI)

    line_n = N / np.sqrt(tau)
    line_k = K * np.sqrt(tau / 3.0)
    line_b = B * scf_b * np.ones(len(tau))

    tau_n = 1.0
    tau_k = 3.0
    tau_b = constants["tau_B_s"]

    fig = plt.figure(figsize=(10, 7))
    plt.title(f"Allan deviation - {axis_name}")

    plt.plot(tau, allan_dev, label="Allan deviation")

    plt.plot(tau, line_n, ls="--", label=r"$\sigma_N$")
    plt.plot(tau_n, N, "o")
    plt.text(tau_n, N, f"N={N:.6e}")

    plt.plot(tau, line_k, ls="--", label=r"$\sigma_K$")
    plt.plot(tau_k, K, "o")
    plt.text(tau_k, K, f"K={K:.6e}")

    plt.plot(tau, line_b, ls="--", label=r"$\sigma_B$")
    plt.plot(tau_b, scf_b * B, "o")
    plt.text(tau_b, scf_b * B, f"B={B:.6e}")

    plt.xlabel(r"$\tau$ [s]")
    plt.ylabel(f"$\\sigma(\\tau)$ [{units}]")
    plt.grid(True, which="both", ls="-", color="0.65")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def save_results_txt(
    output_path: Path,
    csv_path: Path,
    fs: float,
    dt_s: float,
    colmap: dict,
    gyro_bias: dict,
    accel_bias: dict,
    axis_results: dict,
):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("IMU ALLAN VARIANCE + BIAS RESULTS\n")
        f.write("=================================\n\n")

        f.write(f"CSV file: {csv_path}\n")
        f.write(f"Sample rate [Hz]: {fs:.9f}\n")
        f.write(f"Sample period [s]: {dt_s:.9f}\n\n")

        f.write("INPUT UNITS ASSUMED\n")
        f.write("-------------------\n")
        f.write("Gyroscope: rad/s\n")
        f.write("Accelerometer: m/s^2\n\n")

        f.write("COLUMN MAPPING\n")
        f.write("--------------\n")
        for k, v in colmap.items():
            f.write(f"{k} -> {v}\n")
        f.write("\n")

        f.write("BIAS ESTIMATION ASSUMPTIONS\n")
        f.write("---------------------------\n")
        f.write("Gyro assumed static => true rate = 0 rad/s on all axes.\n")
        f.write("Accel assumed perfectly horizontal with:\n")
        f.write(f"  ax_true = {G_REF[0]:.5f} m/s^2\n")
        f.write(f"  ay_true = {G_REF[1]:.5f} m/s^2\n")
        f.write(f"  az_true = {G_REF[2]:.5f} m/s^2\n")

        f.write("ESTIMATED GYRO BIASES [rad/s]\n")
        f.write("-----------------------------\n")
        f.write(f"bgx = {gyro_bias['gx_bias']:.12e}\n")
        f.write(f"bgy = {gyro_bias['gy_bias']:.12e}\n")
        f.write(f"bgz = {gyro_bias['gz_bias']:.12e}\n\n")

        f.write("ESTIMATED ACCEL BIASES [m/s^2]\n")
        f.write("------------------------------\n")
        f.write(f"bax = {accel_bias['ax_bias']:.12e}\n")
        f.write(f"bay = {accel_bias['ay_bias']:.12e}\n")
        f.write(f"baz = {accel_bias['az_bias']:.12e}\n\n")

        f.write("CALIBRATION EQUATIONS\n")
        f.write("---------------------\n")
        f.write("gx_cal = gx_raw - bgx\n")
        f.write("gy_cal = gy_raw - bgy\n")
        f.write("gz_cal = gz_raw - bgz\n")
        f.write("ax_cal = ax_raw - bax\n")
        f.write("ay_cal = ay_raw - bay\n")
        f.write("az_cal = az_raw - baz\n\n")

        f.write("ALLAN CONSTANTS PER AXIS\n")
        f.write("------------------------\n")
        f.write("N = White noise / Random Walk coefficient\n")
        f.write("B = Bias Instability coefficient\n")
        f.write("K = Rate Random Walk coefficient\n\n")

        for axis_name, info in axis_results.items():
            c = info["constants"]
            f.write(f"{axis_name.upper()}\n")
            f.write(f"  mean                       = {info['mean']:.12e}\n")
            f.write(f"  std                        = {info['std']:.12e}\n")
            f.write(f"  N                          = {c['N']:.12e}\n")
            f.write(f"  B                          = {c['B']:.12e}\n")
            f.write(f"  K                          = {c['K']:.12e}\n")
            f.write(f"  tau_N_s                    = {c['tau_N_s']:.12e}\n")
            f.write(f"  tau_B_s                    = {c['tau_B_s']:.12e}\n")
            f.write(f"  tau_K_s                    = {c['tau_K_s']:.12e}\n")
            f.write(f"  slope_N_found              = {c['slope_N_found']:.12e}\n")
            f.write(f"  slope_B_found              = {c['slope_B_found']:.12e}\n")
            f.write(f"  slope_K_found              = {c['slope_K_found']:.12e}\n")
            f.write(f"  allan_dev_min              = {c['allan_dev_min']:.12e}\n")
            f.write(f"  tau_at_allan_dev_min_s     = {c['tau_at_allan_dev_min_s']:.12e}\n")
            f.write("\n")

        f.write("IMPORTANT NOTE\n")
        f.write("--------------\n")
        f.write("Biases above are deterministic offsets estimated from the static dataset.\n")
        f.write("B from Allan deviation is NOT the same as the offset to subtract.\n")
        f.write("Use the estimated means as fixed bias for calibration.\n")
        f.write("Use N, B, K for filter tuning and stochastic sensor modeling.\n")


def build_parser():
    parser = argparse.ArgumentParser(
        description="IMU Allan Variance + Bias from CSV + plots"
    )

    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_PATH,
        help=f"Ruta del CSV. Default: {DEFAULT_CSV_PATH}",
    )

    parser.add_argument("--gx-col", default=DEFAULT_GX_COL, help=f"Default: {DEFAULT_GX_COL}")
    parser.add_argument("--gy-col", default=DEFAULT_GY_COL, help=f"Default: {DEFAULT_GY_COL}")
    parser.add_argument("--gz-col", default=DEFAULT_GZ_COL, help=f"Default: {DEFAULT_GZ_COL}")
    parser.add_argument("--ax-col", default=DEFAULT_AX_COL, help=f"Default: {DEFAULT_AX_COL}")
    parser.add_argument("--ay-col", default=DEFAULT_AY_COL, help=f"Default: {DEFAULT_AY_COL}")
    parser.add_argument("--az-col", default=DEFAULT_AZ_COL, help=f"Default: {DEFAULT_AZ_COL}")
    parser.add_argument(
        "--manual-fit",
        action="store_true",
        help="Abre una herramienta interactiva para seleccionar manualmente tramos y calcular N/B/K.",
    )
    parser.add_argument(
        "--use-timestamp",
        action="store_true",
        default=DEFAULT_USE_TIMESTAMP,
        help=(
            "Usa una columna de timestamp en milisegundos para estimar fs. "
            f"Default: {DEFAULT_USE_TIMESTAMP}"
        ),
    )
    parser.add_argument(
        "--timestamp-col",
        default=DEFAULT_TIMESTAMP_COL,
        help=f"Nombre de la columna timestamp [ms]. Default: {DEFAULT_TIMESTAMP_COL}",
    )

    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_FS,
        help=f"Frecuencia de muestreo [Hz] si no se usa timestamp. Default: {DEFAULT_FS}",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta de salida donde se guardarán el TXT y los plots. Si no se indica, se autogenera.",
    )

    parser.add_argument(
        "--show-plots",
        action="store_true",
        default=DEFAULT_SHOW_PLOTS,
        help=f"Muestra plots por pantalla. Default: {DEFAULT_SHOW_PLOTS}",
    )

    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="No guardar plots en PNG.",
    )

    parser.add_argument(
        "--max-clusters",
        type=int,
        default=DEFAULT_MAX_CLUSTERS,
        help=f"Número máximo de clusters Allan. Default: {DEFAULT_MAX_CLUSTERS}",
    )

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"[ERROR] No existe el archivo: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] No se pudo leer el CSV: {e}")
        return

    colmap = {
        "gx": args.gx_col,
        "gy": args.gy_col,
        "gz": args.gz_col,
        "ax": args.ax_col,
        "ay": args.ay_col,
        "az": args.az_col,
    }

    try:
        if args.use_timestamp:
            if args.timestamp_col not in df.columns:
                raise ValueError(
                    f"La columna de timestamp '{args.timestamp_col}' no existe."
                )
            timestamp_ms = pd.to_numeric(
                df[args.timestamp_col], errors="coerce"
            ).to_numpy(dtype=np.float64)
            timestamp_ms = timestamp_ms[np.isfinite(timestamp_ms)]
            fs, dt_s = estimate_fs_from_timestamp_ms(timestamp_ms)
        else:
            fs = float(args.fs)
            if fs <= 0:
                raise ValueError("fs debe ser > 0")
            dt_s = 1.0 / fs
    except Exception as e:
        print(f"[ERROR] No se pudo determinar fs: {e}")
        return

    try:
        signals = {
            "gx": safe_numeric_column(df, args.gx_col),
            "gy": safe_numeric_column(df, args.gy_col),
            "gz": safe_numeric_column(df, args.gz_col),
            "ax": safe_numeric_column(df, args.ax_col),
            "ay": safe_numeric_column(df, args.ay_col),
            "az": safe_numeric_column(df, args.az_col),
        }
    except Exception as e:
        print(f"[ERROR] No se pudieron extraer las señales: {e}")
        return

    try:
        gyro_bias, accel_bias = compute_biases(signals)
    except Exception as e:
        print(f"[ERROR] No se pudieron calcular los bias: {e}")
        return

    try:
        axis_results = compute_axis_results(signals, fs, max_clusters=args.max_clusters)
    except Exception as e:
        print(f"[ERROR] Fallo al calcular Allan variance: {e}")
        return

    axis_units = {
        "gx": "rad/s",
        "gy": "rad/s",
        "gz": "rad/s",
        "ax": "m/s^2",
        "ay": "m/s^2",
        "az": "m/s^2",
    }

    manual_results = {}

    if args.manual_fit:
        try:
            print("\n========================================")
            print("MODO MANUAL ALLAN FIT")
            print("========================================")
            print("En cada ventana:")
            print("  - Click izquierdo: seleccionar 2 puntos")
            print("  - n: calcular N")
            print("  - b: calcular B")
            print("  - k: calcular K")
            print("  - c: limpiar selección")
            print("  - q: cerrar ventana")
            print("")

            for axis_name, info in axis_results.items():
                print(f"[INFO] Abriendo herramienta manual para eje {axis_name.upper()}...")
                manual_results[axis_name] = interactive_manual_fit(
                    tau=info["tau"],
                    allan_dev=info["allan_dev"],
                    axis_name=axis_name.upper(),
                    units=axis_units[axis_name],
                )
        except Exception as e:
            print(f"[ERROR] Fallo en modo manual: {e}")
            return

    if args.output.strip() == "":
        output_dir = csv_path.with_name(csv_path.stem + "_allan_output")
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_txt_path = output_dir / DEFAULT_TXT_FILENAME
    save_plots = not args.no_save_plots

    try:
        save_results_txt(
            output_path=output_txt_path,
            csv_path=csv_path,
            fs=fs,
            dt_s=dt_s,
            colmap=colmap,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
            axis_results=axis_results,
        )
    except Exception as e:
        print(f"[ERROR] No se pudo guardar el TXT: {e}")
        return

    try:
        for axis_name, info in axis_results.items():
            plot_path = None
            if save_plots:
                plot_path = output_dir / f"{axis_name}_allan.png"

            plot_allan_result(
                tau=info["tau"],
                allan_dev=info["allan_dev"],
                constants=info["constants"],
                axis_name=axis_name.upper(),
                units=axis_units[axis_name],
                output_path=plot_path,
                show_plot=args.show_plots,
            )
    except Exception as e:
        print(f"[ERROR] No se pudieron generar los plots: {e}")
        return

    print("\n========================================")
    print("RESUMEN")
    print("========================================")
    print(f"CSV: {csv_path}")
    print(f"Frecuencia usada: {fs:.6f} Hz")
    print(f"Periodo de muestreo: {dt_s:.9f} s")
    print(f"Carpeta de salida: {output_dir.resolve()}")
    print(f"TXT: {output_txt_path.resolve()}")

    if save_plots:
        print("Plots guardados en la misma carpeta de salida.")
    else:
        print("Plots no guardados.")

    print("\nBias gyro [rad/s]:")
    print(f"  bgx = {gyro_bias['gx_bias']:.6e}")
    print(f"  bgy = {gyro_bias['gy_bias']:.6e}")
    print(f"  bgz = {gyro_bias['gz_bias']:.6e}")

    print("\nBias accel [m/s^2]:")
    print(f"  bax = {accel_bias['ax_bias']:.6e}")
    print(f"  bay = {accel_bias['ay_bias']:.6e}")
    print(f"  baz = {accel_bias['az_bias']:.6e}")

    print("\nConstantes Allan automáticas:")
    for axis_name, info in axis_results.items():
        c = info["constants"]
        print(
            f"  {axis_name.upper()}: "
            f"N={c['N']:.6e}, B={c['B']:.6e}, K={c['K']:.6e}"
        )

    if args.manual_fit and manual_results:
        print("\nConstantes Allan manuales:")
        for axis_name, resdict in manual_results.items():
            print(f"  {axis_name.upper()}:")
            any_result = False
            for key in ("N", "B", "K"):
                r = resdict.get(key)
                if r is not None:
                    any_result = True
                    print(
                        f"    {key}={r.value:.6e} "
                        f"(tau: {r.tau1:.3e} -> {r.tau2:.3e}, "
                        f"slope medida: {r.slope_measured:.4f})"
                    )
            if not any_result:
                print("    No se guardó ningún ajuste manual.")
if __name__ == "__main__":
    main()