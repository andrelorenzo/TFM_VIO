#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


AXES = ("x", "y", "z")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plotea gyro y/o acelerómetro desde un CSV, en modo estático o en vivo."
    )
    parser.add_argument("csv_file", help="Ruta al CSV")
    parser.add_argument(
        "--mode",
        choices=["raw", "cal", "both"],
        default="both",
        help="Qué señales mostrar: raw, cal o both",
    )
    parser.add_argument(
        "--sensor",
        choices=["gyro", "accel", "both"],
        default="both",
        help="Qué sensor mostrar: gyro, accel o both",
    )
    parser.add_argument("--time-col", default="timestamp", help="Nombre de la columna de tiempo")
    parser.add_argument("--gyro-raw-prefix", default="gyro_raw_", help="Prefijo gyro raw")
    parser.add_argument("--gyro-cal-prefix", default="gyro_cal_", help="Prefijo gyro calibrado")
    parser.add_argument("--acc-raw-prefix", default="acc_raw_", help="Prefijo accel raw")
    parser.add_argument("--acc-cal-prefix", default="acc_cal_", help="Prefijo accel calibrado")
    parser.add_argument("--title", default="IMU vs tiempo", help="Título de la figura")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Modo en vivo: relee el CSV periódicamente y actualiza la figura",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Periodo de refresco en segundos en modo live",
    )
    parser.add_argument(
        "--relative-time",
        action="store_true",
        help="Muestra t - t0 en el eje X",
    )
    return parser.parse_args()


def is_finite_number(value: str) -> bool:
    try:
        v = float(value)
        return math.isfinite(v)
    except Exception:
        return False


def padded_limits(values, pad_ratio=0.05, min_pad=1e-6):
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * pad_ratio, min_pad)
    else:
        pad = max((vmax - vmin) * pad_ratio, min_pad)
    return vmin - pad, vmax + pad


def build_columns(args):
    cols = {}
    if args.sensor in ("gyro", "both"):
        cols["gyro_raw"] = [args.gyro_raw_prefix + a for a in AXES]
        cols["gyro_cal"] = [args.gyro_cal_prefix + a for a in AXES]
    if args.sensor in ("accel", "both"):
        cols["acc_raw"] = [args.acc_raw_prefix + a for a in AXES]
        cols["acc_cal"] = [args.acc_cal_prefix + a for a in AXES]
    return cols


def selected_series_keys(args):
    keys = []
    if args.sensor in ("gyro", "both"):
        if args.mode in ("raw", "both"):
            keys.append("gyro_raw")
        if args.mode in ("cal", "both"):
            keys.append("gyro_cal")
    if args.sensor in ("accel", "both"):
        if args.mode in ("raw", "both"):
            keys.append("acc_raw")
        if args.mode in ("cal", "both"):
            keys.append("acc_cal")
    return keys


def load_imu_series(csv_path: Path, args):
    columns = build_columns(args)
    series_keys = selected_series_keys(args)

    ts = []
    data = {k: {a: [] for a in AXES} for k in series_keys}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("El CSV no tiene cabecera.")

        required = [args.time_col]
        for key in series_keys:
            required.extend(columns[key])

        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Faltan columnas en el CSV: {missing}. Cabecera encontrada: {reader.fieldnames}"
            )

        for row in reader:
            tv = row.get(args.time_col, "")
            if not is_finite_number(tv):
                continue

            valid = True
            for key in series_keys:
                for col in columns[key]:
                    if not is_finite_number(row.get(col, "")):
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            ts.append(float(tv))
            for key in series_keys:
                for axis, col in zip(AXES, columns[key]):
                    data[key][axis].append(float(row[col]))

    if not ts:
        raise ValueError("No se encontraron muestras válidas en el CSV.")

    if args.relative_time:
        t0 = ts[0]
        ts = [t - t0 for t in ts]

    return ts, data


def subplot_layout(sensor):
    if sensor == "both":
        return 2, 3
    return 1, 3


def sensor_rows(sensor):
    if sensor == "gyro":
        return ["gyro"]
    if sensor == "accel":
        return ["accel"]
    return ["gyro", "accel"]


def pretty_name(series_key):
    mapping = {
        "gyro_raw": "gyro raw",
        "gyro_cal": "gyro cal",
        "acc_raw": "acc raw",
        "acc_cal": "acc cal",
    }
    return mapping.get(series_key, series_key)


def setup_plot(args):
    rows, cols = subplot_layout(args.sensor)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 6 if rows == 1 else 8), sharex=True)

    if rows == 1:
        axes = [axes]

    time_label = args.time_col + (" (t-t0)" if args.relative_time else "")
    fig.suptitle(args.title)

    plot_handles = {}
    row_names = sensor_rows(args.sensor)

    for r, sensor_name in enumerate(row_names):
        unit = "rad/s" if sensor_name == "gyro" else "m/s²"
        for c, axis_name in enumerate(AXES):
            ax = axes[r][c]
            ax.grid(True)
            ax.set_title(f"{sensor_name} {axis_name}")
            ax.set_ylabel(unit)
            if r == len(row_names) - 1:
                ax.set_xlabel(time_label)

            plot_handles[(sensor_name, axis_name)] = {
                "ax": ax,
                "lines": {}
            }

    for sensor_name in row_names:
        for axis_name in AXES:
            ax = plot_handles[(sensor_name, axis_name)]["ax"]
            if sensor_name == "gyro":
                candidates = [k for k in selected_series_keys(args) if k.startswith("gyro_")]
            else:
                candidates = [k for k in selected_series_keys(args) if k.startswith("acc_")]

            for key in candidates:
                line, = ax.plot([], [], linewidth=1.5, label=pretty_name(key))
                plot_handles[(sensor_name, axis_name)]["lines"][key] = line
            if candidates:
                ax.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, plot_handles


def update_plot(plot_handles, ts, data, args):
    if not ts:
        return

    xlim = padded_limits(ts)
    row_names = sensor_rows(args.sensor)

    for sensor_name in row_names:
        for axis_name in AXES:
            handle = plot_handles[(sensor_name, axis_name)]
            ax = handle["ax"]
            values_for_limits = []

            for key, line in handle["lines"].items():
                series = data[key][axis_name]
                line.set_data(ts, series)
                values_for_limits.extend(series)

            ax.set_xlim(*xlim)
            if values_for_limits:
                ax.set_ylim(*padded_limits(values_for_limits))


def plot_static(csv_path: Path, args):
    ts, data = load_imu_series(csv_path, args)
    fig, plot_handles = setup_plot(args)
    update_plot(plot_handles, ts, data, args)
    plt.show()


def plot_live(csv_path: Path, args):
    plt.ion()
    fig, plot_handles = setup_plot(args)
    plt.show(block=False)

    last_mtime_ns = None
    last_size = None

    while plt.fignum_exists(fig.number):
        try:
            stat = csv_path.stat()
        except FileNotFoundError:
            plt.pause(args.interval)
            continue

        changed = (stat.st_mtime_ns != last_mtime_ns) or (stat.st_size != last_size)
        if changed:
            last_mtime_ns = stat.st_mtime_ns
            last_size = stat.st_size
            try:
                ts, data = load_imu_series(csv_path, args)
            except Exception:
                plt.pause(args.interval)
                continue

            update_plot(plot_handles, ts, data, args)
            fig.canvas.draw_idle()

        plt.pause(args.interval)

    plt.ioff()


def main():
    args = parse_args()
    csv_path = Path(args.csv_file)

    if not csv_path.exists():
        print(f"Error: no existe el archivo {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.live:
            plot_live(csv_path, args)
        else:
            plot_static(csv_path, args)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
