#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plotea roll, pitch y yaw respecto al tiempo desde un CSV."
    )
    parser.add_argument("csv_file", help="Ruta al CSV")
    parser.add_argument("--time-col", default="timestamp", help="Nombre de la columna de tiempo")
    parser.add_argument("--roll-col", default="roll_deg", help="Nombre de la columna de roll")
    parser.add_argument("--pitch-col", default="pitch_deg", help="Nombre de la columna de pitch")
    parser.add_argument("--yaw-col", default="yaw_deg", help="Nombre de la columna de yaw")
    parser.add_argument("--title", default="RPY vs tiempo", help="Título de la figura")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Modo en vivo: relee el CSV periódicamente y actualiza la gráfica",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Periodo de refresco en segundos en modo --live. Default: 0.2",
    )
    return parser.parse_args()


def is_finite_number(value: str) -> bool:
    try:
        v = float(value)
        return math.isfinite(v)
    except Exception:
        return False


def load_rpy_series(csv_path: Path, time_col: str, roll_col: str, pitch_col: str, yaw_col: str):
    ts, rolls, pitches, yaws = [], [], [], []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("El CSV no tiene cabecera.")

        required = [time_col, roll_col, pitch_col, yaw_col]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Faltan columnas en el CSV: {missing}. "
                f"Cabecera encontrada: {reader.fieldnames}"
            )

        for row in reader:
            tv = row.get(time_col, "")
            rv = row.get(roll_col, "")
            pv = row.get(pitch_col, "")
            yv = row.get(yaw_col, "")

            # Ignora filas incompletas o a medio escribir
            if not (
                is_finite_number(tv)
                and is_finite_number(rv)
                and is_finite_number(pv)
                and is_finite_number(yv)
            ):
                continue

            ts.append(float(tv))
            rolls.append(float(rv))
            pitches.append(float(pv))
            yaws.append(float(yv))

    if not ts:
        raise ValueError(
            f"No se encontraron muestras válidas en columnas "
            f"{time_col}, {roll_col}, {pitch_col}, {yaw_col}."
        )

    return ts, rolls, pitches, yaws


def padded_limits(values, pad_ratio=0.05, min_pad=1e-6):
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * pad_ratio, min_pad)
    else:
        pad = max((vmax - vmin) * pad_ratio, min_pad)
    return vmin - pad, vmax + pad


def setup_plot(title: str, time_label: str, roll_label: str, pitch_label: str, yaw_label: str):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    lines = []
    labels = [roll_label, pitch_label, yaw_label]

    for ax, label in zip(axes, labels):
        line, = ax.plot([], [], linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True)
        lines.append(line)

    axes[0].set_title(title)
    axes[2].set_xlabel(time_label)

    plt.tight_layout()
    return fig, axes, lines


def update_plot(axes, lines, ts, rolls, pitches, yaws):
    series = [rolls, pitches, yaws]

    for ax, line, values in zip(axes, lines, series):
        line.set_data(ts, values)
        ax.set_xlim(*padded_limits(ts))
        ax.set_ylim(*padded_limits(values))


def plot_static(ts, rolls, pitches, yaws, title, time_col, roll_col, pitch_col, yaw_col):
    fig, axes, lines = setup_plot(title, time_col, roll_col, pitch_col, yaw_col)
    update_plot(axes, lines, ts, rolls, pitches, yaws)
    plt.show()


def plot_live(csv_path: Path, title, time_col, roll_col, pitch_col, yaw_col, interval):
    plt.ion()
    fig, axes, lines = setup_plot(title, time_col, roll_col, pitch_col, yaw_col)
    plt.show(block=False)

    last_mtime_ns = None
    last_size = None

    while plt.fignum_exists(fig.number):
        try:
            stat = csv_path.stat()
        except FileNotFoundError:
            plt.pause(interval)
            continue

        changed = (stat.st_mtime_ns != last_mtime_ns) or (stat.st_size != last_size)
        if changed:
            last_mtime_ns = stat.st_mtime_ns
            last_size = stat.st_size

            try:
                ts, rolls, pitches, yaws = load_rpy_series(
                    csv_path, time_col, roll_col, pitch_col, yaw_col
                )
            except Exception:
                plt.pause(interval)
                continue

            update_plot(axes, lines, ts, rolls, pitches, yaws)
            fig.canvas.draw_idle()

        plt.pause(interval)

    plt.ioff()


def main():
    args = parse_args()
    csv_path = Path(args.csv_file)

    if not csv_path.exists():
        print(f"Error: no existe el archivo {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.live:
        try:
            plot_live(
                csv_path=csv_path,
                title=args.title,
                time_col=args.time_col,
                roll_col=args.roll_col,
                pitch_col=args.pitch_col,
                yaw_col=args.yaw_col,
                interval=args.interval,
            )
        except KeyboardInterrupt:
            pass
        return

    try:
        ts, rolls, pitches, yaws = load_rpy_series(
            csv_path, args.time_col, args.roll_col, args.pitch_col, args.yaw_col
        )
    except Exception as e:
        print(f"Error leyendo CSV: {e}", file=sys.stderr)
        sys.exit(2)

    plot_static(
        ts, rolls, pitches, yaws,
        title=args.title,
        time_col=args.time_col,
        roll_col=args.roll_col,
        pitch_col=args.pitch_col,
        yaw_col=args.yaw_col,
    )


if __name__ == "__main__":
    main()