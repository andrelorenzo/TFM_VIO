#!/usr/bin/env python3
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plotea trayectoria desde un CSV usando x,y,z o gt_x,gt_y,gt_z."
    )
    parser.add_argument("csv_file", help="Ruta al CSV")
    parser.add_argument(
        "--series",
        choices=["est", "gt"],
        default="est",
        help="est -> x,y,z | gt -> gt_x,gt_y,gt_z",
    )
    parser.add_argument(
        "--topdown",
        choices=["xy", "xz", "yz"],
        default=None,
        help="Plano 2D a mostrar. Si no se indica, se muestra la trayectoria en 3D.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Titulo opcional de la figura",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Modo en vivo: relee el CSV periódicamente y actualiza la gráfica.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Periodo de refresco en segundos en modo --live. Default: 0.2",
    )
    return parser.parse_args()


def is_finite_number(value: str) -> bool:
    try:
        v = float(value)
        return math.isfinite(v)
    except Exception:
        return False


def load_series(csv_path: Path, series: str):
    if series == "gt":
        x_col, y_col, z_col = "posgt_m_x", "posgt_m_y", "posgt_m_z"
    else:
        x_col, y_col, z_col = "x", "y", "z"

    xs, ys, zs = [], [], []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("El CSV no tiene cabecera.")

        missing = [c for c in (x_col, y_col, z_col) if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Faltan columnas en el CSV: {missing}. "
                f"Cabecera encontrada: {reader.fieldnames}"
            )

        for row in reader:
            xv = row.get(x_col, "")
            yv = row.get(y_col, "")
            zv = row.get(z_col, "")

            # Si la última línea está a medio escribir, se ignora.
            if not (is_finite_number(xv) and is_finite_number(yv) and is_finite_number(zv)):
                continue

            xs.append(float(xv))
            ys.append(float(yv))
            zs.append(float(zv))

    return xs, ys, zs, (x_col, y_col, z_col)


def padded_limits(values, pad_ratio=0.05, min_pad=1e-3):
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * pad_ratio, min_pad)
    else:
        pad = max((vmax - vmin) * pad_ratio, min_pad)
    return vmin - pad, vmax + pad


def setup_2d_plot(label_a, label_b, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], linewidth=1.5, label="trayectoria")
    start_scatter = ax.scatter([], [], s=50, marker="o", label="inicio")
    end_scatter = ax.scatter([], [], s=50, marker="x", label="fin")

    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    return fig, ax, line, start_scatter, end_scatter


def setup_3d_plot(title):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    line, = ax.plot([], [], [], linewidth=1.5, label="trayectoria")
    start_scatter = ax.scatter([], [], [], s=50, marker="o", label="inicio")
    end_scatter = ax.scatter([], [], [], s=50, marker="x", label="fin")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()

    return fig, ax, line, start_scatter, end_scatter


def update_2d_plot(ax, line, start_scatter, end_scatter, a, b):
    line.set_data(a, b)

    start_scatter.set_offsets([[a[0], b[0]]])
    end_scatter.set_offsets([[a[-1], b[-1]]])

    ax.set_xlim(*padded_limits(a))
    ax.set_ylim(*padded_limits(b))


def update_3d_plot(ax, line, start_scatter, end_scatter, xs, ys, zs):
    line.set_data(xs, ys)
    line.set_3d_properties(zs)

    # Rehacer los scatters 3D es más robusto que intentar mutarlos.
    start_scatter.remove()
    end_scatter.remove()

    start_scatter = ax.scatter([xs[0]], [ys[0]], [zs[0]], s=50, marker="o", label="inicio")
    end_scatter = ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=50, marker="x", label="fin")

    ax.set_xlim(*padded_limits(xs))
    ax.set_ylim(*padded_limits(ys))
    ax.set_zlim(*padded_limits(zs))

    return start_scatter, end_scatter


def plot_static(xs, ys, zs, title, topdown):
    if topdown == "xy":
        fig, ax, line, start_scatter, end_scatter = setup_2d_plot("x", "y", title + " - plano XY")
        update_2d_plot(ax, line, start_scatter, end_scatter, xs, ys)
    elif topdown == "xz":
        fig, ax, line, start_scatter, end_scatter = setup_2d_plot("x", "z", title + " - plano XZ")
        update_2d_plot(ax, line, start_scatter, end_scatter, xs, zs)
    elif topdown == "yz":
        fig, ax, line, start_scatter, end_scatter = setup_2d_plot("y", "z", title + " - plano YZ")
        update_2d_plot(ax, line, start_scatter, end_scatter, ys, zs)
    else:
        fig, ax, line, start_scatter, end_scatter = setup_3d_plot(title)
        start_scatter, end_scatter = update_3d_plot(
            ax, line, start_scatter, end_scatter, xs, ys, zs
        )

    plt.tight_layout()
    plt.show()


def plot_live(csv_path: Path, series: str, title: str, topdown: str | None, interval: float):
    plt.ion()

    if topdown == "xy":
        fig, ax, line, start_scatter, end_scatter = setup_2d_plot("x", "y", title + " - plano XY")
    elif topdown == "xz":
        fig, ax, line, start_scatter, end_scatter = setup_2d_plot("x", "z", title + " - plano XZ")
    elif topdown == "yz":
        fig, ax, line, start_scatter, end_scatter = setup_2d_plot("y", "z", title + " - plano YZ")
    else:
        fig, ax, line, start_scatter, end_scatter = setup_3d_plot(title)

    plt.tight_layout()
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
                xs, ys, zs, _ = load_series(csv_path, series)
            except Exception:
                # Mientras el otro proceso escribe, puede haber instantes inconsistentes.
                plt.pause(interval)
                continue

            if not xs:
                plt.pause(interval)
                continue

            if topdown == "xy":
                update_2d_plot(ax, line, start_scatter, end_scatter, xs, ys)
            elif topdown == "xz":
                update_2d_plot(ax, line, start_scatter, end_scatter, xs, zs)
            elif topdown == "yz":
                update_2d_plot(ax, line, start_scatter, end_scatter, ys, zs)
            else:
                start_scatter, end_scatter = update_3d_plot(
                    ax, line, start_scatter, end_scatter, xs, ys, zs
                )

            fig.canvas.draw_idle()

        plt.pause(interval)

    plt.ioff()


def main():
    args = parse_args()
    csv_path = Path(args.csv_file)

    if not csv_path.exists():
        print(f"Error: no existe el archivo {csv_path}", file=sys.stderr)
        sys.exit(1)

    default_title = (
        f"Trayectoria {'GT' if args.series == 'gt' else 'estimada'}"
    )
    title = args.title or default_title

    if args.live:
        try:
            plot_live(csv_path, args.series, title, args.topdown, args.interval)
        except KeyboardInterrupt:
            pass
        return

    try:
        xs, ys, zs, cols = load_series(csv_path, args.series)
    except Exception as e:
        print(f"Error leyendo CSV: {e}", file=sys.stderr)
        sys.exit(2)

    if not xs:
        print(
            f"Error: no se encontraron muestras válidas en columnas {cols}",
            file=sys.stderr,
        )
        sys.exit(3)

    title = args.title or (
        f"Trayectoria {'GT' if args.series == 'gt' else 'estimada'} "
        f"({cols[0]}, {cols[1]}, {cols[2]})"
    )
    plot_static(xs, ys, zs, title, args.topdown)


if __name__ == "__main__":
    main()