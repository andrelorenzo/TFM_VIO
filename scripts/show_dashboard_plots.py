#!/usr/bin/env python3
"""Selector de gráficas CSV exportadas por el dashboard de VISLAM."""

import argparse
import csv
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


PLOT_INFO = {
    "vio_traj": ("Trayectoria estimada VIO", "Posición X [m]", "Posición Z [m]"),
    "vis_traj": ("Trayectoria estimada solo con visión", "Posición X [m]", "Posición Z [m]"),
    "imu_traj": ("Trayectoria estimada solo con IMU", "Posición X [m]", "Posición Z [m]"),
    "height": ("Altura estimada y de referencia", "Tiempo [s]", "Altura Y [m]"),
    "imu_gyro": ("Velocidad angular del giroscopio", "Tiempo [s]", "Velocidad angular [rad/s]"),
    "imu_accel": ("Aceleración del acelerómetro", "Tiempo [s]", "Aceleración [m/s²]"),
    "vio_rpy": ("Orientación estimada VIO", "Tiempo [s]", "Ángulo [rad]"),
    "vis_rpy": ("Orientación estimada solo con visión", "Tiempo [s]", "Ángulo [rad]"),
    "imu_rpy": ("Orientación estimada solo con IMU", "Tiempo [s]", "Ángulo [rad]"),
    "preint_dpos": ("Posición IMU: incremento local y posición propagada", "Tiempo [s]", "Posición [m]"),
    "preint_dvel": ("Velocidad IMU: incremento local y velocidad propagada", "Tiempo [s]", "Velocidad [m/s]"),
    "state_velocity": ("Velocidad estimada del estado", "Tiempo [s]", "Velocidad [m/s]"),
    "planner_velocity_cmd": ("Consigna de velocidad antes y después del PID", "Tiempo [s]", "Velocidad lineal [m/s]"),
    "planner_yaw_cmd": ("Consigna de guiñada antes y después del PID", "Tiempo [s]", "Velocidad angular [rad/s]"),
    "da3_angle": ("Ángulo de evasión DA3", "Tiempo [s]", "Ángulo [rad]"),
    "da3_magnitude": ("Magnitud de evasión DA3", "Tiempo [s]", "Magnitud [adimensional]"),
    "da3_evade_flag": ("Activación de la maniobra de evasión DA3", "Tiempo [s]", "Estado de evasión [0/1]"),
    "da3_activation_metrics": ("Métricas de activación de evasión DA3", "Tiempo [s]", "Puntuación / proximidad [adimensional]"),
    "da3_occupancy_metrics": ("Métricas de ocupación de obstáculos DA3", "Tiempo [s]", "Proporción [0–1]"),
    "da3_depth_metrics": ("Percentiles de profundidad detectada por DA3", "Tiempo [s]", "Distancia [m]"),
    "da3_guidance_metrics": ("Métricas de guía de espacio libre DA3", "Tiempo [s]", "Puntuación / índice [adimensional]"),
}

REQUIRED_SERIES = {
    "preint_dpos": ("propagated pk x", "propagated pk y", "propagated pk z"),
    "preint_dvel": ("propagated vk x", "propagated vk y", "propagated vk z"),
}

PLOT_GROUPS = [
    ("Trayectorias", ("vio_traj", "vis_traj", "imu_traj")),
    ("Evasión y evitación DA3", tuple(key for key in PLOT_INFO if key.startswith("da3_"))),
    ("Preintegración IMU", ("preint_dpos", "preint_dvel")),
    ("IMU", ("imu_gyro", "imu_accel", "imu_rpy")),
    ("Orientación", ("vio_rpy", "vis_rpy")),
    ("Velocidad y control", ("state_velocity", "planner_velocity_cmd", "planner_yaw_cmd")),
    ("Altura y estado", ("height",)),
]


def describe_plot(stem: str):
    return PLOT_INFO.get(
        stem,
        (stem.replace("_", " ").capitalize(), "Tiempo [s]", "Valor [unidad por determinar]"),
    )


def load_plot(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "x" not in reader.fieldnames:
            raise ValueError(f"{path.name}: falta la columna x")
        x_values = []
        series = {name: [] for name in reader.fieldnames if name != "x"}
        for row in reader:
            try:
                x_values.append(float(row["x"]))
            except (TypeError, ValueError):
                continue
            for name in series:
                try:
                    value = float(row[name])
                except (TypeError, ValueError):
                    value = math.nan
                series[name].append(value)
    return x_values, series


def render_plot(path: Path, save_directory=None):
    x_values, series = load_plot(path)
    if not x_values:
        raise ValueError(f"{path.name}: no contiene muestras")
    required = REQUIRED_SERIES.get(path.stem, ())
    missing = [name for name in required if name not in series]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(
            f"{path}: faltan las series propagadas ({missing_names}). "
            "Este CSV es de una ejecuciÃ³n anterior. Vuelve a ejecutar VISLAM "
            "con la versiÃ³n actual y ciÃ©rralo normalmente para exportar las grÃ¡ficas."
        )
    title, x_label, y_label = describe_plot(path.stem)
    # El CSV guarda segundos; StateOut.ts_ms y el eje solicitado están en ms.
    if x_label == "Tiempo [s]":
        x_values = [value * 1000.0 for value in x_values]
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    lines = []
    for label, values in series.items():
        if any(math.isfinite(value) for value in values):
            line, = ax.plot(x_values, values, linewidth=2.4, label=label)
            lines.append(line)
    ax.set_title(title, fontsize=17)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)

    # if save_directory is None:
        # Controles de zoom independientes por eje. Se conserva el rango
        # # automático como punto de partida y se puede volver a él con Reset.
        # fig.subplots_adjust(bottom=0.27)
        # initial_xlim = ax.get_xlim()
        # initial_ylim = ax.get_ylim()
        # x_center = (initial_xlim[0] + initial_xlim[1]) / 2.0
        # y_center = (initial_ylim[0] + initial_ylim[1]) / 2.0
        # x_half = max((initial_xlim[1] - initial_xlim[0]) / 2.0, 1e-12)
        # y_half = max((initial_ylim[1] - initial_ylim[0]) / 2.0, 1e-12)
        # slider_x_ax = fig.add_axes([0.16, 0.15, 0.68, 0.025])
        # slider_y_ax = fig.add_axes([0.16, 0.09, 0.68, 0.025])
        # # slider_x = Slider(slider_x_ax, "Zoom X", 0.25, 10.0, valinit=1.0)
        # # slider_y = Slider(slider_y_ax, "Zoom Y", 0.25, 10.0, valinit=1.0)

        # def update_x(zoom):
        #     half = x_half / zoom
        #     ax.set_xlim(x_center - half, x_center + half)
        #     fig.canvas.draw_idle()

        # def update_y(zoom):
        #     half = y_half / zoom
        #     ax.set_ylim(y_center - half, y_center + half)
        #     fig.canvas.draw_idle()

        # slider_x.on_changed(update_x)
        # slider_y.on_changed(update_y)
        # reset_ax = fig.add_axes([0.86, 0.085, 0.1, 0.1])
        # reset_button = Button(reset_ax, "Reset")

        # def reset_zoom(_event):
        #     slider_x.reset()
        #     slider_y.reset()
        #     ax.set_xlim(initial_xlim)
        #     ax.set_ylim(initial_ylim)
        #     fig.canvas.draw_idle()

        # reset_button.on_clicked(reset_zoom)
    if lines:
        legend = ax.legend(fontsize=10, ncols=2)
        # Cada elemento de la leyenda se puede pulsar para ocultar/mostrar
        # su curva. Se marca el estado con transparencia en la propia leyenda.
        legend_lines = list(legend.get_lines())
        for legend_line, plot_line in zip(legend_lines, lines):
            legend_line.set_picker(6)

        line_by_legend = dict(zip(legend_lines, lines))

        def toggle_curve(event):
            legend_line = event.artist
            plot_line = line_by_legend.get(legend_line)
            if plot_line is None:
                return
            visible = not plot_line.get_visible()
            plot_line.set_visible(visible)
            legend_line.set_alpha(1.0 if visible else 0.25)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", toggle_curve)
    if save_directory is not None:
        fig.savefig(save_directory / f"{path.stem}.png", dpi=180, facecolor="white")
        plt.close(fig)
    else:
        plt.show()


def show_dashboard(files):
    root = tk.Tk()
    root.title("VISLAM - gráficas disponibles")
    root.geometry("620x760")
    root.minsize(460, 360)
    ttk.Label(root, text="Gráficas exportadas", font=("Segoe UI", 16, "bold")).pack(
        anchor="w", padx=18, pady=(16, 4)
    )
    ttk.Label(root, text="Elige una gráfica para abrirla en una ventana independiente.").pack(
        anchor="w", padx=18, pady=(0, 12)
    )

    container = ttk.Frame(root)
    container.pack(fill="both", expand=True, padx=18, pady=(0, 18))
    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    content = ttk.Frame(canvas)
    content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    by_stem = {path.stem: path for path in files}

    def open_plot(path):
        try:
            render_plot(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("No se pudo abrir la gráfica", str(error), parent=root)

    shown = set()
    for group_name, stems in PLOT_GROUPS:
        group_files = [by_stem[stem] for stem in stems if stem in by_stem]
        if not group_files:
            continue
        section = ttk.LabelFrame(content, text=group_name, padding=(10, 6))
        section.pack(fill="x", expand=True, pady=5)
        for path in group_files:
            title, _, _ = describe_plot(path.stem)
            ttk.Button(
                section,
                text=title,
                command=lambda selected=path: open_plot(selected),
            ).pack(fill="x", pady=2)
            shown.add(path.stem)

    other_files = [path for path in files if path.stem not in shown]
    if other_files:
        section = ttk.LabelFrame(content, text="Otras gráficas", padding=(10, 6))
        section.pack(fill="x", expand=True, pady=5)
        for path in other_files:
            title, _, _ = describe_plot(path.stem)
            ttk.Button(
                section,
                text=title,
                command=lambda selected=path: open_plot(selected),
            ).pack(fill="x", pady=2)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Abre un selector agrupado para las gráficas exportadas por VISLAM."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("output/plots_export"),
        help="Carpeta de CSV (por defecto: output/plots_export)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Guarda un PNG por gráfica en esta carpeta, sin abrir ventanas.",
    )
    args = parser.parse_args()
    files = sorted(args.directory.glob("*.csv"))
    if not files:
        parser.error(f"No se encontraron CSV en {args.directory}")
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        for path in files:
            try:
                render_plot(path, args.save)
            except (OSError, ValueError) as error:
                print(f"Aviso: {error}")
        print(f"Imágenes guardadas en: {args.save}")
    else:
        show_dashboard(files)


if __name__ == "__main__":
    main()
