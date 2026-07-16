#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from synthetic_test import case_names


def replace_yaml_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(?P<prefix>\s*{re.escape(key)}\s*:\s*).*$", re.MULTILINE)
    new_text, count = pattern.subn(rf"\g<prefix>{value}", text, count=1)
    if count != 1:
        raise RuntimeError(f"No se encontro la clave '{key}' en el YAML")
    return new_text


def build_runtime_config_text(base_text: str, exe_dir: Path, csv_path: Path, out_csv_path: Path) -> str:
    csv_rel = Path(os.path.relpath(csv_path, exe_dir)).as_posix()
    out_rel = Path(os.path.relpath(out_csv_path, exe_dir)).as_posix()

    text = base_text
    text = replace_yaml_value(text, "gen.input", f"\"{csv_rel}\"")
    text = replace_yaml_value(text, "gen.output", f"\"{out_rel}\"")

    overrides = {
        "gen.show": "false",
        "gen.color_on": "false",
        "gen.depth_on": "false",
        "gen.imu_on": "true",
        "da3.enabled": "false",
        "da3.show_window": "false",
        "ctrl.enabled": "false",
        "gen.plot_imu": "false",
        "gen.plot_tray": "false",
        "gen.plot_vis_tray": "false",
        "gen.plot_imu_tray": "false",
        "gen.plot_height": "false",
        "gen.plot_rpy": "false",
        "gen.plot_vis_rpy": "false",
        "gen.plot_imu_rpy": "false",
        "gen.plot_dpos": "false",
        "gen.plot_dvel": "false",
        "gen.plot_da3": "false",
    }

    for key, value in overrides.items():
        text = replace_yaml_value(text, key, value)

    return text


def load_last_pose(csv_path: Path) -> dict:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"El CSV de salida esta vacio: {csv_path}")
    return rows[-1]


def camera_rpy_to_matrix(roll_z: float, pitch_x: float, yaw_neg_y: float) -> list[list[float]]:
    yaw_y = -yaw_neg_y

    cz = math.cos(roll_z)
    sz = math.sin(roll_z)
    cx = math.cos(pitch_x)
    sx = math.sin(pitch_x)
    cy = math.cos(yaw_y)
    sy = math.sin(yaw_y)

    rz = [
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ]
    rx = [
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ]
    ry = [
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ]

    def mm(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [
            [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)
        ]

    return mm(mm(rz, rx), ry)


def matrix_to_cli_text(m: list[list[float]]) -> str:
    return ";".join(",".join(f"{v:.12g}" for v in row) for row in m)


def run_case(
    case_name: str,
    repo_root: Path,
    exe_path: Path,
    config_path: Path,
    base_config_text: str,
    csv_path: Path,
    out_csv_path: Path,
    out_png_path: Path,
    timeout_s: float,
) -> None:
    exe_dir = exe_path.parent
    runtime_text = build_runtime_config_text(base_config_text, exe_dir, csv_path, out_csv_path)
    config_path.write_text(runtime_text, encoding="utf-8")

    config_arg = Path(os.path.relpath(config_path, exe_dir)).as_posix()
    subprocess.run(
        [str(exe_path), str(config_arg)],
        cwd=str(exe_dir),
        check=True,
        timeout=timeout_s,
    )

    last_row = load_last_pose(out_csv_path)
    px = float(last_row["x"])
    py = float(last_row["y"])
    pz = float(last_row["z"])
    roll = float(last_row["roll"])
    pitch = float(last_row["pitch"])
    yaw = float(last_row["yaw"])
    rot = camera_rpy_to_matrix(roll, pitch, yaw)

    plot_env = os.environ.copy()
    plot_env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "plot_synthetic_pose_compare.py"),
            "--case",
            case_name,
            "--p",
            f"{px},{py},{pz}",
            "--R",
            matrix_to_cli_text(rot),
            "--save",
            str(out_png_path),
        ],
        cwd=str(repo_root),
        check=True,
        timeout=timeout_s,
        env=plot_env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta vio.exe para todos los casos sinteticos y guarda CSV + PNG por caso."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=case_names(),
        choices=case_names(),
        help="Lista de casos a ejecutar. Default: todos los sinteticos conocidos.",
    )
    parser.add_argument(
        "--exe",
        default="build/Release/vio.exe",
        help="Ruta al ejecutable principal. Default: build/Release/vio.exe",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al YAML de configuracion. Default: config/config.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="output/synthetic_suite",
        help="Directorio de salida para CSV, PNG y resumen.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout por ejecucion en segundos. Default: 180",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continua con el resto de casos aunque uno falle.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    exe_path = (repo_root / args.exe).resolve()
    config_path = (repo_root / args.config).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    csv_dir = out_dir / "csv"
    png_dir = out_dir / "png"
    csv_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    if not exe_path.exists():
        raise RuntimeError(f"No existe el ejecutable: {exe_path}")
    if not config_path.exists():
        raise RuntimeError(f"No existe el config.yaml: {config_path}")

    base_config_text = config_path.read_text(encoding="utf-8")
    summary_lines = []

    try:
        for case_name in args.cases:
            csv_path = repo_root / "bags" / "synz" / f"{case_name}.csv"
            out_csv_path = csv_dir / f"{case_name}.csv"
            out_png_path = png_dir / f"{case_name}.png"

            if not csv_path.exists():
                msg = f"[SKIP] No existe {csv_path}"
                print(msg)
                summary_lines.append(msg)
                continue

            print(f"[RUN ] {case_name}")
            try:
                run_case(
                    case_name=case_name,
                    repo_root=repo_root,
                    exe_path=exe_path,
                    config_path=config_path,
                    base_config_text=base_config_text,
                    csv_path=csv_path,
                    out_csv_path=out_csv_path,
                    out_png_path=out_png_path,
                    timeout_s=args.timeout,
                )
                msg = f"[ OK ] {case_name} -> {out_png_path}"
                print(msg)
                summary_lines.append(msg)
            except Exception as exc:
                msg = f"[FAIL] {case_name} -> {exc}"
                print(msg, file=sys.stderr)
                summary_lines.append(msg)
                if not args.keep_going:
                    raise
    finally:
        config_path.write_text(base_config_text, encoding="utf-8")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nResumen: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
