import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthetic_test import build_case, case_names


def parse_vec3(text: str) -> np.ndarray:
    vals = np.fromstring(text.replace(";", ",").replace(" ", ","), sep=",", dtype=float)
    if vals.size != 3:
        raise ValueError("Vector must have exactly 3 values, e.g. --p '0.1,0.0,-0.2'")
    return vals


def parse_mat3(text: str) -> np.ndarray:
    rows = [row.strip() for row in text.split(";") if row.strip()]
    if len(rows) == 3:
        data = [np.fromstring(row.replace(" ", ","), sep=",", dtype=float) for row in rows]
        if any(row.size != 3 for row in data):
            raise ValueError("Each matrix row must have 3 values")
        return np.vstack(data)

    vals = np.fromstring(text.replace(";", ",").replace(" ", ","), sep=",", dtype=float)
    if vals.size != 9:
        raise ValueError(
            "Matrix must have 9 values, e.g. --R '1,0,0;0,1,0;0,0,1'"
        )
    return vals.reshape(3, 3)


def rotation_error_deg(r_expected: np.ndarray, r_obtained: np.ndarray) -> float:
    r_err = r_expected.T @ r_obtained
    cos_ang = np.clip((np.trace(r_err) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def draw_frame(ax, r: np.ndarray, p: np.ndarray, scale: float, alpha: float, linewidth: float, prefix: str):
    colors = ["tab:red", "tab:green", "tab:blue"]
    labels = [f"{prefix} x", f"{prefix} y", f"{prefix} z"]
    for i, (color, label) in enumerate(zip(colors, labels)):
        axis = r[:, i]
        q = p + scale * axis
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
        )


def set_axes_equal(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radii = 0.5 * (maxs - mins)
    radius = max(float(np.max(radii)), 1e-3)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def fmt_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"


def fmt_mat(m: np.ndarray) -> str:
    return "\n".join("[" + ", ".join(f"{x:.6f}" for x in row) + "]" for row in m)


def main():
    parser = argparse.ArgumentParser(
        description="Plot synthetic expected 3D trajectory and overlay an obtained final pose."
    )
    parser.add_argument("--case", required=True, choices=case_names(), help="Synthetic case name")
    parser.add_argument("--R", required=True, help="Obtained final rotation, e.g. '1,0,0;0,1,0;0,0,1'")
    parser.add_argument("--p", required=True, help="Obtained final position, e.g. '0.1,0.0,-0.2'")
    parser.add_argument("--fs", type=float, default=200.0, help="Sampling frequency used by the generator")
    parser.add_argument("--n", type=int, default=20, help="Number of intervals for fixed-length cases")
    parser.add_argument("--axis-scale", type=float, default=-1.0, help="Axis size; negative means auto")
    parser.add_argument("--save", type=str, default="", help="Optional output image path")
    args = parser.parse_args()

    _, _, _, p_w, _, r_wb = build_case(args.case, fs=args.fs, n=args.n)
    p_expected = p_w[-1]
    r_expected = r_wb[-1]

    p_obtained = parse_vec3(args.p)
    r_obtained = parse_mat3(args.R)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(p_w[:, 0], p_w[:, 1], p_w[:, 2], color="black", linewidth=2.0, label="expected path")
    ax.scatter(p_w[0, 0], p_w[0, 1], p_w[0, 2], color="black", s=35, label="start")
    ax.scatter(p_expected[0], p_expected[1], p_expected[2], color="tab:orange", s=60, label="expected final p")
    ax.scatter(p_obtained[0], p_obtained[1], p_obtained[2], color="tab:purple", s=60, label="obtained final p")

    auto_scale = max(np.linalg.norm(p_expected), np.linalg.norm(p_obtained), 1.0) * 0.15
    axis_scale = auto_scale if args.axis_scale <= 0.0 else args.axis_scale

    draw_frame(ax, r_expected, p_expected, axis_scale, alpha=0.95, linewidth=2.5, prefix="expected")
    draw_frame(ax, r_obtained, p_obtained, axis_scale, alpha=0.55, linewidth=1.5, prefix="obtained")

    all_points = np.vstack([p_w, p_expected[None, :], p_obtained[None, :]])
    set_axes_equal(ax, all_points)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Synthetic Pose Compare: {args.case}")
    ax.legend(loc="upper left", fontsize=8)

    pos_err = float(np.linalg.norm(p_expected - p_obtained))
    rot_err = rotation_error_deg(r_expected, r_obtained)

    summary = (
        f"Expected p:\n{fmt_vec(p_expected)}\n\n"
        f"Obtained p:\n{fmt_vec(p_obtained)}\n\n"
        f"|dp| = {pos_err:.6f}\n"
        f"dR error = {rot_err:.6f} deg\n\n"
        f"Expected R:\n{fmt_mat(r_expected)}\n\n"
        f"Obtained R:\n{fmt_mat(r_obtained)}"
    )
    fig.text(
        0.68,
        0.50,
        summary,
        fontsize=9,
        family="monospace",
        va="center",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    plt.tight_layout(rect=[0.0, 0.0, 0.66, 1.0])

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    print("Expected final p:", p_expected)
    print("Obtained final p:", p_obtained)
    print("Position error norm:", pos_err)
    print("Rotation error deg:", rot_err)
    print("Expected final R:\n", r_expected)
    print("Obtained final R:\n", r_obtained)

    plt.show()


if __name__ == "__main__":
    main()
