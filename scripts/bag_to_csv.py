#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

G_REF = np.array([0.0, -9.770184612169, 0.0], dtype=float)


def mean_triplets(samples):
    if not samples:
        return None
    arr = np.asarray(samples, dtype=float)
    return arr.mean(axis=0)


def collect_imu_rows(bag_path: Path):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    config.enable_stream(rs.stream.gyro)
    config.enable_stream(rs.stream.accel)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    gyro_buffer = []
    last_accel_ts_written = None
    rows = []

    gyro_count = 0
    accel_count = 0
    accel_skipped_duplicates = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break

            for fr in frames:
                motion = fr.as_motion_frame()
                if not motion:
                    continue

                data = motion.get_motion_data()
                ts = float(motion.get_timestamp())
                stream_type = motion.get_profile().stream_type()

                if stream_type == rs.stream.gyro:
                    gyro_count += 1
                    gyro_buffer.append((data.x, data.y, data.z))

                elif stream_type == rs.stream.accel:
                    accel_count += 1

                    if last_accel_ts_written is not None and ts == last_accel_ts_written:
                        accel_skipped_duplicates += 1
                        continue

                    gyro_mean = mean_triplets(gyro_buffer)
                    if gyro_mean is not None:
                        rows.append([
                            ts,
                            float(gyro_mean[0]), float(gyro_mean[1]), float(gyro_mean[2]),
                            float(data.x), float(data.y), float(data.z),
                        ])
                        last_accel_ts_written = ts

                    gyro_buffer.clear()
    finally:
        pipeline.stop()

    stats = {
        "gyro_count": gyro_count,
        "accel_count": accel_count,
        "accel_skipped_duplicates": accel_skipped_duplicates,
        "rows_written_raw": len(rows),
    }
    return rows, stats


def robust_outlier_mask(data, mad_k=6.0, accel_norm_k=6.0):
    if data.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if data.shape[0] < 5:
        return np.ones((data.shape[0],), dtype=bool)

    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    sigma = 1.4826 * np.maximum(mad, 1e-12)
    channel_ok = np.all(np.abs(data - median) <= mad_k * sigma, axis=1)

    accel = data[:, 4:7]
    accel_norm = np.linalg.norm(accel, axis=1)
    accel_norm_med = float(np.median(accel_norm))
    accel_norm_mad = float(np.median(np.abs(accel_norm - accel_norm_med)))
    accel_norm_sigma = max(1.4826 * accel_norm_mad, 1e-12)
    accel_norm_ok = np.abs(accel_norm - accel_norm_med) <= accel_norm_k * accel_norm_sigma

    return channel_ok & accel_norm_ok


def trim_time_window(rows, trim_start_ms=0.0, trim_end_ms=0.0):
    if not rows:
        return rows

    arr = np.asarray(rows, dtype=float)
    t0 = arr[0, 0]
    tf = arr[-1, 0]
    t_min = t0 + max(0.0, trim_start_ms)
    t_max = tf - max(0.0, trim_end_ms)
    if t_max <= t_min:
        return []
    keep = (arr[:, 0] >= t_min) & (arr[:, 0] <= t_max)
    return arr[keep].tolist()


def write_csv(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "gx", "gy", "gz", "ax", "ay", "az"])
        writer.writerows(rows)


def summarize_rows(rows):
    if not rows:
        return None

    arr = np.asarray(rows, dtype=float)
    gyro = arr[:, 1:4]
    accel = arr[:, 4:7]
    accel_norm = np.linalg.norm(accel, axis=1)

    summary = {
        "rows": arr.shape[0],
        "t0_ms": float(arr[0, 0]),
        "tf_ms": float(arr[-1, 0]),
        "span_ms": float(arr[-1, 0] - arr[0, 0]),
        "gyro_mean": gyro.mean(axis=0),
        "gyro_std": gyro.std(axis=0),
        "accel_mean": accel.mean(axis=0),
        "accel_std": accel.std(axis=0),
        "accel_norm_mean": float(accel_norm.mean()),
        "accel_norm_std": float(accel_norm.std()),
    }
    summary["gyro_bias_suggested"] = summary["gyro_mean"]
    summary["accel_bias_suggested"] = summary["accel_mean"] - G_REF
    return summary


def write_summary(summary_path: Path, bag_path: Path, raw_stats, raw_summary, filt_summary, removed_outliers):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        f.write(f"bag: {bag_path}\n")
        f.write(f"gyro_frames_read: {raw_stats['gyro_count']}\n")
        f.write(f"accel_frames_read: {raw_stats['accel_count']}\n")
        f.write(f"accel_duplicates_skipped: {raw_stats['accel_skipped_duplicates']}\n")
        f.write(f"raw_rows: {raw_stats['rows_written_raw']}\n")
        f.write(f"outliers_removed: {removed_outliers}\n")
        f.write("\n")

        def dump_block(name, summary):
            if summary is None:
                f.write(f"{name}: EMPTY\n\n")
                return

            f.write(f"{name}\n")
            f.write(f"  rows: {summary['rows']}\n")
            f.write(f"  t0_ms: {summary['t0_ms']:.6f}\n")
            f.write(f"  tf_ms: {summary['tf_ms']:.6f}\n")
            f.write(f"  span_ms: {summary['span_ms']:.6f}\n")
            f.write("  gyro_mean: " + " ".join(f"{x:.12e}" for x in summary["gyro_mean"]) + "\n")
            f.write("  gyro_std: " + " ".join(f"{x:.12e}" for x in summary["gyro_std"]) + "\n")
            f.write("  accel_mean: " + " ".join(f"{x:.12e}" for x in summary["accel_mean"]) + "\n")
            f.write("  accel_std: " + " ".join(f"{x:.12e}" for x in summary["accel_std"]) + "\n")
            f.write(f"  accel_norm_mean: {summary['accel_norm_mean']:.12e}\n")
            f.write(f"  accel_norm_std: {summary['accel_norm_std']:.12e}\n")
            f.write("  suggested imu.bg: " + " ".join(f"{x:.12e}" for x in summary["gyro_bias_suggested"]) + "\n")
            f.write("  suggested imu.ba: " + " ".join(f"{x:.12e}" for x in summary["accel_bias_suggested"]) + "\n")
            f.write("\n")

        dump_block("RAW", raw_summary)
        dump_block("FILTERED", filt_summary)


def export_imu_bag_to_csv(bag_path: str, csv_path: str, mad_k=6.0, accel_norm_k=6.0,
                          keep_raw=False, trim_start_ms=0.0, trim_end_ms=0.0) -> None:
    bag_path = Path(bag_path).expanduser().resolve()
    csv_path = Path(csv_path).expanduser().resolve()

    rows_raw, raw_stats = collect_imu_rows(bag_path)
    rows_trimmed = trim_time_window(rows_raw, trim_start_ms=trim_start_ms, trim_end_ms=trim_end_ms)

    raw_summary = summarize_rows(rows_trimmed)

    arr = np.asarray(rows_trimmed, dtype=float) if rows_trimmed else np.empty((0, 7), dtype=float)
    keep_mask = robust_outlier_mask(arr, mad_k=mad_k, accel_norm_k=accel_norm_k)
    rows_filtered = arr[keep_mask].tolist() if arr.size else []
    removed_outliers = int(arr.shape[0] - len(rows_filtered))
    filt_summary = summarize_rows(rows_filtered)

    write_csv(csv_path, rows_filtered)
    if keep_raw:
        write_csv(csv_path.with_name(csv_path.stem + "_raw.csv"), rows_trimmed)

    write_summary(
        csv_path.with_name(csv_path.stem + "_summary.txt"),
        bag_path,
        raw_stats=raw_stats,
        raw_summary=raw_summary,
        filt_summary=filt_summary,
        removed_outliers=removed_outliers,
    )

    print(f"CSV filtrado generado: {csv_path}")
    if keep_raw:
        print(f"CSV bruto generado: {csv_path.with_name(csv_path.stem + '_raw.csv')}")
    print(f"Resumen generado: {csv_path.with_name(csv_path.stem + '_summary.txt')}")
    print(f"Gyro frames leidos: {raw_stats['gyro_count']}")
    print(f"Accel frames leidos: {raw_stats['accel_count']}")
    print(f"Accel duplicados ignorados: {raw_stats['accel_skipped_duplicates']}")
    print(f"Filas brutas: {raw_stats['rows_written_raw']}")
    print(f"Filas tras recorte temporal: {len(rows_trimmed)}")
    print(f"Outliers eliminados: {removed_outliers}")
    print(f"Filas escritas: {len(rows_filtered)}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Exporta IMU de un bag de RealSense a CSV, con filtrado robusto de outliers y resumen de bias."
    )
    parser.add_argument("input_bag", type=str, help="Ruta al .bag")
    parser.add_argument("output_csv", type=str, help="Ruta del CSV de salida filtrado")
    parser.add_argument("--mad-k", type=float, default=6.0, help="Umbral robusto por canal en sigmas-MAD")
    parser.add_argument("--accel-norm-k", type=float, default=6.0, help="Umbral robusto sobre la norma de aceleracion")
    parser.add_argument("--trim-start-ms", type=float, default=0.0, help="Descarta este tiempo inicial del bag")
    parser.add_argument("--trim-end-ms", type=float, default=0.0, help="Descarta este tiempo final del bag")
    parser.add_argument("--keep-raw", action="store_true", help="Guarda tambien un CSV bruto antes del filtrado")
    return parser


def main():
    args = build_arg_parser().parse_args()
    export_imu_bag_to_csv(
        args.input_bag,
        args.output_csv,
        mad_k=args.mad_k,
        accel_norm_k=args.accel_norm_k,
        keep_raw=args.keep_raw,
        trim_start_ms=args.trim_start_ms,
        trim_end_ms=args.trim_end_ms,
    )


if __name__ == "__main__":
    main()
