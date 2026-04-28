#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibra una camara con tablero de ajedrez y guarda un YAML OpenCV.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="glob con imagenes de calibracion, por ejemplo data/*.jpg")
    src.add_argument("--video", help="video mp4 para extraer capturas de calibracion")
    src.add_argument("--camera-id", type=int, help="camara OpenCV en directo")
    parser.add_argument("--output", required=True, help="ruta de salida YAML")
    parser.add_argument("--cols", type=int, required=True, help="esquinas internas horizontales")
    parser.add_argument("--rows", type=int, required=True, help="esquinas internas verticales")
    parser.add_argument("--square-size", type=float, required=True, help="tamano de casilla en metros o unidad deseada")
    parser.add_argument("--max-frames", type=int, default=40, help="maximo de capturas para video/camara")
    parser.add_argument("--frame-step", type=int, default=15, help="cada cuantos frames intentar deteccion en video/camara")
    parser.add_argument("--show", action="store_true", help="mostrar detecciones mientras calibra")
    return parser.parse_args()


def make_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    grid *= square_size
    return grid


def detect_corners(gray: np.ndarray, board_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def collect_from_images(pattern: str, board_size: tuple[int, int], objp: np.ndarray, show: bool):
    obj_points, img_points = [], []
    image_size = None
    for path in sorted(Path().glob(pattern)):
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_corners(gray, board_size)
        if not ok:
            continue
        obj_points.append(objp.copy())
        img_points.append(corners)
        image_size = gray.shape[::-1]
        if show:
            view = img.copy()
            cv2.drawChessboardCorners(view, board_size, corners, ok)
            cv2.imshow("calibration", view)
            cv2.waitKey(150)
    return obj_points, img_points, image_size


def collect_from_stream(capture: cv2.VideoCapture, board_size: tuple[int, int], objp: np.ndarray, max_frames: int, frame_step: int, show: bool):
    obj_points, img_points = [], []
    image_size = None
    frame_idx = 0
    while len(obj_points) < max_frames:
        ok, img = capture.read()
        if not ok or img is None:
            break
        if frame_idx % max(frame_step, 1) != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = detect_corners(gray, board_size)
        view = img.copy()
        if found:
            obj_points.append(objp.copy())
            img_points.append(corners)
            image_size = gray.shape[::-1]
        if show:
            cv2.drawChessboardCorners(view, board_size, corners, found)
            cv2.putText(view, f"capturas={len(obj_points)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("calibration", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
        frame_idx += 1
    return obj_points, img_points, image_size


def save_yaml(path: str, image_size: tuple[int, int], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, rms: float):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"No se pudo abrir el YAML de salida: {path}")
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs.reshape(1, -1))
    fs.write("rms", float(rms))
    fs.release()


def main() -> None:
    args = parse_args()
    board_size = (args.cols, args.rows)
    objp = make_object_points(args.cols, args.rows, args.square_size)

    if args.images:
        obj_points, img_points, image_size = collect_from_images(args.images, board_size, objp, args.show)
    else:
        cap = cv2.VideoCapture(args.video if args.video else args.camera_id)
        if not cap.isOpened():
            raise SystemExit("No se pudo abrir la fuente de calibracion")
        obj_points, img_points, image_size = collect_from_stream(
            cap, board_size, objp, args.max_frames, args.frame_step, args.show
        )
        cap.release()

    if args.show:
        cv2.destroyAllWindows()

    if not obj_points or image_size is None:
        raise SystemExit("No se detectaron suficientes tableros para calibrar")

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    save_yaml(args.output, image_size, camera_matrix, dist_coeffs, rms)

    # print(f"Calibracion guardada en: {args.output}")
    print(f"RMS reprojection error: {rms:.6f}")
    print("camera_matrix:")
    print(camera_matrix)
    print("dist_coeffs:")
    print(dist_coeffs.reshape(1, -1))


if __name__ == "__main__":
    main()
