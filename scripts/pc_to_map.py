import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


input_cloud = r"matlab/pointclouds/aligned_global_cloud.ply"
output_dir = "matlab/floor_occupancy_output"

resolution = 0.05
margin = 0.50

voxel_size = 0.03
remove_outliers = True

use_ransac_floor = True
floor_ransac_threshold = 0.04
floor_max_tilt_deg = 25.0
floor_percentile_fallback = 2.0

ground_band = 0.07
min_obstacle_height = 0.12
max_obstacle_height = 2.20

min_points_free_cell = 2
min_points_occupied_cell = 2

inflate_obstacles = True
inflation_radius = 0.05

unknown_value = 50
free_value = 0
occupied_value = 100

os.makedirs(output_dir, exist_ok=True)


def inflate_binary(mask, radius_cells):
    if radius_cells <= 0:
        return mask

    ny, nx = mask.shape
    inflated = np.zeros_like(mask, dtype=bool)

    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx*dx + dy*dy > radius_cells*radius_cells:
                continue

            y_src0 = max(0, -dy)
            y_src1 = ny - max(0, dy)
            x_src0 = max(0, -dx)
            x_src1 = nx - max(0, dx)

            y_dst0 = max(0, dy)
            y_dst1 = ny - max(0, -dy)
            x_dst0 = max(0, dx)
            x_dst1 = nx - max(0, -dx)

            inflated[y_dst0:y_dst1, x_dst0:x_dst1] |= mask[y_src0:y_src1, x_src0:x_src1]

    return inflated


def compute_floor_reference(points):
    if not use_ransac_floor:
        floor_z = np.percentile(points[:, 2], floor_percentile_fallback)
        origin = np.array([0.0, 0.0, floor_z])
        normal = np.array([0.0, 0.0, 1.0])
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        return origin, normal, x_axis, y_axis, "percentile"

    work = o3d.geometry.PointCloud()
    work.points = o3d.utility.Vector3dVector(points)

    candidates = []
    cos_limit = np.cos(np.deg2rad(floor_max_tilt_deg))

    for _ in range(8):
        if len(work.points) < 1000:
            break

        plane_model, inliers = work.segment_plane(distance_threshold=floor_ransac_threshold, ransac_n=3, num_iterations=1500)

        if len(inliers) < 500:
            break

        local_points = np.asarray(work.points)
        plane_model = np.asarray(plane_model, dtype=float)

        normal = plane_model[:3]
        d = plane_model[3]
        norm_normal = np.linalg.norm(normal)

        if norm_normal < 1e-9:
            work = work.select_by_index(inliers, invert=True)
            continue

        normal = normal / norm_normal
        d = d / norm_normal

        if normal[2] < 0.0:
            normal = -normal
            d = -d

        is_horizontal = abs(normal[2]) > cos_limit

        if is_horizontal:
            inlier_points = local_points[inliers, :]
            median_z = np.median(inlier_points[:, 2])
            candidates.append((median_z, len(inliers), normal, d))

        work = work.select_by_index(inliers, invert=True)

    if len(candidates) == 0:
        floor_z = np.percentile(points[:, 2], floor_percentile_fallback)
        origin = np.array([0.0, 0.0, floor_z])
        normal = np.array([0.0, 0.0, 1.0])
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        return origin, normal, x_axis, y_axis, "percentile_fallback"

    candidates = sorted(candidates, key=lambda c: c[0])
    _, _, normal, d = candidates[0]

    origin = -d * normal

    x_axis = np.array([1.0, 0.0, 0.0])
    x_axis = x_axis - np.dot(x_axis, normal) * normal

    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([0.0, 1.0, 0.0])
        x_axis = x_axis - np.dot(x_axis, normal) * normal

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    return origin, normal, x_axis, y_axis, "ransac"


def project_to_floor_frame(points, origin, normal, x_axis, y_axis):
    rel = points - origin
    x_floor = rel @ x_axis
    y_floor = rel @ y_axis
    z_floor = rel @ normal
    return x_floor, y_floor, z_floor


pcd = o3d.io.read_point_cloud(input_cloud)

if len(pcd.points) == 0:
    raise RuntimeError("La nube de puntos está vacía o no se ha podido leer.")

pcd = pcd.voxel_down_sample(voxel_size)

if remove_outliers:
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

points = np.asarray(pcd.points)

origin, normal, x_axis, y_axis, floor_method = compute_floor_reference(points)
x_floor, y_floor, z_floor = project_to_floor_frame(points, origin, normal, x_axis, y_axis)

ground_mask = np.abs(z_floor) <= ground_band
obstacle_mask = (z_floor >= min_obstacle_height) & (z_floor <= max_obstacle_height)

x_min = np.floor((np.min(x_floor) - margin) / resolution) * resolution
x_max = np.ceil((np.max(x_floor) + margin) / resolution) * resolution
y_min = np.floor((np.min(y_floor) - margin) / resolution) * resolution
y_max = np.ceil((np.max(y_floor) + margin) / resolution) * resolution

nx = int(np.ceil((x_max - x_min) / resolution))
ny = int(np.ceil((y_max - y_min) / resolution))

free_count = np.zeros((ny, nx), dtype=np.uint16)
occupied_count = np.zeros((ny, nx), dtype=np.uint16)


def accumulate_cells(x, y, mask, grid):
    x_sel = x[mask]
    y_sel = y[mask]

    ix = np.floor((x_sel - x_min) / resolution).astype(np.int32)
    iy = np.floor((y_sel - y_min) / resolution).astype(np.int32)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    ix = ix[valid]
    iy = iy[valid]

    np.add.at(grid, (iy, ix), 1)


accumulate_cells(x_floor, y_floor, ground_mask, free_count)
accumulate_cells(x_floor, y_floor, obstacle_mask, occupied_count)

free_cells = free_count >= min_points_free_cell
occupied_cells = occupied_count >= min_points_occupied_cell

if inflate_obstacles:
    radius_cells = int(np.ceil(inflation_radius / resolution))
    occupied_cells = inflate_binary(occupied_cells, radius_cells)

occupancy_grid = np.full((ny, nx), unknown_value, dtype=np.uint8)
occupancy_grid[free_cells] = free_value
occupancy_grid[occupied_cells] = occupied_value

csv_file = os.path.join(output_dir, "occupancy_grid.csv")
png_file = os.path.join(output_dir, "occupancy_grid.png")
npy_file = os.path.join(output_dir, "occupancy_grid.npy")
metadata_file = os.path.join(output_dir, "occupancy_metadata.json")
figure_file = os.path.join(output_dir, "occupancy_grid_plot.png")

np.savetxt(csv_file, occupancy_grid, delimiter=",", fmt="%d")
np.save(npy_file, occupancy_grid)

image = np.zeros_like(occupancy_grid, dtype=np.uint8)
image[occupancy_grid == free_value] = 255
image[occupancy_grid == unknown_value] = 127
image[occupancy_grid == occupied_value] = 0

plt.imsave(png_file, np.flipud(image), cmap="gray", vmin=0, vmax=255)

metadata = {
    "resolution_m_per_cell": resolution,
    "x_min_m": float(x_min),
    "x_max_m": float(x_max),
    "y_min_m": float(y_min),
    "y_max_m": float(y_max),
    "nx": int(nx),
    "ny": int(ny),
    "free_value": int(free_value),
    "unknown_value": int(unknown_value),
    "occupied_value": int(occupied_value),
    "floor_method": floor_method,
    "floor_origin": origin.tolist(),
    "floor_normal": normal.tolist(),
    "x_axis": x_axis.tolist(),
    "y_axis": y_axis.tolist(),
    "ground_band_m": ground_band,
    "min_obstacle_height_m": min_obstacle_height,
    "max_obstacle_height_m": max_obstacle_height,
    "inflation_radius_m": inflation_radius if inflate_obstacles else 0.0
}

with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)

plt.figure(figsize=(9, 8))
plt.imshow(occupancy_grid, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="gray_r", vmin=0, vmax=100, aspect="equal")
plt.xlabel("X suelo [m]")
plt.ylabel("Y suelo [m]")
plt.title("Mapa de ocupación 2D desde nube de puntos")
plt.colorbar(label="0 libre | 50 desconocido | 100 ocupado")
plt.grid()
plt.savefig(figure_file, dpi=200)
plt.show()

print("Mapa de ocupación generado.")
print(f"Método de suelo: {floor_method}")
print(f"Puntos totales usados: {points.shape[0]}")
print(f"Puntos de suelo: {np.sum(ground_mask)}")
print(f"Puntos de obstáculo: {np.sum(obstacle_mask)}")
print(f"Tamaño del mapa: {ny} x {nx} celdas")
print(f"Resolución: {resolution} m/celda")
print(f"CSV: {csv_file}")
print(f"PNG: {png_file}")
print(f"NPY: {npy_file}")
print(f"Metadata: {metadata_file}")