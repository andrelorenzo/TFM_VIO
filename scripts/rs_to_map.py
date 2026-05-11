import os
import copy
import numpy as np
import pyrealsense2 as rs
import open3d as o3d


bag_file = r"bags/FastStraightLong_Full.bag"

output_dir = "matlab/pointclouds"

frame_step = 2
max_frames = 4000
max_depth_m = 3.0

voxel_size = 0.04
icp_threshold_coarse = 0.35
icp_threshold_fine = 0.12

min_points_per_frame = 800
min_icp_fitness = 0.12
max_icp_rmse = 0.25

save_map_every = 20
visualize_result = True

os.makedirs(output_dir, exist_ok=True)


def depth_frame_to_pointcloud(depth_frame, pc, max_depth_m):
    points = pc.calculate(depth_frame)
    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3).copy()

    valid = np.isfinite(vertices).all(axis=1)
    valid = valid & (vertices[:, 2] > 0.0)
    valid = valid & (vertices[:, 2] < max_depth_m)

    vertices = vertices[valid, :]

    points_uav = np.zeros_like(vertices)
    points_uav[:, 0] = vertices[:, 2]
    points_uav[:, 1] = -vertices[:, 0]
    points_uav[:, 2] = -vertices[:, 1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_uav.astype(np.float64))

    return pcd


def preprocess_pointcloud(pcd, voxel_size):
    pcd = pcd.voxel_down_sample(voxel_size)

    if len(pcd.points) == 0:
        return pcd

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0*voxel_size, max_nn=30))

    return pcd


def register_icp(source, target, voxel_size, icp_threshold_coarse, icp_threshold_fine):
    source_icp = copy.deepcopy(source)
    target_icp = copy.deepcopy(target)

    source_icp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0*voxel_size, max_nn=30))
    target_icp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0*voxel_size, max_nn=30))

    T_init = np.eye(4)

    reg_coarse = o3d.pipelines.registration.registration_icp(source_icp, target_icp, icp_threshold_coarse, T_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    reg_fine = o3d.pipelines.registration.registration_icp(source_icp, target_icp, icp_threshold_fine, reg_coarse.transformation, o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return reg_fine


pipeline = rs.pipeline()
config = rs.config()

rs.config.enable_device_from_file(config, bag_file, repeat_playback=False)
config.enable_stream(rs.stream.depth)

profile = pipeline.start(config)

playback = profile.get_device().as_playback()
playback.set_real_time(False)

pc = rs.pointcloud()

decimation_filter = rs.decimation_filter()
spatial_filter = rs.spatial_filter()
temporal_filter = rs.temporal_filter()
hole_filter = rs.hole_filling_filter()

global_map = o3d.geometry.PointCloud()

prev_pcd = None
T_global_prev = np.eye(4)

trajectory = []

frame_id = 0
used_frames = 0
accepted_frames = 0
rejected_frames = 0

try:
    while True:
        try:
            frames = pipeline.wait_for_frames(5000)
        except RuntimeError:
            break

        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        frame_id = frame_id + 1

        depth_frame = decimation_filter.process(depth_frame).as_depth_frame()
        depth_frame = spatial_filter.process(depth_frame).as_depth_frame()
        depth_frame = temporal_filter.process(depth_frame).as_depth_frame()
        depth_frame = hole_filter.process(depth_frame).as_depth_frame()

        if frame_id % frame_step != 0:
            continue

        pcd = depth_frame_to_pointcloud(depth_frame, pc, max_depth_m)
        pcd = preprocess_pointcloud(pcd, voxel_size)

        if len(pcd.points) < min_points_per_frame:
            print(f"Frame {frame_id} descartado: pocos puntos válidos.")
            continue

        used_frames = used_frames + 1

        if prev_pcd is None:
            T_global_current = np.eye(4)

            pcd_global = copy.deepcopy(pcd)
            pcd_global.transform(T_global_current)

            global_map += pcd_global

            prev_pcd = copy.deepcopy(pcd)
            T_global_prev = T_global_current

            trajectory.append([frame_id, T_global_current[0, 3], T_global_current[1, 3], T_global_current[2, 3]])

            accepted_frames = accepted_frames + 1

            print(f"Frame {frame_id} usado como referencia inicial.")

        else:
            reg = register_icp(pcd, prev_pcd, voxel_size, icp_threshold_coarse, icp_threshold_fine)

            if reg.fitness < min_icp_fitness or reg.inlier_rmse > max_icp_rmse:
                rejected_frames = rejected_frames + 1
                print(f"Frame {frame_id} rechazado por ICP: fitness={reg.fitness:.3f}, rmse={reg.inlier_rmse:.3f}")
                continue

            T_prev_current = reg.transformation
            T_global_current = T_global_prev @ T_prev_current

            pcd_global = copy.deepcopy(pcd)
            pcd_global.transform(T_global_current)

            global_map += pcd_global

            prev_pcd = copy.deepcopy(pcd)
            T_global_prev = T_global_current

            trajectory.append([frame_id, T_global_current[0, 3], T_global_current[1, 3], T_global_current[2, 3]])

            accepted_frames = accepted_frames + 1

            print(f"Frame {frame_id} alineado: fitness={reg.fitness:.3f}, rmse={reg.inlier_rmse:.3f}, puntos mapa={len(global_map.points)}")

        if accepted_frames % save_map_every == 0:
            global_map = global_map.voxel_down_sample(voxel_size)

        if accepted_frames >= max_frames:
            break

finally:
    pipeline.stop()


print("Filtrando mapa final...")

global_map = global_map.voxel_down_sample(voxel_size)

if len(global_map.points) > 100:
    global_map, _ = global_map.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

ply_file = os.path.join(output_dir, "aligned_global_cloud.ply")
pcd_file = os.path.join(output_dir, "aligned_global_cloud.pcd")
xyz_file = os.path.join(output_dir, "aligned_global_cloud.xyz")
traj_file = os.path.join(output_dir, "estimated_trajectory.csv")

o3d.io.write_point_cloud(ply_file, global_map)
o3d.io.write_point_cloud(pcd_file, global_map)
np.savetxt(xyz_file, np.asarray(global_map.points), fmt="%.6f")
np.savetxt(traj_file, np.asarray(trajectory), delimiter=",", header="frame,x_m,y_m,z_m", comments="", fmt="%.6f")

print("")
print("Proceso terminado.")
print(f"Frames usados: {used_frames}")
print(f"Frames aceptados: {accepted_frames}")
print(f"Frames rechazados: {rejected_frames}")
print(f"Puntos finales: {len(global_map.points)}")
print(f"PLY guardado en: {ply_file}")
print(f"PCD guardado en: {pcd_file}")
print(f"XYZ guardado en: {xyz_file}")
print(f"Trayectoria estimada guardada en: {traj_file}")

if visualize_result:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([global_map, frame])