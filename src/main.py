import copy
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# =========================
# Path settings
# =========================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "DemoICPPointClouds"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_point_cloud_file(filename):
    """
    从 data/DemoICPPointClouds 文件夹中读取点云文件。
    """
    file_path = DATA_DIR / filename

    if file_path.exists():
        return file_path

    raise FileNotFoundError(
        f"Cannot find {filename}. Please put it in: {DATA_DIR}"
    )


def save_matrix(matrix, save_path):
    """
    保存 4x4 变换矩阵。
    """
    np.savetxt(save_path, matrix, fmt="%.8f")


def save_text_report(
        save_path,
        source_path,
        target_path,
        ransac_result,
        icp_result,
        translation,
        euler,
        t_preprocess,
        t_ransac,
        t_icp,
        total_time,
):
    """
    保存完整运行结果到 txt 文件。
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Open3D 6D Pose Estimation Result\n")
        f.write("=" * 60 + "\n\n")

        f.write("Data\n")
        f.write("-" * 60 + "\n")
        f.write(f"Source point cloud: {source_path}\n")
        f.write(f"Target point cloud: {target_path}\n\n")

        f.write("RANSAC Result\n")
        f.write("-" * 60 + "\n")
        f.write(f"Fitness: {ransac_result.fitness:.6f}\n")
        f.write(f"Inlier RMSE: {ransac_result.inlier_rmse:.6f}\n")
        f.write("Transformation:\n")
        f.write(str(ransac_result.transformation))
        f.write("\n\n")

        f.write("ICP Result\n")
        f.write("-" * 60 + "\n")
        f.write(f"Fitness: {icp_result.fitness:.6f}\n")
        f.write(f"Inlier RMSE: {icp_result.inlier_rmse:.6f}\n")
        f.write("Transformation:\n")
        f.write(str(icp_result.transformation))
        f.write("\n\n")

        f.write("Estimated 6D Pose\n")
        f.write("-" * 60 + "\n")
        f.write(f"x: {translation[0]:.6f}\n")
        f.write(f"y: {translation[1]:.6f}\n")
        f.write(f"z: {translation[2]:.6f}\n")
        f.write(f"roll:  {euler[0]:.6f} deg\n")
        f.write(f"pitch: {euler[1]:.6f} deg\n")
        f.write(f"yaw:   {euler[2]:.6f} deg\n\n")

        f.write("Metrics\n")
        f.write("-" * 60 + "\n")
        f.write(f"Fitness: {icp_result.fitness:.6f}\n")
        f.write(f"Inlier RMSE: {icp_result.inlier_rmse:.6f}\n\n")

        f.write("Runtime\n")
        f.write("-" * 60 + "\n")
        f.write(f"Preprocess time: {t_preprocess:.4f} s\n")
        f.write(f"RANSAC time:     {t_ransac:.4f} s\n")
        f.write(f"ICP time:        {t_icp:.4f} s\n")
        f.write(f"Total time:      {total_time:.4f} s\n")


def draw_registration_result(
        source,
        target,
        transformation,
        title="Registration",
        save_image_path=None,
        show_window=True,
):
    """
    可视化点云配准结果，并可保存截图。
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)

    if show_window:
        o3d.visualization.draw_geometries(
            [source_temp, target_temp],
            window_name=title,
            width=1000,
            height=800
        )

    if save_image_path is not None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=title,
            width=1000,
            height=800,
            visible=False
        )

        vis.add_geometry(source_temp)
        vis.add_geometry(target_temp)

        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1, 1, 1])
        render_option.point_size = 2.0

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(save_image_path))
        vis.destroy_window()

        print(f"[INFO] Saved visualization image: {save_image_path}")


def preprocess_point_cloud(pcd, voxel_size):
    """
    点云预处理：
    1. 体素降采样
    2. 法向量估计
    3. FPFH 特征提取
    """
    print("[INFO] Downsample point cloud")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print("[INFO] Estimate normals")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=30
        )
    )

    radius_feature = voxel_size * 5
    print("[INFO] Compute FPFH feature")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100
        )
    )

    return pcd_down, pcd_fpfh


def execute_global_registration(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        voxel_size
):
    """
    基于 FPFH 特征和 RANSAC 的全局粗配准。
    """
    distance_threshold = voxel_size * 1.5

    print("[INFO] RANSAC global registration")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            100000,
            0.999
        ),
    )

    return result


def refine_registration(source, target, initial_transformation, voxel_size):
    """
    使用 ICP 进行精配准。
    """
    distance_threshold = voxel_size * 0.5

    print("[INFO] ICP refinement")

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2,
            max_nn=30
        )
    )
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2,
            max_nn=30
        )
    )

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return result


def matrix_to_xyz_rpy(transformation):
    """
    将 4x4 位姿变换矩阵转换为平移向量和欧拉角。
    """
    rotation_matrix = np.array(transformation[:3, :3], copy=True)
    translation = np.array(transformation[:3, 3], copy=True)

    euler = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)

    return translation, euler


def create_demo_pair():
    """
    读取真实点云数据。
    默认读取：
    data/DemoICPPointClouds/cloud_bin_0.pcd
    data/DemoICPPointClouds/cloud_bin_1.pcd
    """
    print("[INFO] Load real point cloud data")

    source_path = find_point_cloud_file("cloud_bin_0.pcd")
    target_path = find_point_cloud_file("cloud_bin_1.pcd")

    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))

    if source.is_empty():
        raise FileNotFoundError(f"Failed to load source point cloud: {source_path}")

    if target.is_empty():
        raise FileNotFoundError(f"Failed to load target point cloud: {target_path}")

    print(f"[INFO] Source path: {source_path}")
    print(f"[INFO] Target path: {target_path}")
    print(f"[INFO] Source points: {len(source.points)}")
    print(f"[INFO] Target points: {len(target.points)}")

    return source, target, source_path, target_path


def save_point_cloud_results(source, target, transformation):
    """
    保存配准后的点云结果。
    """
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)

    source_transformed.paint_uniform_color([1, 0.706, 0])

    target_colored = copy.deepcopy(target)
    target_colored.paint_uniform_color([0, 0.651, 0.929])

    merged = source_transformed + target_colored

    source_transformed_path = RESULTS_DIR / "source_transformed_icp.pcd"
    merged_path = RESULTS_DIR / "merged_after_icp.pcd"

    o3d.io.write_point_cloud(str(source_transformed_path), source_transformed)
    o3d.io.write_point_cloud(str(merged_path), merged)

    print(f"[INFO] Saved transformed source point cloud: {source_transformed_path}")
    print(f"[INFO] Saved merged point cloud: {merged_path}")


def main():
    voxel_size = 0.1

    source, target, source_path, target_path = create_demo_pair()

    print("\n========== Initial Transformation ==========")
    initial_transformation = np.eye(4)
    print(initial_transformation)

    print("\n[INFO] Visualize before registration")
    draw_registration_result(
        source,
        target,
        initial_transformation,
        "Before Registration",
        save_image_path=RESULTS_DIR / "before_registration.png",
        show_window=True
    )

    total_start = time.time()

    t0 = time.time()
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    t_preprocess = time.time() - t0

    t1 = time.time()
    result_ransac = execute_global_registration(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        voxel_size
    )
    t_ransac = time.time() - t1

    print("\n========== RANSAC Result ==========")
    print(result_ransac)
    print(result_ransac.transformation)

    save_matrix(
        result_ransac.transformation,
        RESULTS_DIR / "ransac_transformation.txt"
    )

    print("\n[INFO] Visualize after RANSAC")
    draw_registration_result(
        source,
        target,
        result_ransac.transformation,
        "After RANSAC",
        save_image_path=RESULTS_DIR / "after_ransac.png",
        show_window=True
    )

    t2 = time.time()
    result_icp = refine_registration(
        source,
        target,
        result_ransac.transformation,
        voxel_size
    )
    t_icp = time.time() - t2

    total_time = time.time() - total_start

    print("\n========== ICP Result ==========")
    print(result_icp)
    print(result_icp.transformation)

    save_matrix(
        result_icp.transformation,
        RESULTS_DIR / "icp_transformation.txt"
    )

    translation, euler = matrix_to_xyz_rpy(result_icp.transformation)

    print("\n========== Estimated 6D Pose ==========")
    print(f"x: {translation[0]:.6f}")
    print(f"y: {translation[1]:.6f}")
    print(f"z: {translation[2]:.6f}")
    print(f"roll:  {euler[0]:.6f} deg")
    print(f"pitch: {euler[1]:.6f} deg")
    print(f"yaw:   {euler[2]:.6f} deg")

    print("\n========== Metrics ==========")
    print(f"Fitness: {result_icp.fitness:.6f}")
    print(f"Inlier RMSE: {result_icp.inlier_rmse:.6f}")

    print("\n========== Runtime ==========")
    print(f"Preprocess time: {t_preprocess:.4f} s")
    print(f"RANSAC time:     {t_ransac:.4f} s")
    print(f"ICP time:        {t_icp:.4f} s")
    print(f"Total time:      {total_time:.4f} s")

    save_text_report(
        RESULTS_DIR / "registration_report.txt",
        source_path,
        target_path,
        result_ransac,
        result_icp,
        translation,
        euler,
        t_preprocess,
        t_ransac,
        t_icp,
        total_time,
        )

    save_point_cloud_results(
        source,
        target,
        result_icp.transformation
    )

    print("\n[INFO] Visualize after ICP")
    draw_registration_result(
        source,
        target,
        result_icp.transformation,
        "After ICP",
        save_image_path=RESULTS_DIR / "after_icp.png",
        show_window=True
    )

    print("\n========== Saved Results ==========")
    print(f"All results have been saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()