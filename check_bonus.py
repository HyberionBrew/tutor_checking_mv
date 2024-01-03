import sys
import utility_functions as utils
from pathlib import Path
import open3d as o3d
import numpy as np

if __name__ == '__main__':
    student_name = sys.argv[1]
    # Load the student's solution
    solution_dir = Path(__file__).parent.joinpath('solution')
    utils.modify_sys_path(add_path=solution_dir)
    #from helper_functions import plot_clustering_results, silhouette_score
    from clustering import *
    from helper_functions import *
    utils.modify_sys_path(remove_path=solution_dir)

    # Load student's module
    student_module_path = Path(__file__).parent.joinpath(f'students/{student_name}/fit_plane.py')
    utils.modify_sys_path(add_path=student_module_path.parent)
    student_module =  utils.load_module(student_module_path, "student_clustering")
    utils.modify_sys_path(remove_path=student_module_path.parent)

    import sklearn.metrics as skmetrics
    import sklearn.cluster as skcluster

    # Selects which single-plane file to use
    pointcloud_idx = 0

    # Pick which clustering algorithm to apply:
    use_kmeans = False
    use_iterative_kmeans = False
    use_gmeans = False
    use_dbscan = False
    tutor_check = True

    # RANSAC parameters:
    confidence = 0.85
    inlier_threshold = 0.015  # Might need to be adapted, depending on how you implement fit_plane

    # Downsampling parameters:
    use_voxel_downsampling = True
    voxel_size = 0.01
    uniform_every_k_points = 10

    # Clustering Parameters
    kmeans_n_clusters = 6
    kmeans_iterations = 25
    max_singlerun_iterations = 100
    iterative_kmeans_max_clusters = 10
    gmeans_tolerance = 10
    dbscan_eps = 0.05
    dbscan_min_points = 15
    debug_output = True

    # Read Pointcloud
    current_path = Path(__file__).parent
    pcd = o3d.io.read_point_cloud("solution/pointclouds/image00" + str(pointcloud_idx) + ".pcd",
                                  remove_nan_points=True, remove_infinite_points=True)
    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Down-sample the loaded point cloud to reduce computation time
    if use_voxel_downsampling:
        pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        pcd_sampled = pcd.uniform_down_sample(uniform_every_k_points)

    # Apply your own plane-fitting algorithm
    plane_model, best_inliers = student_module.fit_plane(pcd=pcd_sampled,
                                          confidence=confidence,
                                          inlier_threshold=inlier_threshold)
    
    inlier_indices = np.nonzero(best_inliers)[0]

    # Alternatively use the built-in function of Open3D
    #plane_model, inlier_indices = pcd_sampled.segment_plane(distance_threshold=inlier_threshold,
    #                                                        ransac_n=3,
    #                                                        num_iterations=500)

    # Convert the inlier indices to a Boolean mask for the pointcloud
    best_inliers = np.full(shape=len(pcd_sampled.points, ), fill_value=False, dtype=bool)
    best_inliers[inlier_indices] = True

    # Store points without plane in scene_pcd
    scene_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)

    # Plot detected plane and remaining pointcloud
    if debug_output:
        plot_dominant_plane(pcd_sampled, best_inliers, plane_model)
        o3d.visualization.draw_geometries([scene_pcd])

    # Convert to NumPy array
    points = np.asarray(scene_pcd.points, dtype=np.float32)