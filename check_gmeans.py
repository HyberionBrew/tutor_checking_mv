import sys
import utility_functions as utils
from pathlib import Path
import open3d as o3d
import numpy as np
import time

def are_clusters_equivalent(labels1, labels2):
    if len(labels1) != len(labels2):
        return False

    # Create a mapping of labels to sets of indices (points in clusters)
    clusters1 = {label: set(np.where(labels1 == label)[0]) for label in np.unique(labels1)}
    clusters2 = {label: set(np.where(labels2 == label)[0]) for label in np.unique(labels2)}

    # Convert the sets of indices into a list of sets for easier comparison
    cluster_sets1 = list(clusters1.values())
    cluster_sets2 = list(clusters2.values())

    # Check if every set in cluster_sets1 has an equivalent set in cluster_sets2
    for cluster_set1 in cluster_sets1:
        if not any(cluster_set1 == cluster_set2 for cluster_set2 in cluster_sets2):
            return False

    return True

if __name__ == '__main__':
    student_name = sys.argv[1]
    # Load the student's solution
    solution_dir = Path(__file__).parent.joinpath('solution')
    utils.modify_sys_path(add_path=solution_dir)
    from helper_functions import plot_clustering_results, silhouette_score
    from clustering import *
    from helper_functions import *
    utils.modify_sys_path(remove_path=solution_dir)

    # Load student's module
    student_module_path = Path(__file__).parent.joinpath(f'students/{student_name}/clustering.py')
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
    


    # Alternatively use the built-in function of Open3D
    plane_model, inlier_indices = pcd_sampled.segment_plane(distance_threshold=inlier_threshold,
                                                            ransac_n=3,
                                                            num_iterations=500)

    # Convert the inlier indices to a Boolean mask for the pointcloud
    best_inliers = np.full(shape=len(pcd_sampled.points, ), fill_value=False, dtype=bool)
    best_inliers[inlier_indices] = True

    # Store points without plane in scene_pcd
    scene_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)

    # Plot detected plane and remaining pointcloud
    #if debug_output:
    #    plot_dominant_plane(pcd_sampled, best_inliers, plane_model)
    #    o3d.visualization.draw_geometries([scene_pcd])
    # Convert to NumPy array
    points = np.asarray(scene_pcd.points, dtype=np.float32)

    # Check GMEANS:
        # Check if it can handle empty clusters:
    try:
        gmeans_student_centers_low_tolerance, \
        gmeans_student_labels_low_tolerance = student_module.gmeans(points,
                                                        tolerance=.1,  # This usually produces empty clusters
                                                        max_singlerun_iterations=max_singlerun_iterations)
    except:
        print("Student Program Crashes at G-means with low tolerance. "
                "Might be due to not checking if there are enough points in a cluster to calculate np.cov")

    # Check if it procudes the same output as the solution by Matthias
    start = time.time()
    gmeans_student_centers, gmeans_student_labels = student_module.gmeans(points,
                                                            tolerance=gmeans_tolerance,
                                                            max_singlerun_iterations=max_singlerun_iterations)
    stop = time.time()
    print("Time for Gmeans: ", stop - start)
    plot_clustering_results(scene_pcd,
                        gmeans_student_labels,
                        "G-means student")
    gmeans_matthias_centers, gmeans_matthias_labels = gmeans(points,
                                                            tolerance=gmeans_tolerance,
                                                            max_singlerun_iterations=max_singlerun_iterations)

    plot_clustering_results(scene_pcd,
                    gmeans_matthias_labels,
                    "G-means student")
    
    print(f"The Labels of the implemented gmeans and matthias version are the same: "
            f"{np.all(gmeans_student_labels==gmeans_matthias_labels)}")
    #print(gmeans_matthias_labels)
    #print(gmeans_student_labels)
    ## Should be correct
    print(np.unique(gmeans_matthias_labels, return_counts=True))
    print(np.unique(gmeans_student_labels, return_counts=True))
    equivalent = are_clusters_equivalent(gmeans_student_labels, gmeans_matthias_labels)
    print(f"The clustering results are equivalent after relabeling: {equivalent}")   

    
    gmeans_matthias_centers, gmeans_matthias_labels = gmeans(points,
                                                            tolerance=0.1,
                                                            max_singlerun_iterations=max_singlerun_iterations)

    plot_clustering_results(scene_pcd,
                    gmeans_matthias_labels,
                    "G-means student")
    
    start = time.time()
    gmeans_student_centers, gmeans_student_labels = student_module.gmeans(points,
                                                            tolerance=0.1,
                                                            max_singlerun_iterations=max_singlerun_iterations)
    stop = time.time()
    print("Time for Gmeans: ", stop - start)
    plot_clustering_results(scene_pcd,
                        gmeans_student_labels,
                        "G-means solution")