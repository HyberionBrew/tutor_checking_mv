import cv2
import numpy as np
from pathlib import Path
import sys
import importlib.util

from utility_functions import modify_sys_path, load_module, show_images_side_by_side

def compare_homography_ransac(homography_solution, homography_student, inliers_solution, inliers_student):
    # Implement logic to compare homographies and inliers
    # This could involve comparing the homography matrices, number and distribution of inliers
    # Return a summary of differences
    pass #return differences_summary

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Couldn't load image {image_path}")
    return img # .astype(np.float32) / 255.

def compare_ransac_results(homography_solution, homography_student, inliers_solution, inliers_student, its_solution, its_student, tolerance=1e-4):
    differences = {}

    # 1. Check homography matrices
    if homography_solution.shape != homography_student.shape:
        differences['homography_shape'] = "Homography shapes differ."
    else:
        homography_diff = np.abs(homography_solution - homography_student)
        differences['homography_difference'] = np.sum(homography_diff)

    # 2. Check inliers
    # Ensure both inlier arrays are the same shape for comparison
    if inliers_solution.shape != inliers_student.shape:
        differences['inliers_shape'] = "Inliers shapes differ."
    else:
        # Count the number of inliers that differ
        inliers_discrepancy = np.sum(inliers_solution != inliers_student)
        differences['inliers_discrepancy'] = inliers_discrepancy
    # 3. Check error
    differences['num_iterations'] = (its_solution, its_student)
    return differences

if __name__ == '__main__':
    student_name = sys.argv[1]
    ransac_confidence = 0.85
    ransac_inlier_threshold = 5.

    # Load solution's find_homography_ransac
    solution_dir = Path(__file__).parent.joinpath('solution')
    modify_sys_path(add_path=solution_dir)
    from find_homography import find_homography_ransac as solution_find_homography_ransac
    from helper_functions import filter_matches
    modify_sys_path(remove_path=solution_dir)

    # Load student's find_homography_ransac
    student_module_path = Path(__file__).parent.joinpath(f'students/{student_name}/find_homography.py')
    modify_sys_path(add_path=student_module_path.parent)
    student_module = load_module(student_module_path, "student_find_homography_ransac")
    modify_sys_path(remove_path=student_module_path.parent)

    # Load and process images, detect keypoints, compute descriptors, and match descriptors
    current_path = Path(__file__).parent
    scene_img_path = str(current_path.joinpath("solution/data/image")) + "5.jpg"
    object_img_path = str(current_path.joinpath("solution/data/object.jpg"))

    # Load and preprocess images
    scene_img_gray = load_and_preprocess_image(scene_img_path)
    object_img_gray = load_and_preprocess_image(object_img_path)

    # SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    target_keypoints, target_descriptors = sift.detectAndCompute(scene_img_gray, None)
    source_keypoints, source_descriptors = sift.detectAndCompute(object_img_gray, None)

    # FLANN (Fast Library for Approximate Nearest Neighbors) parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Matching descriptors using FLANN
    matches = flann.knnMatch(source_descriptors, target_descriptors, k=2)
    matches = filter_matches(matches)

    # Extract source and target points from matches
    source_points = np.float32([source_keypoints[match.queryIdx].pt for match in matches])#.reshape(-1, 1, 2)
    target_points = np.float32([target_keypoints[match.trainIdx].pt for match in matches])#.reshape(-1, 1, 2)

    # Apply RANSAC to find homography using solution's implementation
    print(source_points.shape)
    print(target_points.shape)
    homography_solution, best_inliers_solution, its_solution = solution_find_homography_ransac(source_points, target_points, ransac_confidence, ransac_inlier_threshold)

    # Apply RANSAC to find homography using student's implementation
    homography_student, best_inliers_student, its_student = student_module.find_homography_ransac(source_points, target_points, ransac_confidence, ransac_inlier_threshold)

    # Compare results
    #differences = compare_homography_ransac(homography_solution, homography_student, best_inliers_solution, best_inliers_student)
    #print("Differences:", differences)

    # Visualization of the results
    transformed_image_solution = cv2.warpPerspective(object_img_gray, homography_solution, (scene_img_gray.shape[1], scene_img_gray.shape[0]))
    transformed_image_student = cv2.warpPerspective(object_img_gray, homography_student, (scene_img_gray.shape[1], scene_img_gray.shape[0]))
    differences = compare_ransac_results(homography_solution, homography_student, best_inliers_solution, best_inliers_student, its_solution, its_student)
    print("Differences:", differences)
    show_images_side_by_side(transformed_image_solution, transformed_image_student, "Homography Comparison")

