import cv2
import numpy as np
from pathlib import Path
import sys
import importlib.util
from utility_functions import modify_sys_path, load_module

# Reuse modify_sys_path and load_module functions from previous scripts

def compare_filtered_matches(matches_solution, matches_student):
    # Implement logic to compare filtered matches
    # This could involve comparing the number of matches, and potentially the match quality or indices
    # Return a summary of differences
    # check the length
    if len(matches_solution) != len(matches_student):
        return "Length of matches differ: Solution length {} vs Student length {}".format(len(matches_solution), len(matches_student))
    else:
        return "Length of matches are the same: Solution length {} vs Student length {}".format(len(matches_solution), len(matches_student))

if __name__ == '__main__':
    student_name = sys.argv[1]

    # Load solution modules (harris_corner, compute_descriptors, and filter_matches)
    solution_dir = Path(__file__).parent.joinpath('solution')
    modify_sys_path(add_path=solution_dir)
    from harris_corner import harris_corner
    from descriptors import compute_descriptors
    from helper_functions import filter_matches as solution_filter_matches
    modify_sys_path(remove_path=solution_dir)

    # Load student's filter_matches module
    student_module_path = Path(__file__).parent.joinpath(f'students/{student_name}/helper_functions.py')
    modify_sys_path(add_path=student_module_path.parent)
    student_module = load_module(student_module_path, "student_filter_matches")
    modify_sys_path(remove_path=student_module_path.parent)

    # Load and process images, detect keypoints, compute descriptors, and match descriptors
    # Load and process images
    img_path_1 = 'solution/desk/Image-00.jpg'
    img_path_2 = 'solution/desk/Image-01.jpg'
    sigma1, sigma2, threshold, k, patch_size = 0.8, 1.5, 0.01, 0.04, 5

    img_gray_1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
    if img_gray_1 is None:
        raise FileNotFoundError(f"Couldn't load image {img_path_1}")
    img_gray_1 = img_gray_1.astype(np.float32) / 255.

    img_gray_2 = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)
    if img_gray_2 is None:
        raise FileNotFoundError(f"Couldn't load image {img_path_2}")
    img_gray_2 = img_gray_2.astype(np.float32) / 255.

    # Harris corner detector
    keypoints_1 = harris_corner(img_gray_1, sigma1, sigma2, k, threshold)
    keypoints_2 = harris_corner(img_gray_2, sigma1, sigma2, k, threshold)

    # Compute descriptors
    _, descriptors_1 = compute_descriptors(img_gray_1, keypoints_1, patch_size)
    _, descriptors_2 = compute_descriptors(img_gray_2, keypoints_2, patch_size)

    # FLANN (Fast Library for Approximate Nearest Neighbors) parameters and matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # For each keypoint in img_gray_1 get the best matches
    matches = flann.knnMatch(descriptors_1.astype(np.float32), descriptors_2.astype(np.float32), k=2)


    # Filter matches using both solution and student's implementations
    filtered_matches_solution = solution_filter_matches(matches)
    filtered_matches_student = student_module.filter_matches(matches)

    # Compare filtered matches
    differences = compare_filtered_matches(filtered_matches_solution, filtered_matches_student)
    print("Differences:", differences)
