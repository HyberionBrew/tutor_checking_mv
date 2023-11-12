import cv2
import numpy as np
from pathlib import Path
import sys
import importlib.util

from utility_functions import modify_sys_path, load_module
# You can reuse the modify_sys_path and load_module functions from the previous script

import numpy as np

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def compare_descriptors(descriptors_solution, descriptors_student, tolerance=0.1):
    # Check if the descriptor lengths are the same
    if descriptors_solution.shape[1] != descriptors_student.shape[1]:
        return "Descriptor lengths differ: Solution length {} vs Student length {}".format(descriptors_solution.shape[1], descriptors_student.shape[1])

    # Check if each descriptor in the student's output is contained in the solution
    match_count = 0
    for desc_stu in descriptors_student:
        if any(euclidean_distance(desc_stu, desc_sol) < tolerance for desc_sol in descriptors_solution):
            match_count += 1

    summary = [
        f"Total descriptors (solution): {descriptors_solution.shape[0]}",
        f"Total descriptors (student): {descriptors_student.shape[0]}",
        f"Student descriptors matching with solution: {match_count} / {descriptors_student.shape[0]}"
    ]

    return "\n".join(summary)

def compare_keypoints(solution, student):
    # compare the length
    summary = [
        f"Solution keypoints: {len(solution)}",
        f"Student keypoints: {len(student)}",
    ]
    return "\n".join(summary)

if __name__ == '__main__':
    student_name = sys.argv[1]

    # Load solution modules (harris_corner and compute_descriptors)
    solution_dir = Path(__file__).parent.joinpath('solution')
    modify_sys_path(add_path=solution_dir)
    from harris_corner import harris_corner
    from descriptors import compute_descriptors as solution_compute_descriptors
    modify_sys_path(remove_path=solution_dir)

    # Load student's compute_descriptors module
    student_module_path = Path(__file__).parent.joinpath(f'students/{student_name}/descriptors.py')
    modify_sys_path(add_path=student_module_path.parent)
    student_module = load_module(student_module_path, "student_compute_descriptors")
    modify_sys_path(remove_path=student_module_path.parent)

    # Image paths and parameters
    img_path_1 = 'solution/desk/Image-00.jpg'
    sigma1, sigma2, threshold, k, patch_size = 0.8, 1.5, 0.01, 0.04, 5

    # Load and process images
    img_gray = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Couldn't load image {img_path_1}")
    img_gray = img_gray.astype(np.float32) / 255.

    # Run Harris corner detection
    keypoints = harris_corner(img_gray, sigma1, sigma2, k, threshold)

    # Compute descriptors using both solution and student's implementations
    filtered_solution, descriptors_solution = solution_compute_descriptors(img_gray, keypoints, patch_size)
    filtered_student, descriptors_student = student_module.compute_descriptors(img_gray.copy(), keypoints.copy(), patch_size)

    # Compare descriptors
    differences = compare_descriptors(descriptors_solution, descriptors_student)
    
    print("Differences Descriptors:", differences)
    # compare filtered keypoints
    differences = compare_keypoints(filtered_solution, filtered_student)
    print("Differences Filtered Keypoints:", differences)

    print("Even patch size")

    filtered_solution, descriptors_solution = solution_compute_descriptors(img_gray, keypoints, 6)
    filtered_student, descriptors_student = student_module.compute_descriptors(img_gray.copy(), keypoints.copy(), 6)

    # Compare descriptors
    differences = compare_descriptors(descriptors_solution, descriptors_student)
    
    print("Differences Descriptors:", differences)
    # compare filtered keypoints
    differences = compare_keypoints(filtered_solution, filtered_student)
    print("Differences Filtered Keypoints:", differences)