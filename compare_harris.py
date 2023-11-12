import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util

def modify_sys_path(add_path=None, remove_path=None):
    if add_path and add_path not in sys.path:
        sys.path.insert(0, str(add_path))
    if remove_path and remove_path in sys.path:
        sys.path.remove(str(remove_path))

def load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def bin_keypoints(keypoints, tolerance=5):
    # Round the (x, y) coordinates to the nearest tolerance
    binned_keypoints = {(round(kp.pt[0] / tolerance) * tolerance, round(kp.pt[1] / tolerance) * tolerance) for kp in keypoints}
    return binned_keypoints

def bin_responses(keypoints, response_tolerance=0.01):
    # Bin the responses of the keypoints
    binned_responses = {round(kp.response / response_tolerance) * response_tolerance for kp in keypoints}
    return binned_responses

def bin_responses(keypoints, response_tolerance=0.01):
    # Bin the responses of the keypoints
    binned_responses = {round(kp.response / response_tolerance) * response_tolerance for kp in keypoints}
    return binned_responses

def compare_keypoints(keypoints1, keypoints2, position_tolerance=5, response_tolerance=0.01):
    count1 = len(keypoints1)
    count2 = len(keypoints2)

    # Bin keypoints for both position and response
    binned_coords1 = bin_keypoints(keypoints1, position_tolerance)
    binned_coords2 = bin_keypoints(keypoints2, position_tolerance)
    binned_responses1 = bin_responses(keypoints1, response_tolerance)
    binned_responses2 = bin_responses(keypoints2, response_tolerance)

    # Compare the sets of binned coordinates and responses
    common_positions = binned_coords1.intersection(binned_coords2)
    unique_positions_solution = binned_coords1 - binned_coords2
    unique_positions_student = binned_coords2 - binned_coords1
    common_responses = binned_responses1.intersection(binned_responses2)
    # Prepare the comparison summary
    summary = [
        f"Solution keypoints: {count1}",
        f"Student keypoints: {count2}",
        f"Common positions: {len(common_positions)}",
        f"Unique positions to solution: {len(unique_positions_solution)}",
        f"Unique positions to student: {len(unique_positions_student)}",
        f"Common responses: {len(common_responses)} / # solution: {len(binned_responses1)}, # student: {len(binned_responses2)}" # no repsonses?
    ]

    return "\n".join(summary)


def show_images_side_by_side(img1, img2, window_title="Comparison Solution/Student"):
    # Concatenate images horizontally
    combined_image = np.hstack((img1, img2))

    # Display the concatenated image
    cv2.imshow(window_title, combined_image)
    cv2.waitKey(0)  # Wait for a key press to close
    cv2.destroyAllWindows()

if __name__ == '__main__':
    student_name = sys.argv[1]

    # Load the student's solution
    solution_dir = Path(__file__).parent.joinpath('solution')
    modify_sys_path(add_path=solution_dir)
    from harris_corner import harris_corner
    from helper_functions import show_image
    modify_sys_path(remove_path=solution_dir)

    # Load student's module
    student_module_path = Path(__file__).parent.joinpath(f'students/{student_name}/harris_corner.py')
    modify_sys_path(add_path=student_module_path.parent)
    student_module = load_module(student_module_path, "student_harris_corner")
    modify_sys_path(remove_path=student_module_path.parent)

    # Image paths and parameters
    img_path_1 = 'solution/desk/Image-00.jpg'

    sigma1, sigma2, threshold, k, patch_size = 0.8, 1.5, 0.01, 0.04, 5

    # Load and process images
    img_gray = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Couldn't load image {img_path_1}")
    img_gray_int = img_gray.copy()
    img_gray = img_gray.astype(np.float32) / 255.

    # Run Harris corner detection
    keypoints_solution = harris_corner(img_gray, sigma1, sigma2, k, threshold)
    # print(keypoints_solution)
    keypoints_student = student_module.harris_corner(img_gray, sigma1, sigma2, k, threshold)

    # Compare keypoints
    differences = compare_keypoints(keypoints_solution, keypoints_student)
    print("Differences:", differences)

    # Plot results
    keypoints_img_solution = np.zeros(shape=img_gray.shape, dtype=np.uint8)
    keypoints_img_solution = cv2.drawKeypoints(img_gray_int, keypoints_solution, keypoints_img_solution)
    
    keypoints_img_student = np.zeros(shape=img_gray.shape, dtype=np.uint8)
    keypoints_img_student = cv2.drawKeypoints(img_gray_int, keypoints_student, keypoints_img_student)
    show_images_side_by_side(keypoints_img_solution, keypoints_img_student)
