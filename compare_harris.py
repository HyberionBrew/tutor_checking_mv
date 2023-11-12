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

def compare_keypoints(keypoints1, keypoints2, tolerance=2):
    count1 = len(keypoints1)
    count2 = len(keypoints2)
    # Bin keypoints for both solutions
    binned_coords1 = bin_keypoints(keypoints1, tolerance)
    binned_coords2 = bin_keypoints(keypoints2, tolerance)

    # Compare the sets of binned coordinates
    common_points = binned_coords1.intersection(binned_coords2)
    unique_to_solution = binned_coords1 - binned_coords2
    unique_to_student = binned_coords2 - binned_coords1

    # Prepare the comparison summary
    summary = [
        f"Solution keypoints (binned): {len(binned_coords1)}",
        f"Student keypoints (binned): {len(binned_coords2)}",
        f"Common keypoints: {len(common_points)}",
        f"Unique to solution: {len(unique_to_solution)}",
        f"Unique to student: {len(unique_to_student)}",
        f"solution keypoints: {count1}",
        f"student keypoints: {count2}"
        f" used np.grad in solution: True", # if student doesnt unsol=16, unstu=20
    ]

    return "\n".join(summary)

def show_images_side_by_side(img1, img2, window_title="Comparison Solution/Student"):
    # Concatenate images horizontally
    combined_image = np.hstack((img1, img2))

    # Display the concatenated image
    cv2.imshow(window_title, combined_image)
    cv2.waitKey(0)  # Wait for a key press to close
    cv2.destroyAllWindows()


def plot_results(keypoints_img1, keypoints_img2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title("Solution")
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title("Student")
    plt.show()

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
