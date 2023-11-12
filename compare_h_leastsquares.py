import cv2
import numpy as np
from pathlib import Path
import sys
import importlib.util
from solution.helper_functions import show_image  # Import from solution folder
from solution.helper_functions import debug_homography, draw_rectangles  # Import debug functions from solution

from utility_functions import modify_sys_path, load_module, show_images_side_by_side

# Reuse modify_sys_path and load_module functions from previous scripts

def compare_homographies(homography_solution, homography_student, tolerance=1e-4):
    # Check if the shapes are the same
    if homography_solution.shape != homography_student.shape:
        return "Homography shapes differ."

    # check if h22 is 1
    if homography_solution[2, 2] != 1 or not(np.isclose(homography_student[2, 2],1)):
        return "h22 is not 1"
    # Calculate the difference matrix
    diff_matrix = np.abs(homography_solution - homography_student)

    # Check if the difference is within the tolerance
    if np.all(diff_matrix < tolerance):
        return "Homographies are similar within the tolerance."
    else:
        # Count the number of elements exceeding the tolerance
        count_exceeding_tolerance = np.sum(diff_matrix >= tolerance)
        return f"Homographies differ at {count_exceeding_tolerance} elements (tolerance {tolerance})."
def generate_random_points(num_points=5, img_size=(100, 100)):
    """
    Generates random points within the specified image dimensions.
    
    :param num_points: Number of random points to generate.
    :param img_size: Tuple representing the size of the image (height, width).
    :return: Array of generated points.
    """
    height, width = img_size
    points = np.random.rand(num_points, 2)  # Generate random points in range [0, 1)
    points[:, 0] *= width  # Scale x coordinate to image width
    points[:, 1] *= height # Scale y coordinate to image height
    return points

def apply_transformation(points, angle=30, scale=1.0, translation=(10, 10)):
    """
    Applies a simple geometric transformation to the points.
    
    :param points: Array of points to transform.
    :param angle: Rotation angle in degrees.
    :param scale: Scaling factor.
    :param translation: Translation vector (dx, dy).
    :return: Array of transformed points.
    """
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    scaled_rotation_matrix = scale * rotation_matrix
    transformed_points = np.dot(points, scaled_rotation_matrix.T) + translation
    return transformed_points



if __name__ == '__main__':
    student_name = sys.argv[1]

    # Load solution's find_homography_leastsquares
    solution_dir = Path(__file__).parent.joinpath('solution')
    modify_sys_path(add_path=solution_dir)
    from find_homography import find_homography_leastsquares as solution_find_homography
    modify_sys_path(remove_path=solution_dir)

    # Load student's find_homography_leastsquares
    student_module_path = Path(__file__).parent.joinpath(f'students/{student_name}/find_homography.py')
    modify_sys_path(add_path=student_module_path.parent)
    student_module = load_module(student_module_path, "student_find_homography")
    modify_sys_path(remove_path=student_module_path.parent)

    # Generate image with randomly projected rectangle to test the find_homography_leastsquares function
    rectangle_img, rectangle_tf, rectangle = debug_homography()
    print(rectangle.shape)
    print(rectangle_tf.shape)
    # Find homography using solution's implementation
    homography_solution = solution_find_homography(rectangle, rectangle_tf)

    # Find homography using student's implementation
    homography_student = student_module.find_homography_leastsquares(rectangle, rectangle_tf)

    # Compare homographies
    differences = compare_homographies(homography_solution, homography_student)
    print("Differences:", differences)

    # Visual comparison (optional)
    transformed_rectangle_img_solution = draw_rectangles(rectangle_img, np.zeros(shape=(60, 100)), homography=homography_solution)
    transformed_rectangle_img_student = draw_rectangles(rectangle_img, np.zeros(shape=(60, 100)), homography=homography_student)

    solution_img = transformed_rectangle_img_solution.astype(np.float32) / 255
    student_img = transformed_rectangle_img_student.astype(np.float32) / 255
    
    source_points_5 = generate_random_points(num_points=5)
    transformed_target_points_5 = apply_transformation(source_points_5)

    # Find homography using solution's implementation for 5 points
    homography_solution_5 = solution_find_homography(source_points_5, transformed_target_points_5)

    # Find homography using student's implementation for 5 points
    homography_student_5 = student_module.find_homography_leastsquares(source_points_5, transformed_target_points_5)

    # Compare homographies for 5 points
    differences_5 = compare_homographies(homography_solution_5, homography_student_5)
    print("Differences for 5 points:", differences_5)
    show_images_side_by_side(solution_img, student_img)

    #show_image(transformed_rectangle_img_solution.astype(np.float32) / 255., title="Transformed (Solution)", use_matplotlib=False)
    #show_image(transformed_rectangle_img_student.astype(np.float32) / 255., title="Transformed (Student)", use_matplotlib=False)
