import numpy as np
import cv2
import sys
import os
import importlib.util

def load_module(module_path):
    spec = importlib.util.spec_from_file_location("module.name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def show_images_side_by_side(img1, img2, title1='Image 1', title2='Image 2'):
    # Concatenate images horizontally
    combined_img = np.hstack((img1, img2))

    # Normalize the images for display
    combined_img = (combined_img - np.min(combined_img)) / (np.max(combined_img) - np.min(combined_img))

    # Show images
    cv2.imshow(f'{title1} - {title2}', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_arrays(arr1, arr2, name):
    if np.allclose(arr1, arr2, atol=1e-5):
        print(f"The {name} arrays are approximately the same.")
    else:
        print(f"The {name} arrays are different.")
        show_images_side_by_side(arr1, arr2, f"Student {name}", f"Solution {name}")

def check_return_values(gradient, orientation, student_name):
    if not isinstance(gradient, np.ndarray):
        print(f"Error: {student_name}'s sobel function returned gradient is not a NumPy array")
        return False

    if not isinstance(orientation, np.ndarray):
        print(f"Error: {student_name}'s sobel function returned orientation is not a NumPy array")
        return False

    if gradient.dtype != np.float32:
        print(f"Error: {student_name}'s sobel function returned gradient does not have dtype np.float32")
        return False

    if orientation.dtype != np.float32:
        print(f"Error: {student_name}'s sobel function returned orientation does not have dtype np.float32")
        return False

    if not (0.0 <= gradient).all() or not (gradient <= 1.0).all():
        print(f"Error: {student_name}'s sobel function returned gradient values are not in the range [0., 1.]")
        return False

    if not (-np.pi <= orientation).all() or not (orientation <= np.pi).all():
        print(f"Error: {student_name}'s sobel function returned orientation values are not in the range [-np.pi, np.pi]")
        return False

    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_sobel.py <StudentFolderName>")
        sys.exit(1)

    student_name = sys.argv[1]
    student_folder = os.path.join("students", student_name)
    solution_folder = "solution"

    if not os.path.exists(student_folder):
        print(f"Error: The folder {student_folder} does not exist.")
        sys.exit(1)

    student_module = load_module(os.path.join(student_folder, "sobel.py"))
    solution_module = load_module(os.path.join(solution_folder, "sobel.py"))

    # Load the image
    img = cv2.imread(os.path.join(solution_folder, "image", "circle.jpg"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load the image.")
        sys.exit(1)
    
    # Normalize image to [0., 1.]
    img = img.astype(np.float32) / 255.0

    # Running sobel from student's code
    student_gradient, student_orientation = student_module.sobel(img)

    if check_return_values(student_gradient, student_orientation, "Student"):
        print("Student's sobel function returned correct values.")
    # Running sobel from solution
    solution_gradient, solution_orientation = solution_module.sobel(img)
    solution_gradient_rotated, solution_orientation_rotated = solution_module.sobel_rotated_kernel(img)
    show_images_side_by_side(student_gradient, solution_gradient, "Student gradient", "Solution gradient")
    # Comparing gradients
    compare_arrays(student_gradient, solution_gradient, "gradient")

    # Comparing orientations
    compare_arrays(student_orientation, solution_orientation, "orientation")
    
    compare_arrays(student_gradient, solution_gradient_rotated, "gradient rotated")
    compare_arrays(student_orientation, solution_orientation_rotated, "gradient rotated")
if __name__ == "__main__":
    main()
