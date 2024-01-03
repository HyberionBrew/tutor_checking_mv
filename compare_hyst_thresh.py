import numpy as np
import cv2
import sys
import os
import importlib.util
from pathlib import Path

def load_module(module_path):
    spec = importlib.util.spec_from_file_location("module.name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def check_array_properties(arr, name):
    if arr.dtype != np.float32:
        print(f"The {name} array does not have the correct dtype. Expected np.float32, but got {arr.dtype}.")
        return False

    if len(arr.shape) != 2:
        print(f"The {name} array does not have the correct number of dimensions. Expected 2, but got {len(arr.shape)}.")
        return False
    
    unique_values = np.unique(arr)
    if not all(val in [0.0, 1.0] for val in unique_values):
        print(f"The {name} array contains values other than 0 or 1.")
        return False
    
    return True

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
        show_images_side_by_side(arr1, arr2, f"Student {name}", f"Solution {name}")
    else:
        print(f"The {name} arrays are different.")
        show_images_side_by_side(arr1, arr2, f"Student {name}", f"Solution {name}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_hyst_thresh.py <StudentFolderName>")
        sys.exit(1)

    student_name = sys.argv[1]
    student_folder = os.path.join("students", student_name)
    solution_folder = "solution"

    if not os.path.exists(student_folder):
        print(f"Error: The folder {student_folder} does not exist.")
        sys.exit(1)

    # Load solution and student modules
    student_module = load_module(os.path.join(student_folder, "hyst_thresh.py"))
    solution_module = load_module(os.path.join(solution_folder, "hyst_thresh.py"))
    sobel_module = load_module(os.path.join(solution_folder, "sobel.py"))
    non_max_module = load_module(os.path.join(solution_folder, "non_max.py"))
    blur_gauss_module = load_module(os.path.join(solution_folder, "blur_gauss.py"))

    # Load image
    current_path = Path(__file__).parent
    img_gray = cv2.imread(str(current_path.joinpath("solution/image/parliament.jpg")), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Couldn't load image in " + str(current_path))
    img_gray = img_gray.astype(np.float32) / 255.0
    # img_gray = np.random.rand(100, 100).astype(np.float32)
    # Apply processing pipeline
    sigma = 1.  # Change this value as necessary
    img_blur,_ = blur_gauss_module.blur_gauss(img_gray, sigma)
    print(img_blur.shape)

    gradients, orientations = sobel_module.sobel(img_blur)
    edges = non_max_module.non_max(gradients, orientations)

    # Apply hysteresis thresholding using solution
    edges_not_student = edges.copy()
    solution_output = solution_module.hyst_thresh(edges_not_student, 0.3, 0.4)

    # Apply hysteresis thresholding using student's implementation
    student_output = student_module.hyst_thresh(edges, 0.3, 0.4)
    # check properties of student output
    if not check_array_properties(student_output, "student"):
        print("Student output is not correct.")
    else:
        print("Student output properties are correct.")
    # Compare outputs
    compare_arrays(student_output, solution_output, "Hysteresis Thresholding Result")
    print(np.unique(student_output, return_counts=True))
    print(np.unique(solution_output, return_counts=True))
if __name__ == "__main__":
    main()
