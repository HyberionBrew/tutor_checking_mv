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
        difference = np.abs(arr1 - arr2)
        show_images_side_by_side(arr1, arr2, f"Student {name}", f"Solution {name}")
        show_images_side_by_side(difference, difference, f"Difference {name}", f"Difference {name}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_non_max.py <StudentFolderName>")
        sys.exit(1)

    student_name = sys.argv[1]
    student_folder = os.path.join("students", student_name)
    solution_folder = "solution"

    if not os.path.exists(student_folder):
        print(f"Error: The folder {student_folder} does not exist.")
        sys.exit(1)

    student_module = load_module(os.path.join(student_folder, "non_max.py"))
    solution_module = load_module(os.path.join(solution_folder, "non_max.py"))
    sobel_module = load_module(os.path.join(solution_folder, "sobel.py"))

    # Load image
    img = cv2.imread('solution/image/circle.jpg', cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0

    # Apply Sobel to find gradients and orientations
    gradient_og, orientation_og = sobel_module.sobel(img)

    # Apply non-maximum suppression
    gradients_student = gradient_og.copy()
    orientation_student = orientation_og.copy()
    student_output = student_module.non_max(gradients_student, orientation_student)
    # check if gradients are still the same 
    compare_arrays(gradient_og, gradients_student, "Gradients")
    # compare_arrays(orientation_og, orientation_student, "Orientations")
    if not (np.allclose(orientation_og, orientation_student, atol=1e-5)):
        print("unintended consquences of non_max, orientation")
    if not (np.allclose(gradient_og, gradients_student, atol=1e-5)):
        print("unintended consquences of non_max, gradient")

    solution_output = solution_module.non_max_alt(gradient_og, orientation_og)

    # Compare outputs
    # student_output[:60] = 1
    show_images_side_by_side(student_output, solution_output, f"Student ", f"Solution")
    compare_arrays(student_output, solution_output, "Non-Max Suppression Result")
    
    # compare_arrays(solution_output, solution_output, "Non-Max Suppression Result")
if __name__ == "__main__":
    main()
