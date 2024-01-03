import numpy as np
import cv2
import sys
import os
import importlib.util

# To use return the gaussian kernel as well in the students code
# run with 1.0 and 1.5

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

def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)

def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_blur.py <StudentFolderName>")
        sys.exit(1)

    student_name = sys.argv[1]
    sigmas = [1.0, 1.5]
    error = False
    for sigma in sigmas:
        #sigma = float(sys.argv[2])
        student_folder = os.path.join("students", student_name)
        solution_folder = "solution"

        if not os.path.exists(student_folder):
            print(f"Error: The folder {student_folder} does not exist.")
            sys.exit(1)

        student_module = load_module(os.path.join(student_folder, "blur_gauss.py"))
        solution_module = load_module(os.path.join(solution_folder, "blur_gauss.py"))

        # Creating a dummy image for testing
        img = np.random.rand(100, 100).astype(np.float32)

        # Defining sigma

        # Running blur_gauss from student's code
        student_result, student_kernel = student_module.blur_gauss(img, sigma)

        # Running blur_gauss from solution
        solution_result, solution_kernel = solution_module.blur_gauss(img, sigma)

        # Comparing results
        if  np.allclose(student_result, solution_result, atol=1e-5):
            print("The output images are exactly the same.")
        else:
            print("The output images are different.")
            # compare images in the middle (to check for border error)
            if np.allclose(student_result[10:-10, 10:-10], solution_result[10:-10, 10:-10], atol=1e-5):
                print("The output images are the same in the middle, probably just other border.")
            else:
                error = True

            print(f"Student result: {student_result}")
            print(f"Solution result: {solution_result}")
            show_images_side_by_side(student_result, solution_result, "Student Result", "Solution Result")

        # Checking the type and shape of the output images
        print(f"Student result dtype: {student_result.dtype}, {student_result.shape}")
        if student_result.dtype == solution_result.dtype:
            print("The data types of the output images are the same.")
        else:
            print("The data types of the output images are different.")
            error = True
        if student_result.shape == solution_result.shape:
            print("The shapes of the output images are the same.")
        else:
            error = True
            print("The shapes of the output images are different.")

        # Comparing Gaussian kernels
        # check if shapes are the same kernel
        if student_kernel.shape == solution_kernel.shape:
            print("The Gaussian kernels have the same shape.")
        else:
            error = True
            print("The Gaussian kernels have different shapes.")


        if np.allclose(student_kernel, solution_kernel, atol=1e-5):
            print("The Gaussian kernels are close/the same.")
        else:
            error = True
            print("kernels:")
            print(student_kernel[2:5, 2:5])
            print("----")
            print(solution_kernel[2:5,2:5])
            print("The Gaussian kernels are different.")
            if np.allclose(student_kernel, solution_kernel, atol=1e-3):
                print("Put only slightly different.")

        # Checking if the student's Gaussian kernel is symmetric
        if is_symmetric(student_kernel):
            print("The student's Gaussian kernel is symmetric.")
        else:
            error = True
            print("The student's Gaussian kernel is not symmetric.")

        # Checking if the student's Gaussian kernel sums to 1
        if np.isclose(np.sum(student_kernel), 1):
            error = True
            print("The sum of the student's Gaussian kernel is 1.")
        else:
            print(f"The sum of the student's Gaussian kernel is not 1, but {np.sum(student_kernel)}.")
        print(f"##### FINISHED SIGMA ##### {sigma}")
    if error:
        print("There were errors.")
    else:
        print("No errors found.")
if __name__ == "__main__":
    main()
