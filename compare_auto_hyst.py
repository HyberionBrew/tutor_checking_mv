import sys
import os
import importlib.util
import numpy as np


def load_module(module_path):
    spec = importlib.util.spec_from_file_location("module.name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def compare_arrays(arr1, arr2):
    if np.allclose(arr1, arr2, atol=1e-5):
        print("The outputs are approximately the same.")
    else:
        print("The outputs are different.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_hyst_auto.py <StudentFolderName>")
        sys.exit(1)

    student_name = sys.argv[1]
    student_folder = os.path.join("students", student_name)
    solution_folder = "solution"

    if not os.path.exists(student_folder):
        print(f"Error: The folder {student_folder} does not exist.")
        sys.exit(1)

    sys.path.append(solution_folder)
    try:
        student_module = load_module(os.path.join(student_folder, "hyst_auto.py"))

        # Remove the solution folder from sys.path to avoid potential conflicts
        
        solution_module = load_module(os.path.join(solution_folder, "hyst_auto.py"))
    finally:
        sys.path.remove(solution_folder)
    # Generate a synthetic image and set parameters for the test
    img = np.random.rand(100, 100).astype(np.float32)
    low_prop = 0.1
    high_prop = 0.05

    # Apply hysteresis thresholding
    student_output = student_module.hyst_thresh_auto(img, low_prop, high_prop)
    solution_output = solution_module.hyst_thresh_auto(img, low_prop, high_prop)

    # Compare outputs
    compare_arrays(student_output, solution_output)
    
    

if __name__ == "__main__":
    main()
