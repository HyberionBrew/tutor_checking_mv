import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib

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

def show_images_side_by_side(img1, img2, window_title="Comparison Solution/Student"):
    # Concatenate images horizontally
    combined_image = np.hstack((img1, img2))

    # Display the concatenated image
    cv2.imshow(window_title, combined_image)
    cv2.waitKey(0)  # Wait for a key press to close
    cv2.destroyAllWindows()
