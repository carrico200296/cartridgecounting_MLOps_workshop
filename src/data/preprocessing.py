import cv2
import numpy as np
import pathlib


def get_image_and_circles(path, id):
    # load image and centers (labels)
    image_path = f"{path}/{id}.jpg"
    centers_path = f"{path}/{id}.npy"

    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    circles = np.load(centers_path)

    return orig_img, circles


def create_dot_mask(original_image, circles, size):
    target = np.zeros((size, size), np.uint8)
    orig_width, orig_height = original_image.shape
    new_width, new_height = target.shape
    new_centers = []
    for (y, x, _) in circles[0, :]:
        x_new, y_new = (x / orig_width) * new_width, (y / orig_height) * new_height
        new_centers.append((x_new, y_new))
        target[int(x_new), int(y_new)] = 255

    return target


def create_segmentation_labels(original_image, circles, size):
    target = np.zeros((size, size), np.uint8)
    orig_width, orig_height = original_image.shape
    new_width, new_height = target.shape
    new_centers = []
    if circles.shape[0] > 1:
        circles = np.expand_dims(circles, axis=0)
    for circle_cord in circles[0, :]:
        y, x = circle_cord[0], circle_cord[1]
        x_new, y_new = (x / orig_width) * new_width, (y / orig_height) * new_height
        new_centers.append((x_new, y_new))
        # target = cv2.circle(target, (int(x_new), int(y_new)), 3, 255, cv2.FILLED, 8, 0)
        target[int(x_new) - 2 : int(x_new) + 2, int(y_new) - 2 : int(y_new) + 2] = 255.0

    return target


def preprocess_image_segmentation(path, id, size=256, gaussian_blur_kernel_size=(1, 1)):

    # load image and centers (labels)
    orig_img, circles = get_image_and_circles(path, id)

    # resize image
    resized_image = cv2.resize(orig_img, (size, size))

    # create dot mask
    target = create_segmentation_labels(orig_img, circles, size)

    # apply gaussian blur
    # blurred_img = cv2.GaussianBlur(target, gaussian_blur_kernel_size, 1, 1)

    return resized_image, target
