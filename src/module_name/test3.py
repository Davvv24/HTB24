import cv2

import numpy as np

def check_diff(colour1, colour2, threshold=200):
    distance = np.linalg.norm(colour1 - colour2)  # Euclidean distance
    return distance > threshold  # There's a big difference

def apply_lidar_smoothing(image_path, lidar_path, output_path='images/white_image.jpg'):
    """Processes an image using LiDAR data to smooth out sections."""
    lidar = cv2.imread(lidar_path)
    image = cv2.imread(image_path)

    height, width, _ = image.shape
    white_image = np.full((height, width, 3), 255, dtype=np.uint8)

    for row in range(5000):  # Process the first 500 rows
        print(f"Processing row {row}")
        column = 0

        while column+1 < width:
            if check_diff(lidar[row, column], lidar[row, column + 1]):
                white_image[row, column] = (255, 0, 0)
            column += 1

    cv2.imwrite(output_path, white_image)
    return white_image

# Load the image and LiDAR data
image_path = 'images/map_enlarged.jpg'
lidar_path = 'images/image_converted.jpg'

smoothed_image = apply_lidar_smoothing(image_path, lidar_path)

