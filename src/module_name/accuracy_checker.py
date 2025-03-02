import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def image_similarity(image_path_1, image_path_2):
    # Load the images
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)

    # Convert to grayscale for similarity calculation (SSIM works better with grayscale images)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    ssim_value, _ = ssim(gray1, gray2, full=True)

    # Convert SSIM value to percentage similarity
    similarity_percentage = ssim_value * 100

    return similarity_percentage

# Example usage
image_path_1 = 'images/google_maps.jpg'
image_path_2 = 'images/benchmark.jpg'
image_path_3 = 'images/block.jpg'
similarity = image_similarity(image_path_1, image_path_2)
print(f"Image similarity: {similarity:.2f}%")
similarity = image_similarity(image_path_1, image_path_3)
print(f"Image similarity: {similarity:.2f}%")
