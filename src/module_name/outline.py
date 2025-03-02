import cv2
import numpy as np


def darken_with_outline(upscaled_image_path, outline_image_path, darkening_factor=0.5):
    # Load the images
    upscaled_image = cv2.imread(upscaled_image_path)
    outline_image = cv2.imread(outline_image_path)

    # Check if the images have the same size
    if upscaled_image.shape != outline_image.shape:
        raise ValueError("The upscaled image and outline image must have the same dimensions.")

    # Convert the outline image to a mask of red pixels (255, 0, 0)
    non_white_pixels_mask = np.any(outline_image != [255, 255, 255], axis=-1)  # Mask where pixels are not white
    # Darken the pixels in the upscaled image that correspond to the red pixels in the outline
    darkened_image = upscaled_image.copy()

    # Apply darkening: reduce the pixel intensity by the factor (can adjust based on how dark you want)
    darkened_image[non_white_pixels_mask] = darkened_image[non_white_pixels_mask] * (1 - darkening_factor)

    # Clip the values to stay within the valid range (0-255)
    darkened_image = np.clip(darkened_image, 0, 255).astype(np.uint8)

    # Save the final image
    cv2.imwrite('images/darkened_image.jpg', darkened_image)

    # Display the final result
    cv2.imshow('Darkened Image', darkened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Usage example:
upscaled_image_path = 'images/map_enlarged.jpg'
outline_image_path = 'images/white_image.jpg'
darken_with_outline(upscaled_image_path, outline_image_path)
