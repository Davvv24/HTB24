import cv2
import numpy as np


def sharpen_with_outline(image, outline_image):
    """
    Sharpen the image using the high-resolution outline image.
    Sharpen areas where there is a red outline in the outline image.

    :param image: Low-resolution input image (BGR format).
    :param outline_image: High-resolution outline image (with red edges).
    :return: Sharpened image.
    """
    # Convert the outline image to grayscale and threshold it to create a binary mask
    outline_gray = cv2.cvtColor(outline_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(outline_gray, 1, 255, cv2.THRESH_BINARY)  # Binary mask

    # Sharpening kernel (using a simple kernel)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    # Apply sharpening filter to the original image
    sharpened_image = cv2.filter2D(image, -1, kernel)

    # Use the mask to apply the sharpening only where the outline is present
    sharpened_image = cv2.bitwise_and(sharpened_image, sharpened_image, mask=mask)

    # Combine the sharpened image with the original image using the mask
    result_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    result_image = cv2.add(result_image, sharpened_image)

    return result_image


def resize_image_to_outline(image, outline_image):
    """
    Resize the color image to match the high-resolution outline image.

    :param image: Low-resolution input image (BGR format).
    :param outline_image: High-resolution outline image (BGR format).
    :return: Resized color image.
    """
    height, width = outline_image.shape[:2]
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image


def grab_block(image):

    first_block = image[0:450, 0:450]

    cv2.imwrite('images/block.jpg', first_block)

    return first_block

def antialiasing(image, kernel_size=5):
    """
    Apply box blur-based antialiasing to smooth out jagged edges.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Load the color image (blurry) and the outline image (high-res)
image_path = 'images/map_enlarged.jpg'
outline_path = 'images/white_image.jpg'

image = cv2.imread(image_path)
outline_image = cv2.imread(outline_path)

# Resize the color image to match the outline resolution
resized_image = resize_image_to_outline(image, outline_image)

smoothed_image = antialiasing(resized_image)

smoothed_image = cv2.bilateralFilter(smoothed_image, 15, 75, 75)

# Sharpen the resized image using the outline image as a guide
sharpened_image = sharpen_with_outline(smoothed_image, outline_image)

# Save and display the results
cv2.imwrite('images/sharpened_image.jpg', sharpened_image)

cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

grab_block(sharpened_image)
