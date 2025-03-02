import cv2
import numpy as np


def bilateral_filter_smoothing(image, diameter=3, sigma_color=50, sigma_space=50):
    """
    Apply Bilateral Filtering to smooth an image while preserving edges.

    :param image: Input image (np.array).
    :param diameter: Diameter of the pixel neighborhood used during filtering.
    :param sigma_color: Filter sigma in color space.
    :param sigma_space: Filter sigma in coordinate space.
    :return: Smoothed image (np.array).
    """
    smoothed_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return smoothed_image


def check_diff(colour1, colour2, threshold=100):
    distance = np.linalg.norm(colour1 - colour2)  # Euclidean distance
    return distance > threshold  # There's a big difference


def apply_lidar_smoothing(image_path, lidar_path, output_path='images/white_image.jpg'):
    """Processes an image using LiDAR data to smooth out sections."""
    lidar = cv2.imread(lidar_path)
    image = cv2.imread(image_path)

    if image is None or lidar is None:
        raise ValueError("Error: One of the input images could not be loaded. Check file paths.")

    height, width, _ = image.shape
    white_image = np.full((height, width, 3), 255, dtype=np.uint8)
    full_width = image[:, :width]

    for row in range(500):  # Process the first 500 rows
        print(f"Processing row {row}")
        column = 0

        while column < width:
            start = column
            end = width  # Default to the end unless a boundary is found

            # Find the next boundary based on LiDAR differences
            for i in range(column, width - 1):
                if check_diff(lidar[row, i], lidar[row, i + 1]):
                    end = i + 1  # Include the differing pixel
                    break

            # Define the section
            if end - start > 16:
                section_start = start + 8
                section_end = end - 8
            else:
                section_start = start
                section_end = end

            section = full_width[row, section_start:section_end]

            # If section width > 3, use a 3×3 area centered on the middle pixel
            if section_end - section_start > 3:
                center_index = (section_start + section_end) // 2
                y_min = max(row - 1, 0)
                y_max = min(row + 2, height)
                x_min = max(center_index - 1, 0)
                x_max = min(center_index + 2, width)

                square_region = full_width[y_min:y_max, x_min:x_max]
                carrying_colour = np.mean(square_region, axis=(0, 1)).astype(int)
            else:
                carrying_colour = np.mean(section, axis=0).astype(int)

            # Apply the computed color to the section
            white_image[row, start:end] = carrying_colour
            column = end  # Move to next section

    # Save and return the processed image
    cv2.imwrite(output_path, white_image)
    return white_image


def rotate_image(image_path, output_path, rotation_flag=cv2.ROTATE_90_CLOCKWISE):
    """Rotates the image by 90 degrees and saves it."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Image could not be loaded. Check file path.")

    rotated_image = cv2.rotate(image, rotation_flag)
    cv2.imwrite(output_path, rotated_image)


def remove_jagged_streaks(image, kernel_size=2):
    """
    Apply morphological operations to clean up jagged streaks.

    :param image: Input image (np.array).
    :param kernel_size: Size of the kernel used for the operation.
    :return: Cleaned image (np.array).
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a square kernel
    # Perform erosion followed by dilation to remove jagged streaks
    eroded_image = cv2.erode(image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    return dilated_image

def grab_block(image):

    first_block = image[0:450, 0:450]

    cv2.imwrite('images/block.jpg', first_block)

    return first_block

# Step 1: Apply smoothing to the base images
#apply_lidar_smoothing('images/map_enlarged.jpg', 'images/image_converted.png', output_path='images/white_image.jpg')

# Optionally: Display the processed images
white_image = cv2.imread('images/white_image.jpg')

final_image = remove_jagged_streaks(white_image)

final_image = bilateral_filter_smoothing(final_image)



cv2.imshow('Smoothed and Rotated Image', grab_block(final_image))
cv2.waitKey(0)
cv2.destroyAllWindows()

