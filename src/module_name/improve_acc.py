import cv2
import numpy as np


def bilateral_filter_smoothing(image, diameter=3, sigma_color=50, sigma_space=50):

    smoothed_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return smoothed_image


def check_diff(colour1, colour2, threshold=100):
    distance = np.linalg.norm(colour1 - colour2)
    return distance > threshold


def apply_lidar_smoothing(image_path, lidar_path, output_path='images/white_image.jpg'):
    lidar = cv2.imread(lidar_path)
    image = cv2.imread(image_path)

    if image is None or lidar is None:
        raise ValueError("Error: One of the input images could not be loaded. Check file paths.")

    height, width, _ = image.shape
    white_image = np.full((height, width, 3), 255, dtype=np.uint8)
    full_width = image[:, :width]

    for row in range(5000):
        print(f"Processing row {row}")
        column = 0

        while column < width:
            start = column
            end = width

            for i in range(column, width - 1):
                if check_diff(lidar[row, i], lidar[row, i + 1]):
                    end = i + 1
                    break

            if end - start > 16:
                section_start = start + 8
                section_end = end - 8
            else:
                section_start = start
                section_end = end

            section = full_width[row, section_start:section_end]

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

            white_image[row, start:end] = carrying_colour
            column = end

    cv2.imwrite(output_path, white_image)
    return white_image


def rotate_image(image_path, output_path, rotation_flag=cv2.ROTATE_90_CLOCKWISE):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Image could not be loaded. Check file path.")

    rotated_image = cv2.rotate(image, rotation_flag)
    cv2.imwrite(output_path, rotated_image)


def remove_jagged_streaks(image, kernel_size=5):

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    return dilated_image

def grab_block(image):

    first_block = image[0:450, 0:450]

    cv2.imwrite('images/block.jpg', first_block)

    return first_block


apply_lidar_smoothing('images/map_enlarged.jpg', 'images/image_converted.png', output_path='images/white_image.jpg')

white_image = cv2.imread('images/white_image.jpg')

final_image = white_image

final_image = remove_jagged_streaks(final_image)

cv2.imwrite('images/MEOWMEOW.jpg', final_image)

apply_lidar_smoothing('images/MEOWMEOW.jpg', 'images/image_converted.png', output_path='images/white_image.jpg')

white_image = cv2.imread('images/white_image.jpg')

final_image = white_image

final_image = remove_jagged_streaks(final_image)

cv2.imwrite('images/MEOWMEOW.jpg', final_image)

cv2.imshow('Smoothed and Rotated Image', grab_block(final_image))
cv2.waitKey(0)
cv2.destroyAllWindows()

