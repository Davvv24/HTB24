import cv2
import numpy as np
from PIL import Image



def lidar_to_greyscale(lidar_image):
    """
    Convert a LiDAR image (depth data) to a greyscale image.
    Higher depth values will be mapped to lighter pixels (closer to white).

    :param lidar_image: The LiDAR image (2D numpy array, depth data).
    :return: A greyscale image (2D numpy array).
    """
    # Normalize the depth values to the range 0-255 (greyscale)
    lidar_min = np.min(lidar_image)
    lidar_max = np.max(lidar_image)

    # Normalize depth to 0-255 range
    normalized_lidar = (lidar_image - lidar_min) / (lidar_max - lidar_min) * 255

    # Convert to uint8 type
    greyscale_image = normalized_lidar.astype(np.uint8)

    return greyscale_image


# Load your LiDAR image (assuming it's a TIFF file with depth values)
lidar_path = 'images/DSM_TQ0075_P_12757_20230109_20230315.tif'
lidar = cv2.imread(lidar_path, cv2.IMREAD_UNCHANGED)  # Load as is (not color)

# If the image is read successfully, its shape should match (height, width)
if lidar is not None:
    print(f"Lidar image shape: {lidar.shape}")
else:
    print(f"Error loading LiDAR image from {lidar_path}")

# Convert LiDAR depth to greyscale
greyscale_image = lidar_to_greyscale(lidar)

# Save and display the result
output_path = 'images/lidar_greyscale.tif'
cv2.imwrite(output_path, greyscale_image)

# Open the TIFF image with Pillow
image_tif = Image.open(output_path)

# Convert the image to RGB (if it's in a different mode)
image_rgb = image_tif.convert('RGB')

# Save the image as PNG
image_rgb.save('images/image_converted_ge.png')

# Save the image as JPEG
image_rgb.save('images/image_converted_ge.jpg')
