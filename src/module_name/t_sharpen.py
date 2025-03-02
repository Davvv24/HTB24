import cv2
import numpy as np


def lidar_pan_sharpen(lidar_path, rgb_path):
    # Load RGB and LiDAR images
    image_rgb = cv2.imread(rgb_path)
    image_lidar = cv2.imread(lidar_path, cv2.IMREAD_GRAYSCALE)  # LiDAR intensity or depth

    image_rgb = cv2.resize(image_rgb, (image_lidar.shape[1], image_lidar.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Convert RGB to YUV color space
    dark_source = image_rgb
    dark_source_yuv = cv2.cvtColor(dark_source, cv2.COLOR_BGR2YUV)
    dark_y = dark_source_yuv[:, :, 0]

    # Percentile thresholds for darkest and lightest pixels
    percentile_dark = 10
    percentile_light = 90  # Top 30% lightest pixels

    threshold_dark = np.percentile(dark_y, percentile_dark)
    threshold_light = np.percentile(dark_y, percentile_light)

    # Create masks for dark and light pixels
    dark_mask = dark_y <= threshold_dark
    light_mask = dark_y >= threshold_light

    # Convert RGB to YUV color space for enhancement
    image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2YUV)

    # Extract Y (luminance) channel
    y_channel = image_yuv[:, :, 0].astype(np.float32)

    # Normalize LiDAR intensity to match Y channel range (0-255)
    lidar_normalized = cv2.normalize(image_lidar, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

    # Enhance the Y channel using LiDAR data (weighted blending)
    alpha = 0.6  # Control the influence of LiDAR data
    enhanced_y = cv2.addWeighted(y_channel, 1 - alpha, lidar_normalized, alpha, 0).astype(np.uint8)

    # Merge back enhanced Y channel with original UV channels
    image_yuv[:, :, 0] = enhanced_y
    enhanced_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    # Replace dark pixels in the enhanced image with corresponding pixels from the dark source
    enhanced_rgb_yuv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_BGR2YUV)
    en_y = enhanced_rgb_yuv[:, :, 0]

    target_threshold_dark = 200  # Only replace pixels that are darker than this
    target_threshold_light = 55  # Only replace pixels that are lighter than this

    target_modified = en_y.copy()

    # Replace dark pixels (those below dark threshold)
    target_mask_dark = dark_mask & (en_y <= target_threshold_dark)
    target_modified[target_mask_dark] = dark_y[target_mask_dark]

    # Replace light pixels (those above light threshold)
    target_mask_light = light_mask & (en_y >= target_threshold_light)
    target_modified[target_mask_light] = dark_y[target_mask_light]

    # Update the Y channel with the modified values
    image_yuv[:, :, 0] = target_modified

    # Convert back to BGR
    target_modified = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

    # Save and return the result
    cv2.imwrite('images/pan_sharpened_lidar.jpg', target_modified)
    return enhanced_rgb

def grab_block(image_path):
    image = cv2.imread(image_path)

    first_block = image[0:450, 0:450]

    cv2.imwrite('images/block.jpg', first_block)


def show_image(image):
    window_width = 800
    window_height = 600
    height, width = image.shape[:2]
    scale_factor = min(window_width / width, window_height / height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    cv2.imshow('image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
pan_sharpened = lidar_pan_sharpen('images/image_converted.jpg', 'images/map.jpg')
grab_block('images/pan_sharpened_lidar.jpg')


