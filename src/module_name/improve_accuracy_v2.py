import cv2
import numpy as np

image_path = 'images/map_enlarged.jpg'
lidar_path = 'images/image_converted.png'
scale_factor = 3


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


# Takes the high res lidar image, and the image to improve, both in RGB,
# and returns the combined and smoothed image in RGB
def improve_accuracy(lidar_path, image_path, d, sigmaColor, sigmaSpace, end):
    image_Y = cv2.imread(lidar_path)
    image_UV = cv2.imread(image_path)

    image_Y = cv2.cvtColor(image_Y, cv2.COLOR_RGB2YUV)
    image_UV = cv2.cvtColor(image_UV, cv2.COLOR_RGB2YUV)

    y_channel = image_Y[:, :, 0]
    u_channel = image_UV[:, :, 1]
    v_channel = image_UV[:, :, 2]

    uv_channel_3d = cv2.merge([y_channel, u_channel, v_channel])
    image_rgb = cv2.cvtColor(uv_channel_3d, cv2.COLOR_YUV2RGB)

    if end:
        return image_rgb

    smoothed_image = cv2.bilateralFilter(image_rgb, d, sigmaColor, sigmaSpace)

    cv2.imwrite('images/unsmoothed_image.jpg', image_rgb)
    cv2.imwrite('images/smoothed_image.jpg', smoothed_image)

    return smoothed_image

def convert_lidar(lidar_path):
    lidar_image = cv2.imread(lidar_path, cv2.IMREAD_GRAYSCALE)


    # Normalize the image to the 0-255 range
    normalized_lidar = cv2.normalize(lidar_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_lidar = normalized_lidar.astype(np.uint8)  # Ensure it's in uint8 format for image display

    blurred_image = cv2.GaussianBlur(normalized_lidar, (21, 21), 0)


    edges = cv2.Canny(blurred_image, threshold1=10, threshold2=10)


    return edges


def method_1(image_Y, image_UV):
    # Resize image_UV to match the shape of image_Y
    image_UV_resized = cv2.resize(image_UV, (image_Y.shape[1], image_Y.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Apply joint bilateral filter using resized UV and Y channels
    enhanced_image = cv2.ximgproc.jointBilateralFilter(image_Y, image_UV_resized, d=9, sigmaColor=75, sigmaSpace=75)

    # Combine the enhanced image with the original UV image using weighted addition
    final_image = cv2.addWeighted(image_UV_resized, 0.7, enhanced_image, 0.3, 0)

    # Save the result
    cv2.imwrite('images/final_image.jpg', final_image)

    return final_image



def method_2(lidar_path, image_path):
    lidar = cv2.imread(lidar_path)
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (lidar.shape[1], lidar.shape[0]), interpolation=cv2.INTER_LINEAR)

    first_block = resized_image[0:50, 0:50]
    second_block = lidar[0:50, 0:50]

    show_image(first_block)
    show_image(second_block)



method_2(lidar_path, image_path)
#show_image(improve_accuracy_second(lidar_path, image_path, False))

#show_image(improve_accuracy(lidar_path, image_path, 2 * scale_factor, 80, scale_factor,False))
#show_image(improve_accuracy(lidar_path, "images/smoothed_image.jpg", 2 * scale_factor, 80, scale_factor, False))
#show_image(improve_accuracy(lidar_path, "images/smoothed_image.jpg", 2 * scale_factor, 80, scale_factor, False))
#show_image(improve_accuracy(lidar_path, "images/smoothed_image.jpg", 2 * scale_factor, 80, scale_factor, True))

#show_image(convert_lidar(lidar_path))

