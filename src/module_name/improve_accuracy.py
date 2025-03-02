import math

import cv2
import numpy as np

from PIL import Image

image_tif = Image.open('images/DSM_TQ0075_P_12757_20230109_20230315.tif')

image_rgb = image_tif.convert('RGB')

image_rgb.save('images/image_converted.png')

image_rgb.save('images/image_converted.jpg')

def check_uv_difference(UV1, UV2, threshold=0.1):
    distance = math.sqrt((UV2[0] - UV1[0]) ** 2 + (UV2[1] - UV1[1]) ** 2)

    if distance > threshold:
        print(distance)
        return True
    else:
        return False

image_Y = cv2.imread('images/y_channel.png')
image_UV = cv2.imread('images/high-res.jpg')

image_Y = cv2.cvtColor(image_Y, cv2.COLOR_RGB2YUV)
image_UV = cv2.cvtColor(image_UV, cv2.COLOR_RGB2YUV)

y_channel = image_Y[:, :, 0]
u_channel = image_UV[:, :, 1]
v_channel = image_UV[:, :, 2]

uv_channel_3d = cv2.merge([y_channel, u_channel, v_channel])


image_rgb = cv2.cvtColor(uv_channel_3d, cv2.COLOR_YUV2RGB)

window_width = 800
window_height = 600
height, width = image_rgb.shape[:2]
scale_factor = min(window_width / width, window_height / height)
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)

resized_image = cv2.resize(image_rgb, (new_width, new_height))

cv2.imwrite('images/image_rgb.jpg', image_rgb)

cv2.imwrite('images/resized.jpg', resized_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

image = cv2.imread('images/image_rgb.jpg', 0)  # Read as grayscale


kernel = np.ones((5,5), np.uint8)


image = cv2.imread('images/image_rgb.jpg')

smoothed_image = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Filter', smoothed_image)

cv2.imwrite('images/bilateral_smoothed_image.jpg', smoothed_image)


image_Y = cv2.imread('images/y_channel.png')
image_UV = cv2.imread('images/bilateral_smoothed_image.jpg')

image_Y = cv2.cvtColor(image_Y, cv2.COLOR_RGB2YUV)
image_UV = cv2.cvtColor(image_UV, cv2.COLOR_BGR2YUV)

y_channel = image_Y[:, :, 0]
u_channel = image_UV[:, :, 1]
v_channel = image_UV[:, :, 2]

uv_channel_3d = cv2.merge([y_channel, u_channel, v_channel])

image_rgb = cv2.cvtColor(uv_channel_3d, cv2.COLOR_YUV2BGR)

cv2.imshow('Y Channel', image_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()

