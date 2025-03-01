import math

import cv2
import numpy as np

from PIL import Image

# Open the TIFF image with Pillow
image_tif = Image.open('images/DSM_TQ0075_P_12757_20230109_20230315.tif')

# Convert the image to RGB (if it's in a different mode)
image_rgb = image_tif.convert('RGB')

# Save the image as PNG
image_rgb.save('images/image_converted.png')

# Save the image as JPEG
image_rgb.save('images/image_converted.jpg')

def check_uv_difference(UV1, UV2, threshold=0.1):
    # Calculate Euclidean distance between the two UV coordinates
    distance = math.sqrt((UV2[0] - UV1[0]) ** 2 + (UV2[1] - UV1[1]) ** 2)

    # Compare the distance with the threshold
    if distance > threshold:
        print(distance)
        return True  # There's a big difference
    else:
        return False  # The difference is small

# Load the Y and UV images
image_Y = cv2.imread('images/y_channel.png')  # Load Y channel image (second image)
image_UV = cv2.imread('images/high-res.jpg')   # Load UV channel image (first image)

image_Y = cv2.cvtColor(image_Y, cv2.COLOR_RGB2YUV)
image_UV = cv2.cvtColor(image_UV, cv2.COLOR_RGB2YUV)

y_channel = image_Y[:, :, 0]
u_channel = image_UV[:, :, 1]  # U is the second channel in YUV
v_channel = image_UV[:, :, 2]  # V is the third channel in YUV

# Combine the U and V channels into a UV image
uv_channel_3d = cv2.merge([y_channel, u_channel, v_channel])

# Display the individual channels and combined UV image
#cv2.imshow('Y Channel', y_channel)
#cv2.imshow('U Channel', u_channel)
#cv2.imshow('V Channel', v_channel)
#cv2.imshow('UV Combined', uv_channel_3d)
image_rgb = cv2.cvtColor(uv_channel_3d, cv2.COLOR_YUV2RGB)
#cv2.imshow('RGB Combined', image_rgb)

window_width = 800  # Set your desired window width
window_height = 600  # Set your desired window height
height, width = image_rgb.shape[:2]
scale_factor = min(window_width / width, window_height / height)
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)

resized_image = cv2.resize(image_rgb, (new_width, new_height))

cv2.imwrite('images/image_rgb.jpg', image_rgb)

cv2.imwrite('images/resized.jpg', resized_image)

#cv2.imshow('images/resized.jpg', resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the image
image = cv2.imread('images/image_rgb.jpg', 0)  # Read as grayscale


# Create a kernel for morphological operations
kernel = np.ones((5,5), np.uint8)


image = cv2.imread('images/image_rgb.jpg')  # Read as grayscale

# Apply Bilateral Filter
smoothed_image = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Filter', smoothed_image)

# Save or display the image
cv2.imwrite('images/bilateral_smoothed_image.jpg', smoothed_image)


# Load the Y and UV images
image_Y = cv2.imread('images/y_channel.png')  # Load Y channel image (second image)
image_UV = cv2.imread('images/bilateral_smoothed_image.jpg')   # Load UV channel image (first image)

image_Y = cv2.cvtColor(image_Y, cv2.COLOR_RGB2YUV)
image_UV = cv2.cvtColor(image_UV, cv2.COLOR_BGR2YUV)

y_channel = image_Y[:, :, 0]
u_channel = image_UV[:, :, 1]  # U is the second channel in YUV
v_channel = image_UV[:, :, 2]  # V is the third channel in YUV

# Combine the U and V channels into a UV image
uv_channel_3d = cv2.merge([y_channel, u_channel, v_channel])

image_rgb = cv2.cvtColor(uv_channel_3d, cv2.COLOR_YUV2BGR)

cv2.imshow('Y Channel', image_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()


'''
large_dim = image_Y.shape
small_dim = cv2.imread('images/low-res.jpg').shape
scale_factor = large_dim[0] / small_dim[0]
scale_factor = math.floor(scale_factor)


i = 0
while i < large_dim[0]-scale_factor:
    j=0
    while j < large_dim[1]-scale_factor:
        UV1 = YUV_image[i, j][1:3]
        UV2 = image_UV[i, j + scale_factor][1:3]
        if check_uv_difference(UV1, UV2):
            print("There is a big difference in the UV coordinates between the two pixels")
        j += scale_factor
    i += scale_factor

print(large_dim)

import cv2


codes = [x for x in dir(cv2) if x.startswith("COLOR_")]

img = cv2.imread("images/y_channel.png")
yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

y_channel = yuv[:, :, 0]

# If you want to see the Y channel as an image:
cv2.imshow('Y Channel', y_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''