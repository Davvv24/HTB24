import cv2
import rasterio
import numpy as np
# Open the TIFF file with rasterio
with rasterio.open('images/20230215_SE2B_CGG_GBR_MS4_L3_BGRN.tif') as src:
    # Read the image data as a numpy array (first band)
    img_data = src.read(1)

    # If the image has multiple bands (e.g. RGB), you might want to read all bands:
    # img_data = src.read([1, 2, 3])  # Read Red, Green, and Blue channels

    # Convert the image to a format OpenCV can handle (e.g. 8-bit)
    img_data = np.uint8(img_data)

    # If it's a 1-band image and you want to display it as grayscale, proceed:
    # Display the image using OpenCV
    cv2.imshow('TIFF Image', img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
# You can also work with the affine transformation and CRS if needed
## get all color codes
codes = [x for x in dir(cv2) if x.startswith("COLOR_")]

## print first three color codes
print(codes[:3])
# ['COLOR_BAYER_BG2BGR', 'COLOR_BAYER_BG2BGRA', 'COLOR_BAYER_BG2BGR_EA']

## print all color codes
print(codes)
image = cv2.imread("images/shell.png")
print(image.shape)  # Returns (height, width, channels)
img = cv2.imread("images/shell.png")
yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
cv2.imwrite("images/shell_b_w_yuv.png", yuv)

y_channel = yuv[:, :, 0]

# If you want to see the Y channel as an image:
cv2.imshow('Y Channel', y_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, you can save the Y channel as an image
cv2.imwrite('y_channel.png', y_channel)

# Read the TIFF image
image = cv2.imread('images/20230215_SE2B_CGG_GBR_MS4_L3_BGRN.tif', cv2.IMREAD_UNCHANGED)

# Check if the image was loaded successfully
if image is not None:
    resized_image = cv2.resize(image, (800, 600))
    # Display the image
    cv2.imshow('TIFF Image', resized_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load the image")


# Open your TIFF file
dataset = gdal.Open('images/20230215_SE2B_CGG_GBR_MS4_L3_BGRN.tif')

# Check if the file was opened successfully
if dataset is None:
    print("Failed to open the file.")
else:
    print(f"File opened successfully: {dataset.RasterXSize}x{dataset.RasterYSize}")
'''