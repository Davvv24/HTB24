import cv2
import rasterio
import numpy as np


# You can also work with the affine transformation and CRS if needed
## get all color codes
codes = [x for x in dir(cv2) if x.startswith("COLOR_")]

## print first three color codes
print(codes[:3])
# ['COLOR_BAYER_BG2BGR', 'COLOR_BAYER_BG2BGRA', 'COLOR_BAYER_BG2BGR_EA']

img = cv2.imread("images/high-res.jpg")
yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
cv2.imwrite("images/high_res_y_channel.png", yuv)

y_channel = yuv[:, :, 0]



# If you want to see the Y channel as an image:
cv2.imshow('Y Channel', y_channel)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('images/y_channel.png', y_channel)
