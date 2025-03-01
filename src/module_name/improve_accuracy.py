import cv2
## get all color codes
codes = [x for x in dir(cv2) if x.startswith("COLOR_")]

## print first three color codes
print(codes[:3])
# ['COLOR_BAYER_BG2BGR', 'COLOR_BAYER_BG2BGRA', 'COLOR_BAYER_BG2BGR_EA']

## print all color codes
print(codes)