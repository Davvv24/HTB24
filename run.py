import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Open the TIF file
with rasterio.open("Viridien/Data/20230215_SE2B_CGG_GBR_MS4_L3_BGRN.tif") as src:
    red = src.read(3)  # Red band (Sentinel-2: B4)
    green = src.read(2)  # Green band (B3)
    blue = src.read(1)  # Blue band (B2)

# Normalize bands
def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

rgb = np.dstack((normalize(red), normalize(green), normalize(blue)))

# Display image
plt.imshow(rgb)
plt.title("Satellite Image RGB Composite")
plt.axis("off")
plt.show()
