import math
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from pyproj import Transformer

# -----------------------------
# Helper Functions for TFW & OSGB Conversion
# -----------------------------
def read_tfw(tfw_path):
    """
    Reads a TFW (world file) and returns the six parameters:
    A: pixel size in X direction
    D: rotation term (usually 0)
    B: rotation term (usually 0)
    E: pixel size in Y direction (usually negative for north-up images)
    C: X coordinate of center of upper-left pixel
    F: Y coordinate of center of upper-left pixel
    """
    with open(tfw_path, "r") as f:
        A = float(f.readline().strip())
        D = float(f.readline().strip())
        B = float(f.readline().strip())
        E = float(f.readline().strip())
        C = float(f.readline().strip())
        F = float(f.readline().strip())
    return A, D, B, E, C, F

def pixel_to_osgb(tfw_path, i, j):
    """
    Given pixel coordinates (i, j) (i: column, j: row),
    returns the corresponding OSGB easting and northing using the TFW parameters.
    """
    A, D, B, E, C, F = read_tfw(tfw_path)
    osgb_easting = C + A * i + B * j
    osgb_northing = F + D * i + E * j
    return osgb_easting, osgb_northing

def composite_center_latlon(tfw_path, composite_width, composite_height):
    """
    Computes the latitude/longitude for the center of an image of size composite_width x composite_height.
    The TFW file is assumed to be in OSGB36 (EPSG:27700) and is converted to WGS84 (EPSG:4326).
    """
    i_center = composite_width / 2.0
    j_center = composite_height / 2.0
    osgb_easting, osgb_northing = pixel_to_osgb(tfw_path, i_center, j_center)
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(osgb_easting, osgb_northing)
    return lat, lon

def compute_zoom_from_tfw(tfw_path, center_lat):
    """
    Computes the appropriate Google Maps zoom level based on the TFW pixel resolution and the center latitude.
    The Google Maps ground resolution (m/px) is approximately:
        resolution = 156543.03392 * cos(latitude) / (2^zoom)
    Here, we solve for zoom using the absolute pixel size (in meters) from the TFW file.
    """
    A, _, _, _, _, _ = read_tfw(tfw_path)
    desired_resolution = abs(A)  # meters per pixel from TFW
    zoom_float = math.log2(156543.03392 * math.cos(math.radians(center_lat)) / desired_resolution)
    return zoom_float

# -----------------------------
# Helper Functions for Mercator Projection
# -----------------------------
def latlon_to_pixel(lat, lon, zoom):
    """
    Converts latitude and longitude to global pixel coordinates at a given zoom level
    using the Mercator projection.
    """
    siny = math.sin(lat * math.pi / 180.0)
    siny = min(max(siny, -0.9999), 0.9999)
    world_size = 256 * (2 ** zoom)
    pixel_x = (lon + 180) / 360 * world_size
    pixel_y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * world_size
    return pixel_x, pixel_y

def pixel_to_latlon(pixel_x, pixel_y, zoom):
    """
    Converts global pixel coordinates to latitude and longitude at a given zoom level.
    """
    world_size = 256 * (2 ** zoom)
    lon = pixel_x / world_size * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * pixel_y / world_size))))
    return lat, lon

# -----------------------------
# Configuration
# -----------------------------
tfw_file_path = "Viridien/Data/DSM_TQ0075_P_12757_20230109_20230315.tfw"
tif_image_path = "Viridien/Data/DSM_TQ0075_P_12757_20230109_20230315.tif"
converted_png_path = "Viridien/Data/DSM_TQ0075_P_12757_20230109_20230315.png"

api_key = "AIzaSyDZu4b6tIrbl7OTXs8mNVexCsCV6rjdGvs"
maptype = "satellite"   # Satellite view
tile_size = 500         # Each tile will be 500x500 pixels
grid_count = 10         # 10x10 grid => composite image is 5000x5000 pixels
composite_size = tile_size * grid_count  # 5000 pixels

# -----------------------------
# Step 1: Convert TIF to PNG (Colour)
# -----------------------------
print("Converting TIF to PNG (Colour)...")
tif_image_pil = Image.open(tif_image_path)
tif_image_pil = tif_image_pil.convert("RGB")
tif_image_pil.save(converted_png_path, "PNG")
print(f"Converted TIF saved as {converted_png_path}")

tif_image = cv2.imread(converted_png_path)
if tif_image is None:
    raise IOError("Failed to load the converted PNG image.")

# -----------------------------
# Step 2: Determine Composite Center & Compute Appropriate Zoom
# -----------------------------
print("Computing composite center and appropriate zoom level...")

# Compute composite center (lat, lon) using OSGB conversion
center_lat, center_lon = composite_center_latlon(tfw_file_path, composite_size, composite_size)
print(f"Composite Center (lat, lon): ({center_lat:.6f}, {center_lon:.6f})")

# Compute zoom from TFW pixel resolution and center latitude
computed_zoom = compute_zoom_from_tfw(tfw_file_path, center_lat)
zoom = round(computed_zoom)
print(f"Computed zoom level (float): {computed_zoom:.2f} -> using zoom: {zoom}")

# Convert composite center to global pixel coordinates at the computed zoom
center_pixel_x, center_pixel_y = latlon_to_pixel(center_lat, center_lon, zoom)
top_left_pixel_x = center_pixel_x - composite_size / 2.0
top_left_pixel_y = center_pixel_y - composite_size / 2.0

# -----------------------------
# Step 3: Build the 5000x5000 Composite Google Satellite Image
# -----------------------------
print("Building composite Google satellite image...")
composite_image = np.zeros((composite_size, composite_size, 3), dtype=np.uint8)

for i in range(grid_count):      # columns
    for j in range(grid_count):  # rows
        tile_center_offset_x = i * tile_size + tile_size / 2.0
        tile_center_offset_y = j * tile_size + tile_size / 2.0
        tile_center_pixel_x = top_left_pixel_x + tile_center_offset_x
        tile_center_pixel_y = top_left_pixel_y + tile_center_offset_y

        tile_lat, tile_lon = pixel_to_latlon(tile_center_pixel_x, tile_center_pixel_y, zoom)
        tile_url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={tile_lat},{tile_lon}&zoom={zoom}&size={tile_size}x{tile_size}"
            f"&maptype={maptype}&key={api_key}"
        )
        print(f"Fetching tile ({i}, {j}) at center ({tile_lat:.6f}, {tile_lon:.6f})")
        response = requests.get(tile_url)
        if response.status_code == 200:
            tile_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            tile_image = cv2.imdecode(tile_array, cv2.IMREAD_COLOR)
            if tile_image is None:
                print(f"Failed to decode tile at ({i}, {j}).")
                continue
        else:
            print(f"Failed to fetch tile at ({i}, {j}): HTTP {response.status_code}")
            continue

        x_start = i * tile_size
        x_end = x_start + tile_size
        y_start = j * tile_size
        y_end = y_start + tile_size
        composite_image[y_start:y_end, x_start:x_end, :] = tile_image

composite_output_file = "google_composite_5000x5000.png"
cv2.imwrite(composite_output_file, composite_image)
print(f"Composite Google satellite image saved as {composite_output_file}")

# -----------------------------
# Step 4: Compare the Two Colour Images
# -----------------------------
tif_resized = cv2.resize(tif_image, (composite_size, composite_size))
google_image = composite_image

ssim_index, _ = ssim(tif_resized, google_image, full=True, channel_axis=-1)
mse = np.mean((tif_resized.astype("float") - google_image.astype("float")) ** 2)
diff_image = cv2.absdiff(tif_resized, google_image)

print(f"Colour SSIM: {ssim_index:.4f}")
print(f"Colour MSE: {mse:.2f}")

# -----------------------------
# Step 5: Display the Images (Convert BGR to RGB for Matplotlib)
# -----------------------------
tif_display = cv2.cvtColor(tif_resized, cv2.COLOR_BGR2RGB)
google_display = cv2.cvtColor(google_image, cv2.COLOR_BGR2RGB)
diff_display = cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.title("Converted TIF (Resized, Colour)")
plt.imshow(tif_display)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Google Composite Image (Colour)")
plt.imshow(google_display)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Colour Difference Map")
plt.imshow(diff_display)
plt.axis("off")

plt.tight_layout()
plt.show()