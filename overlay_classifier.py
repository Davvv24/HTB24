from PIL import Image


def overlay_images(base_image_path, overlay_image_path, output_path, alpha=0.5):
    """
    Overlays an image (`overlay_image_path`) on top of a base image (`base_image_path`)
    with a given transparency (`alpha`).

    :param base_image_path: Path to the base image (e.g., map_enlarged.jpg).
    :param overlay_image_path: Path to the overlay image (e.g., prediction_grid.jpg).
    :param output_path: Path to save the blended image.
    :param alpha: Transparency level for the overlay (0 = invisible, 1 = fully opaque).
    """
    # Open base and overlay images
    base_image = Image.open(base_image_path).convert("RGB")
    overlay_image = Image.open(overlay_image_path).convert("RGBA")  # Ensure overlay has an alpha channel

    # Ensure both images are the same size
    if base_image.size != overlay_image.size:
        overlay_image = overlay_image.resize(base_image.size)

    # Apply transparency to overlay image
    overlay_with_alpha = overlay_image.copy()
    overlay_with_alpha.putalpha(int(alpha * 255))  # Convert alpha (0-1) to (0-255)

    # Blend images together
    blended_image = Image.alpha_composite(base_image.convert("RGBA"), overlay_with_alpha)

    # Save the final image
    blended_image.convert("RGB").save(output_path)

    print(f"Overlay saved as: {output_path}")
    blended_image.show()  # Show the final image


# File paths
base_image_path = "2750_tests/map_enlarged.jpg"  # Background image
overlay_image_path = "prediction_grid.png"  # Overlay image
output_path = "overlayed_map.jpg"  # Output image path

# Overlay images with 50% transparency
overlay_images(base_image_path, overlay_image_path, output_path, alpha=0.5)
