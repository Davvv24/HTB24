import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Device configuration (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model checkpoint
checkpoint = torch.load("best_resnet18_eurosat_checkpoint.pth", map_location=device)
num_classes = checkpoint['num_classes']  # Retrieve number of classes

# Load the ResNet model (must match training architecture)
model = models.resnet18(pretrained=False)  # Set to False since we load our trained weights
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Match modified architecture
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Match the number of classes

# Load the model weights
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()  # Set to evaluation mode
print("Model successfully loaded!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")  # Convert image to RGB (ResNet expects 3 channels)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

def predict_image(image_path, model, class_names):
    image = preprocess_image(image_path)  # Preprocess the input image

    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(image)  # Get model predictions
        probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        predicted_class = torch.argmax(probabilities).item()  # Get the class with highest probability

    return class_names[predicted_class], probabilities[0][predicted_class].item()  # Return label and confidence


class_names = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
               "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]  # Adjust if needed
class_colours_list = [
    (255, 0, 0),     # AnnualCrop -> Red
    (0, 255, 0),     # Forest -> Green
    (0, 0, 255),     # HerbaceousVegetation -> Blue
    (255, 255, 0),   # Highway -> Yellow
    (255, 165, 0),   # Industrial -> Orange
    (128, 0, 128),   # Pasture -> Purple
    (0, 255, 255),   # PermanentCrop -> Cyan
    (255, 192, 203), # Residential -> Pink
    (0, 0, 0),       # River -> Black
    (135, 206, 250)  # SeaLake -> Light Blue
]
class_colours = {class_names[i]: class_colours_list[i] for i in range(len(class_names))}

def preprocess_tile(image_path, tile_size, tile_number_x, tile_number_y):

    image = Image.open(image_path).convert("RGB")  # Open and convert to RGB

    width, height = image.size

    top_right_crop = image.crop((tile_size * tile_number_x, tile_size * tile_number_y, tile_size * (tile_number_x +1), tile_size * (tile_number_y +1)))  # (left, top, right, bottom)
    # Display cropped image before resizing
    # plt.imshow(top_right_crop)
    # plt.axis("off")  # Hide axes
    # plt.title("Cropped Tile (top-left)")
    # plt.show()  # Show the cropped tile

    # Apply transformations
    image_tensor = transform(top_right_crop).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)

def predict_tile(image_path, model, class_names, tile_size = 1000, tile_number_x = 0, tile_number_y = 0):

    image_tensor = preprocess_tile(image_path, tile_size, tile_number_x, tile_number_y)  # Extract and preprocess tile

    with torch.no_grad():  # No gradients for inference
        outputs = model(image_tensor)  # Get model predictions
        probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        predicted_class = torch.argmax(probabilities).item()  # Get highest probability class

    return class_names[predicted_class], probabilities[0][predicted_class].item()  # Return label & confidence

# sliding window to go over the whole satellite image
def predict_tiles(image_path, model, class_names, tile_size):

    tile_number_x, tile_number_y, counter = 0, 0, 0
    image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
    predictions = []
    width, height = image.size
    number_of_windows = width // tile_size
    for y in range(number_of_windows):
        for x in range(number_of_windows):
            print("EXTRACTING TILE NUMBER: ", counter)
            counter+=1
            predicted_class, confidence = predict_tile(image_path, model, class_names, tile_size, x, y)
            predictions.append([predicted_class, confidence])

    return predictions

def create_prediction_grid(predictions, class_colours, grid_size=5, tile_size=50):
    """
    Creates a 5x5 grid image where each tile is colored based on the predicted class.

    :param predictions: List of dictionaries containing 'predicted_class' keys.
    :param class_colours: Dictionary mapping class labels to RGB colors.
    :param grid_size: Size of the grid (default is 5x5).
    :param tile_size: Size of each tile in pixels (default is 50x50).
    :return: Generated PIL image.
    """
    # Create a blank white image
    img_size = grid_size * tile_size
    grid_image = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(grid_image)

    # Loop through predictions and fill the grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Get the prediction index (row-major order)
            index = i * grid_size + j

            if index < len(predictions):  # Ensure index is within range
                pred_class = predictions[index][0]
                color = class_colours.get(pred_class, (255, 255, 255))  # Default to white if class not found

                # Draw the colored tile
                left = j * tile_size
                top = i * tile_size
                right = left + tile_size
                bottom = top + tile_size
                draw.rectangle([left, top, right, bottom], fill=color)

    return grid_image

def add_legend(grid_image, class_colours, tile_size=50, legend_tile_size=40):
    """
    Adds a legend to the bottom of the grid showing class names and corresponding colors.

    :param grid_image: The original grid image.
    :param class_colours: Dictionary of class label -> RGB color.
    :param tile_size: Size of tiles in the legend.
    :param legend_tile_size: Size of legend tiles.
    :return: Image with the legend added.
    """
    grid_width, grid_height = grid_image.size
    legend_height = len(class_colours) * legend_tile_size
    new_img = Image.new("RGB", (grid_width, grid_height + legend_height), (255, 255, 255))
    new_img.paste(grid_image, (0, 0))

    draw = ImageDraw.Draw(new_img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)  # Load font
    except IOError:
        font = ImageFont.load_default()

    # Draw legend
    for idx, (class_name, color) in enumerate(class_colours.items()):
        left = 10
        top = grid_height + idx * legend_tile_size
        right = left + legend_tile_size
        bottom = top + legend_tile_size

        # Draw color box
        draw.rectangle([left, top, right, bottom], fill=color)

        # Draw text next to color box
        draw.text((right + 10, top), class_name, fill=(0, 0, 0), font=font)  # Black text for readability

    return new_img

# Test on a sample image
# image_path = "2750_tests/Residential_226.jpg"
# predicted_class, confidence = predict_image(image_path, model, class_names)
image_path = "2750_tests/map_enlarged.jpg"
# predicted_class, confidence = predict_tile(image_path, model, class_names)
# print(f"Predicted Land Use: {predicted_class} ({confidence*100:.2f}%)")
predictions = predict_tiles(image_path, model, class_names, 1000)
grid_img = create_prediction_grid(predictions, class_colours, grid_size=5, tile_size=100)
# grid_img.show()
# grid_img.save("prediction_grid.png")
grid_with_legend = add_legend(grid_img, class_colours)
grid_with_legend.show()
grid_with_legend.save("prediction_grid_with_legend.png")




