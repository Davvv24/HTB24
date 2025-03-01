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
from PIL import Image
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

def preprocess_tile(image_path):
    image = Image.open(image_path).convert("RGB")  # Open and convert to RGB

    # Crop the top-right 64x64 tile from 5000x5000 image
    width, height = image.size  # Should be 5000x5000

    top_right_crop = image.crop((0, 0, 672, 672))  # (left, top, right, bottom)
    # Display cropped image before resizing
    plt.imshow(top_right_crop)
    plt.axis("off")  # Hide axes
    plt.title("Cropped Tile (top-left)")
    plt.show()  # Show the cropped tile

    # Apply transformations
    image_tensor = transform(top_right_crop).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)

def predict_tile(image_path, model, class_names):
    image_tensor = preprocess_tile(image_path)  # Extract and preprocess tile

    with torch.no_grad():  # No gradients for inference
        outputs = model(image_tensor)  # Get model predictions
        probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        predicted_class = torch.argmax(probabilities).item()  # Get highest probability class

    return class_names[predicted_class], probabilities[0][predicted_class].item()  # Return label & confidence


# Test on a sample image
# image_path = "2750_tests/HerbaceousVegetation_2999.jpg"
# predicted_class, confidence = predict_image(image_path, model, class_names)
image_path = "2750_tests/map_enlarged.jpg"
predicted_class, confidence = predict_tile(image_path, model, class_names)
print(f"Predicted Land Use: {predicted_class} ({confidence*100:.2f}%)")