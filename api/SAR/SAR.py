import os
import random
import numpy as np
import pandas as pd
from skimage import io, color
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import plotly.express as px


def visualize_sar_image(image_path: str):
    """
    Load a SAR image and display an interactive plot where hovering
    shows pixel coordinates and backscatter intensity
    """
    image = io.imread(image_path)

    if image.ndim == 3:
        image = color.rgb2gray(image)

    height, width = image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    intensity = image.flatten()
    df = pd.DataFrame({
        'x': x.flatten(),
        'y': y.flatten(),
        'intensity': intensity
    })

    # Create the plot
    fig = px.scatter(
        df, x='x', y='y', color='intensity',
        color_continuous_scale='gray',
        labels={'intensity': 'Backscatter Intensity'}
    )
    fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
    fig.update_layout(title=f'SAR Image - {os.path.basename(image_path)} (Hover to see intensity)')

    fig.show()


def analyze_sar_image(image_path: str):
    """
    Load a SAR image, segment land vs water, and show results
    """
    # Load image
    image = io.imread(image_path)
    if image.ndim == 3:
        image = color.rgb2gray(image)

    # Otsu threshold
    thresh = threshold_otsu(image)
    mask = image > thresh  # land=True, water=False

    # Percentages
    land_pct = np.sum(mask) / mask.size * 100
    water_pct = 100 - land_pct

    print(f"Estimated Land: {land_pct:.2f}% | Water: {water_pct:.2f}%")

    return {
        'land': f"{land_pct:.2f}%",
        'water': f"{water_pct:.2f}%"
    }


def visualize_random_image(folder_path: str):
    """
    Pick a random SAR image and visualize it.
    """
    # List all .bmp files
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.bmp')]
    if not image_files:
        raise ValueError("No .bmp images found in the folder!")

    # Pick one random image
    random_image = random.choice(image_files)
    image_path = os.path.join(folder_path, random_image)

    print(f"Selected image: {random_image}")

    # First visualization (interactive hover)
    visualize_sar_image(image_path)

    # Then analysis (land vs water segmentation)
    analyze_sar_image(image_path)



folder_path = "https://drive.google.com/drive/folders/1F9j3WH9nJQavy67GAN2gIPaoPR8hN6vX"

