import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger le dataset
df = pd.read_csv("mini_mnist.csv")

# Mélanger le dataset
df = df.sample(frac=1).reset_index(drop=True)

# Séparer les features et les labels
labels = df["label"].values
features = df.drop(columns=["label"]).values

# Restructurer les données en images 12x6
images = features.reshape(-1, 12, 6)

# Fonction de visualisation
def visualize_samples(images, labels, num_samples=5):
    """Affiche quelques échantillons du dataset après mélange."""
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i], cmap='binary')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.show()

# Visualiser quelques échantillons après mélange
visualize_samples(images, labels)