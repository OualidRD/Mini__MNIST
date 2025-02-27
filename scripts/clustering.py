import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Charger et prétraiter les données
df = pd.read_csv("mini_mnist.csv")

# Séparer les labels et les features
labels = df["label"].values
features = df.drop(columns=["label"]).values

# Appliquer K-Means avec 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features)

# Réduction de dimension avec PCA pour visualisation (2D)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Affichage des clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='tab10', alpha=0.5)
plt.colorbar(scatter, label="Cluster ID")
plt.title("Visualisation du Clustering K-Means avec PCA")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")

# Ajouter les labels à chaque point
for i, (x, y) in enumerate(features_pca):
    plt.text(x, y, str(labels[i]), fontsize=8, ha='center', va='center')

plt.show()