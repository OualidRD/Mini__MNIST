# Mini MNIST Classification Project

Ce projet vise à créer un modèle de classification pour un dataset personnalisé appelé **Mini MNIST**, qui représente des chiffres de 0 à 9 sous forme de vecteurs de 72 pixels. Le projet est divisé en trois devoirs principaux, chacun abordant une étape clé du processus de machine learning : génération de données, prétraitement, et classification avec un réseau de neurones.

---

## 📁 Structure du Projet

Le projet est organisé comme suit :
│── data/ # Contient les datasets (mini_mnist.csv, X_train.csv, etc.)
│── scripts/ # Contient tous les scripts Python
│ │── generate_data.py # Génération du dataset
│ │── preprocess.py # Visualisation et prétraitement
│ │── clustering.py # Clustering K-Means
│ │── evaluate.py # Évaluation statistique
│ │── split_data.py # Division en train/test
│── notebooks/ 
│ │── train_mini_mnist.ipynb # Notebook pour la classification
│── results/ # Résultats, images.
│── requirements.txt # Fichier des dépendances
│── README.md # Documentation du projet

## 📝 Description

**Génération du Dataset**
- **Objectif** : Créer un dataset appelé `mini_mnist.csv` contenant 1000 échantillons (100 par classe) de vecteurs de 72 pixels représentant les chiffres de 0 à 9.
- **Fichier** : `generate_data.py`
- **Fonctionnalités** :
  - Génération de vecteurs aléatoires de 72 pixels.
  - Attribution de labels (0 à 9) à chaque échantillon.
  - Sauvegarde du dataset au format CSV.

**Prétraitement et Clustering**
- **Objectif** : Prétraiter le dataset, visualiser les données, appliquer un clustering, et diviser le dataset en ensembles d'entraînement et de test.
- **Fichiers** :
  - `preprocess.py` : Visualisation des données.
  - `clustering.py` : Application du clustering K-Means.
  - `evaluate.py` : Évaluation de la qualité du clustering.
  - `split_data.py` : Division du dataset en train/test.
- **Fonctionnalités** :
  - Visualisation des images générées.
  - Clustering des données en 10 clusters.
  - Évaluation de la correspondance entre clusters et labels.
  - Division des données en ensembles d'entraînement (80%) et de test (20%).

**Classification avec PyTorch**
- **Objectif** : Créer un modèle de classification avec PyTorch pour prédire les labels des images.
- **Fichier** : `train_mini_mnist.ipynb`
- **Architecture du modèle** :
  - **Couche d'entrée** : 72 neurones.
  - **Couche cachée 1** : 20 neurones (activation ReLU).
  - **Couche cachée 2** : 10 neurones (activation ReLU).
  - **Couche de sortie** : 10 neurones (activation Softmax).
- **Fonctionnalités** :
  - Définition de l'architecture du modèle.
  - Entraînement du modèle avec SGD (descente de gradient stochastique).
  - Évaluation de la précision sur le test set.
  - Visualisation de la courbe de perte.
  - Test du modèle sur des exemples aléatoires.

  ## 🛠 Installation et Exécution

### **1. Installer les dépendances**
Assure-toi d'avoir Python 3.8+ installé. Ensuite, installe les dépendances nécessaires avec la commande suivante :

```bash
pip install -r requirements.txt
