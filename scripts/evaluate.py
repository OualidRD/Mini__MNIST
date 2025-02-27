import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(filepath):
    # Charger les données à partir d'un fichier CSV
    return pd.read_csv(filepath)

def preprocess_data(data):
    # Séparer les caractéristiques et les étiquettes
    X = data.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y = data.iloc[:, -1].values   # La dernière colonne
    return X, y

def evaluate_model(X, y):
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle k-NN
    model = KNeighborsClassifier(n_neighbors=3)  # Vous pouvez ajuster n_neighbors ici
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Afficher le rapport de classification et la matrice de confusion
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    data_file = 'mini_mnist.csv'  
    data = load_data(data_file)
    X, y = preprocess_data(data)
    evaluate_model(X, y)
