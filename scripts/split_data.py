import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_file='mini_mnist.csv', train_file='train.csv', test_file='test.csv', test_size=0.2, random_state=42):
    # Charger le dataset
    df = pd.read_csv(input_file)
    
    # Séparer les features et les labels
    labels = df["label"].values
    features = df.drop(columns=["label"]).values

    # Diviser le dataset en train/test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # Sauvegarder les ensembles dans des fichiers CSV
    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train
    train_df.to_csv(train_file, index=False)

    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test
    test_df.to_csv(test_file, index=False)

    print(f'Dataset divisé : {train_file} et {test_file} créés.')

# Exécuter la fonction
if __name__ == "__main__":
    split_dataset()
