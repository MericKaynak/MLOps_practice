import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_california_housing():
    """Lädt den California Housing Datensatz und gibt ein DataFrame zurück."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df


def preprocess_data(df: pd.DataFrame):
    """Skaliert numerische Features und trennt Features von Labels."""
    X = df.drop("medhouseval", axis=1)
    y = df["medhouseval"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Teilt Daten in Training, Validierung und Test auf."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size),
                                                    random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_combined_to_csv(X_train, X_val, X_test, y_train, y_val, y_test, path="./data/"):
    """Speichert die gesplitteten Daten als kombinierte CSV-Dateien (X und y zusammen)."""
    os.makedirs(path, exist_ok=True)

    # Kombiniere X und y für jede Teilmenge
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Speichere die kombinierten Daten
    train_data.to_csv(f"{path}/train.csv", index=False)
    val_data.to_csv(f"{path}/validation.csv", index=False)
    test_data.to_csv(f"{path}/test.csv", index=False)


# === Ausführung ===
if __name__ == "__main__":
    df = load_california_housing()
    print("Rohdaten geladen:", df.shape)

    X, y = preprocess_data(df)
    print("Preprocessing abgeschlossen.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    save_combined_to_csv(X_train, X_val, X_test, y_train, y_val, y_test)
    print("Kombinierte Daten gespeichert in ./data/")