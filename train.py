import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from wettbewerb import EEGDataset
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd

# Funktion zum Extrahieren von Features aus rohen EEG-Daten
def extract_features(data, fs):
    features = []
    for channel_data in data:
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        line_length = np.sum(np.abs(np.diff(channel_data)))
        features.extend([mean, std, line_length])
    return np.array(features)

# Funktion zum Verarbeiten eines Batches
def process_batch(dataset, start_idx, end_idx):
    batch_features = []
    batch_labels = []
    for i in range(start_idx, end_idx):
        id, channels, data, fs, ref_system, label = dataset[i]
        features = extract_features(data, fs)
        batch_features.append(features)
        batch_labels.append(label[0])  # Only use the first element of the label
    return batch_features, batch_labels

# Funktion zur Extraktion von Features aus allen Daten in Batches
def extract_all_features(data_folder, batch_size):
    dataset = EEGDataset(data_folder)
    dataset_size = len(dataset)
    print(f"Gesamtzahl der Dateien im Datensatz: {dataset_size}")

    all_features = []
    all_labels = []
    num_batches = (dataset_size + batch_size - 1) // batch_size

    results = Parallel(n_jobs=-1, verbose=10)(delayed(process_batch)(dataset, start_idx, min(start_idx + batch_size, dataset_size))
                                             for start_idx in range(0, dataset_size, batch_size))

    for batch_idx, (batch_features, batch_labels) in enumerate(results):
        max_feature_length = max(len(f) for f in batch_features)
        corrected_batch_features = [
            np.pad(f, (0, max_feature_length - len(f)), 'constant') if len(f) < max_feature_length else f
            for f in batch_features
        ]
        all_features.append(np.array(corrected_batch_features))
        all_labels.extend(batch_labels)
        print(f"Batch {batch_idx + 1}/{num_batches} geladen: {len(batch_features)} Dateien")

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels).astype(int)
    print(f"Gesamtzahl der geladenen Dateien: {len(all_labels)}")

    if len(all_labels) != dataset_size:
        print(f"Warnung: Es wurden nicht alle Dateien geladen! Erwartet: {dataset_size}, Geladen: {len(all_labels)}")

    return all_features, all_labels

# Zusätzliche Überprüfung der geladenen IDs
def verify_loaded_files(data_folder):
    dataset = EEGDataset(data_folder)
    loaded_ids = [dataset[i][0] for i in range(len(dataset))]
    print(f"Anzahl der geladenen IDs: {len(loaded_ids)}")

    reference_file = os.path.join(data_folder, "REFERENCE.csv")
    reference_data = pd.read_csv(reference_file, header=None)
    reference_ids = reference_data[0].tolist()
    print(f"Anzahl der IDs in REFERENCE.csv: {len(reference_ids)}")

    missing_ids = set(reference_ids) - set(loaded_ids)
    extra_ids = set(loaded_ids) - set(reference_ids)

    if missing_ids:
        print(f"Fehlende IDs: {missing_ids}")
    if extra_ids:
        print(f"Zusätzliche IDs: {extra_ids}")

    all_files_correctly_loaded = len(missing_ids) == 0 and len(extra_ids) == 0
    print(f"Alle Dateien korrekt geladen: {all_files_correctly_loaded}")

    return all_files_correctly_loaded

# Optimierungsfunktion
def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    bootstrap = bool(bootstrap)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f1_scores = []

    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                     bootstrap=bootstrap, random_state=42, n_jobs=-1)

        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        f1 = f1_score(Y_test, Y_pred, average='weighted')  # Use weighted averaging for multiclass problems
        f1_scores.append(f1)

    return np.mean(f1_scores)

# Laden der Trainingsdaten
data_folder = '/home/jupyter-wki_team_2/Silvan/training/training'
batch_size = 1000  # Erhöhte Anzahl der Samples pro Batch

# Überprüfen, ob alle Dateien geladen werden
all_files_loaded = verify_loaded_files(data_folder)
if not all_files_loaded:
    print("Nicht alle Dateien wurden korrekt geladen!")
else:
    features, labels = extract_all_features(data_folder, batch_size)

    # Sicherstellen, dass die Labels als Integers für Multiclass vorliegen
    unique_labels = np.unique(labels)
    print(f"Einzigartige Labels: {unique_labels}")

    # Daten normalisieren
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)

    # Daten in Trainings- und Validierungsset aufteilen
    features_train, features_val, labels_train, labels_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Bayesian Optimization
    pbounds = {
        'n_estimators': (50, 200),
        'max_depth': (5, 50),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'bootstrap': (0, 1)
    }

    optimizer = BayesianOptimization(f=rf_cv, pbounds=pbounds, random_state=42, verbose=2)
    optimizer.maximize(init_points=10, n_iter=30)

    # Beste Parameter extrahieren
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
    best_params['bootstrap'] = bool(best_params['bootstrap'])

    # Trainieren des finalen Modells mit den besten Parametern
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(features_train, labels_train)

    # Speichern des Modells und des Scalers
    model_path = '/home/jupyter-wki_team_2/Silvan/test_rf/random_forest_model.joblib'
    scaler_path = '/home/jupyter-wki_team_2/Silvan/test_rf/scaler.joblib'
    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Modell und Scaler gespeichert.")
