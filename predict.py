# predict2.py
# -*- coding: utf-8 -*-
"""
Skript testet das vortrainierte Modell

@author: Silvan Steinhauer
"""

import numpy as np
import os
from typing import List, Dict, Any
import mne
from scipy import signal as sig
import pywt
from joblib import load, Parallel, delayed
from tqdm import tqdm
import json

# Funktion zur Berechnung von Features unter Verwendung von TWD
def calculate_features(signal, fs):
    features = {}

    # Preprocessing
    signal = signal - np.mean(signal)  # Baseline correction
    signal = mne.filter.notch_filter(signal, fs, freqs=[50, 100], verbose=False)  # Notch filter

    # Feste Filterlänge und Übergangsbandbreite
    filter_length = '10s'  # 10 Sekunden
    transition_bandwidth = 0.5  # 0.5 Hz Übergangsbandbreite
    signal = mne.filter.filter_data(signal, fs, 0.5, 40.0, verbose=False, filter_length=filter_length, l_trans_bandwidth=transition_bandwidth, h_trans_bandwidth=transition_bandwidth)  # Bandpass filter
    signal = signal - np.median(signal, axis=0)  # Re-reference

    # Überprüfung auf NaN-Werte und Inf-Werte
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return None

    # Triadic Wavelet Decomposition (TWD)
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    for i, coeff in enumerate(coeffs):
        features[f'coeff_mean_{i}'] = np.mean(coeff)
        features[f'coeff_std_{i}'] = np.std(coeff)
        features[f'coeff_energy_{i}'] = np.sum(coeff ** 2)

    # Zeitbereichs-Analyse
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['ptp'] = np.ptp(signal)  # Peak-to-peak
    features['line_length'] = np.sum(np.abs(np.diff(signal)))

    # Frequenzbereichs-Analyse
    freqs, psd = sig.welch(signal, fs, nperseg=1024)
    features['psd_mean'] = np.mean(psd)
    features['psd_std'] = np.std(psd)

    # Zeit-Frequenz-Analyse mittels STFT
    f, t, Zxx = sig.stft(signal, fs)
    features['stft_mean'] = np.mean(np.abs(Zxx))
    features['stft_std'] = np.std(np.abs(Zxx))

    # Statistische Analyse
    features['kurtosis'] = np.mean((signal - np.mean(signal))**4) / np.var(signal)**2
    features['skewness'] = np.mean((signal - np.mean(signal))**3) / np.var(signal)**1.5

    return features

def process_channel(signal, fs):
    return calculate_features(signal, fs)

### Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str, model_name: str='best_model.joblib') -> Dict[str, Any]:
    '''
    Parameters
    ----------
    channels : List[str]
        Namen der übergebenen Kanäle
    data : ndarray
        EEG-Signale der angegebenen Kanäle
    fs : float
        Sampling-Frequenz der Signale.
    reference_system :  str
        Welches Referenzsystem wurde benutzt, "Bezugselektrode", nicht garantiert korrekt!
    model_name : str
        Name eures Models, das ihr beispielsweise bei Abgabe genannt habt. 
        Kann verwendet werden, um korrektes Model aus Ordner zu laden.
    Returns
    -------
    prediction : Dict[str, Any]
        enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
    '''

    # Initialisiere Return (Ergebnisse)
    seizure_present = True  # Gibt an, ob ein Anfall vorliegt
    seizure_confidence = 0.5  # Gibt die Unsicherheit des Modells an (optional)
    onset = 4.2  # Gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99  # Gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999  # Gibt das Ende des Anfalls an (optional)
    offset_confidence = 0  # Gibt die Unsicherheit bezüglich des Endes an (optional)

    # Überprüfe den Modellnamen und setze ihn ggf. auf 'best_model.joblib'
    if model_name == 'model.json':
        model_name = 'best_model.joblib'

    # Lade das vortrainierte Modell
    model_path = os.path.join('/home/jupyter-wki_team_2/Silvan/test_2/', model_name)
    try:
        clf = load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return {}

    # Lade den Scaler und die PCA
    scaler_path = os.path.join('/home/jupyter-wki_team_2/Silvan/test_2/', 'scaler.pkl')
    pca_path = os.path.join('/home/jupyter-wki_team_2/Silvan/test_2/', 'pca.pkl')
    onset_model_path = os.path.join('/home/jupyter-wki_team_2/Silvan/test_2/', 'onset_model.joblib')
    offset_model_path = os.path.join('/home/jupyter-wki_team_2/Silvan/test_2/', 'offset_model.joblib')
    try:
        scaler = load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler from {scaler_path}: {e}")
        return {}

    try:
        pca = load(pca_path)
        print(f"PCA loaded from {pca_path}")
    except Exception as e:
        print(f"Error loading PCA from {pca_path}: {e}")
        return {}

    try:
        onset_model = load(onset_model_path)
        print(f"Onset model loaded from {onset_model_path}")
    except Exception as e:
        print(f"Error loading onset model from {onset_model_path}: {e}")
        return {}

    try:
        offset_model = load(offset_model_path)
        print(f"Offset model loaded from {offset_model_path}")
    except Exception as e:
        print(f"Error loading offset model from {offset_model_path}: {e}")
        return {}

    # Rohdaten verwenden
    signal_std = np.zeros(len(channels))
    
    # Parallelisiere die Feature-Berechnung
    features_list = Parallel(n_jobs=-1, verbose=10)(delayed(process_channel)(data[j], fs) for j in range(len(channels)))
    
    # Berechne Feature zur Seizure Detektion
    if len(features_list) == 0:
        print("No features extracted.")
        return {}

    X = np.array([list(f.values()) for f in features_list])
    if X.shape[0] == 0:
        print("No features to predict.")
        return {}

    # Standardisierung der Daten
    X = scaler.transform(X)

    # Anwendung von PCA
    X = pca.transform(X)

    # Wähle die Top-n Features aus
    top_n = 10
    with open('/home/jupyter-wki_team_2/Silvan/test_2/best_params.json', 'r') as f:
        best_params = json.load(f)
    importances = clf.named_estimators_['rf'].feature_importances_
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    X_top = X[:, top_indices]
    
    try:
        seizure_present = clf.predict(X_top)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {}

    try:
        onset = onset_model.predict(X_top)[0]
    except Exception as e:
        print(f"Error during onset prediction: {e}")
        return {}

    try:
        offset = offset_model.predict(X_top)[0]
    except Exception as e:
        print(f"Error during offset prediction: {e}")
        return {}

    prediction = {
        "seizure_present": seizure_present,
        "seizure_confidence": seizure_confidence,
        "onset": onset,
        "onset_confidence": onset_confidence,
        "offset": offset,
        "offset_confidence": offset_confidence
    }

    return prediction  # Dictionary mit prediction - Muss unverändert bleiben!
