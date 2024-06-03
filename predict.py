import os
import numpy as np
import joblib
from typing import List, Dict, Any
import argparse
from wettbewerb import EEGDataset

# Laden des Modells und des Scalers
model_path = '/home/jupyter-wki_team_2/Silvan/test_rf/random_forest_model.joblib'
scaler_path = '/home/jupyter-wki_team_2/Silvan/test_rf/scaler.joblib'
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Funktion zum Extrahieren von Features aus rohen EEG-Daten
def extract_features(data, fs):
    features = []
    for channel_data in data:
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        line_length = np.sum(np.abs(np.diff(channel_data)))
        features.extend([mean, std, line_length])
    max_feature_length = 57  # Ensure the same number of features as during training (19 channels * 3 features)
    if len(features) < max_feature_length:
        features = np.pad(features, (0, max_feature_length - len(features)), 'constant')
    return np.array(features)

def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str, model_name: str='model.json') -> Dict[str, Any]:
    try:
        # Extrahieren der Merkmale aus den Daten
        features = extract_features(data, fs)
        features = scaler.transform([features])
        
        # Vorhersage mit dem geladenen Modell
        seizure_present = model.predict(features)[0]
        seizure_confidence = model.predict_proba(features)[0, seizure_present]
        
        # Dummy-Onset und -Offset (müssen durch echte Vorhersagen ersetzt werden)
        onset = 0.0
        onset_confidence = 1.0
        offset = data.shape[1] / fs
        offset_confidence = 1.0
        
        prediction = {
            "seizure_present": seizure_present,
            "seizure_confidence": seizure_confidence,
            "onset": onset,
            "onset_confidence": onset_confidence,
            "offset": offset,
            "offset_confidence": offset_confidence
        }
        
        return prediction
    except Exception as e:
        print(f"Fehler bei der Verarbeitung der Datei: {e}")
        return {"seizure_present": 0, "seizure_confidence": 0.0, "onset": 0.0, "onset_confidence": 0.0, "offset": 0.0, "offset_confidence": 0.0}

def main(test_dir: str):
    # Laden des Testdatensatzes
    dataset = EEGDataset(test_dir)
    predictions = []
    
    for i in range(len(dataset)):
        id, channels, data, fs, ref_system, label = dataset[i]
        prediction = predict_labels(channels, data, fs, ref_system)
        prediction['id'] = id
        predictions.append(prediction)
    
    # Speichern der Vorhersagen in einer CSV-Datei
    prediction_file = os.path.join(test_dir, "predictions.csv")
    with open(prediction_file, 'w') as f:
        f.write("id,seizure_present,seizure_confidence,onset,onset_confidence,offset,offset_confidence\n")
        for pred in predictions:
            f.write(f"{pred['id']},{pred['seizure_present']},{pred['seizure_confidence']},{pred['onset']},{pred['onset_confidence']},{pred['offset']},{pred['offset_confidence']}\n")
    
    print(f"Vorhersagen wurden in {prediction_file} gespeichert.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on EEG test data")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing the test EEG data")
    args = parser.parse_args()
    
    main(args.test_dir)
