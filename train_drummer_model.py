import pretty_midi
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Path to training MIDI files
TRAINING_MIDI_DIR = "drummer_midi_samples"

# Check if the directory exists
if not os.path.exists(TRAINING_MIDI_DIR):
    print(f"Error: The directory {TRAINING_MIDI_DIR} does not exist.")
    exit()

data = []
labels = []

def extract_midi_features(midi_file):
    """Extract per-instrument velocity & timing from MIDI drum track."""
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    drum_features = {
        "kick": {"velocity": [], "timing": []},
        "snare": {"velocity": [], "timing": []},
        "hi_hat": {"velocity": [], "timing": []},
        "ride": {"velocity": [], "timing": []},
        "crash": {"velocity": [], "timing": []},
        "toms": {"velocity": [], "timing": []},
    }

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                pitch = note.pitch
                if pitch == 36:
                    drum_features["kick"]["velocity"].append(note.velocity)
                    drum_features["kick"]["timing"].append(note.start)
                elif pitch == 38:
                    drum_features["snare"]["velocity"].append(note.velocity)
                    drum_features["snare"]["timing"].append(note.start)
                elif pitch in [42, 44]:
                    drum_features["hi_hat"]["velocity"].append(note.velocity)
                    drum_features["hi_hat"]["timing"].append(note.start)
                elif pitch == 51:
                    drum_features["ride"]["velocity"].append(note.velocity)
                    drum_features["ride"]["timing"].append(note.start)
                elif pitch in [49, 57]:
                    drum_features["crash"]["velocity"].append(note.velocity)
                    drum_features["crash"]["timing"].append(note.start)
                elif pitch in [43, 47, 50]:
                    drum_features["toms"]["velocity"].append(note.velocity)
                    drum_features["toms"]["timing"].append(note.start)

    features = []
    for instrument in drum_features:
        avg_velocity = np.mean(drum_features[instrument]["velocity"]) if drum_features[instrument]["velocity"] else 0
        timing_variation = np.std(drum_features[instrument]["timing"]) if drum_features[instrument]["timing"] else 0
        features.extend([avg_velocity, timing_variation])

    return features

# Process training MIDI files
for file in os.listdir(TRAINING_MIDI_DIR):
    if file.endswith(".mid") or file.endswith(".midi"):
        filepath = os.path.join(TRAINING_MIDI_DIR, file)
        features = extract_midi_features(filepath)
        drummer_label = file.split("_")[0]  # Assumes filenames like "NeilPeart_001.mid"
        data.append(features)
        labels.append(drummer_label)

if len(data) == 0:
    print("No training data found! Please add MIDI files to the 'drummer_midi_samples' folder.")
    exit()

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Train the model
model = RandomForestClassifier(n_estimators=200)
model.fit(data, encoded_labels)

# Save the model and label encoder
joblib.dump(model, "drummer_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Drummer classification model trained and saved successfully!")
