# preprocess_egmd.py

import os
import json


def build_midi_database(midi_root):
    """
    Recursively scan the MIDI root directory for MIDI files following the naming convention:
      <id>_<style>_<bpm>_beat_<time_signature>_<version>.midi
    Returns a dictionary mapping the base filename (without extension) to metadata.
    """
    midi_db = {}
    for dirpath, _, filenames in os.walk(midi_root):
        for file in filenames:
            if file.lower().endswith((".midi", ".mid")):
                base = os.path.splitext(file)[0]
                parts = base.split('_')
                if len(parts) >= 5:
                    style = parts[1]  # e.g., "funk-groove1"
                    try:
                        bpm = int(parts[2])
                    except ValueError:
                        bpm = None
                    time_signature = parts[4]  # e.g., "4-4"
                    midi_db[base] = {
                        "midi_path": os.path.join(dirpath, file),
                        "style": style,
                        "bpm": bpm,
                        "time_signature": time_signature
                    }
                else:
                    print(f"Warning: Filename does not match expected pattern: {file}")
    return midi_db


def build_audio_database(audio_root):
    """
    Recursively scan the audio root directory for audio files (wav, mp3).
    Returns a dictionary mapping the base filename (without extension) to its full file path.
    """
    audio_db = {}
    for dirpath, _, filenames in os.walk(audio_root):
        for file in filenames:
            if file.lower().endswith(('.wav', '.mp3')):
                base = os.path.splitext(file)[0]
                audio_db[base] = os.path.join(dirpath, file)
    return audio_db


def build_combined_database(midi_root, audio_root):
    """
    Build a combined database by matching MIDI and audio files based on the base filename.
    Returns a dictionary where each key is the base filename and the value is a dictionary
    with keys "midi" and "audio" for the corresponding file paths and metadata.
    """
    midi_db = build_midi_database(midi_root)
    audio_db = build_audio_database(audio_root)
    combined_db = {}
    for base, midi_info in midi_db.items():
        if base in audio_db:
            combined_db[base] = {
                "midi": midi_info,
                "audio": audio_db[base]
            }
    return combined_db


if __name__ == "__main__":
    # Change these paths as needed.
    midi_root = r"G:\E-GMD Dataset\e-gmd-v1.0.0-midi\e-gmd-v1.0.0"
    audio_root = r"G:\E-GMD Dataset\e-gmd-v1.0.0\e-gmd-v1.0.0"

    print("Scanning MIDI files...")
    midi_db = build_midi_database(midi_root)
    print(f"Found {len(midi_db)} MIDI entries.")

    print("Scanning audio files...")
    audio_db = build_audio_database(audio_root)
    print(f"Found {len(audio_db)} audio entries.")

    combined_db = build_combined_database(midi_root, audio_root)
    print(f"Combined database has {len(combined_db)} matched entries.")

    output_file = "egmd_combined_db.json"
    with open(output_file, "w") as f:
        json.dump(combined_db, f, indent=4)

    print(f"Preprocessing complete. Database saved to {output_file}")
