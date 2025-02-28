#!/usr/bin/env python
import os
import librosa
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

# Set environment variables before any other imports
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
os.environ["DISABLE_INTEL_SVML"] = "1"


def convert_separated_audio_to_midi(audio_path, output_midi_path, instrument):
    """
    Convert a separated instrument audio file (e.g., kick.wav, snare.wav, hh.wav, etc.)
    to a MIDI file by detecting onsets using tuned parameters and assigning
    General MIDI drum note numbers:
      - Kick: 36
      - Snare: 38
      - Tom(s): 45
      - Hi-hat: 42
      - Ride: 51
      - Crash: 49
    This function plots the onset envelope for debugging.
    """
    # Load the audio file
    y, sr = librosa.load(audio_path)
    print(f"[{instrument}] Loaded audio from {audio_path} with shape {y.shape} and duration {len(y) / sr:.2f}s")

    # Compute the onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = np.arange(len(onset_env)) / sr

    # Plot the onset envelope for visual inspection
    plt.figure(figsize=(10, 4))
    plt.plot(times, onset_env, label="Onset Strength")
    plt.xlabel("Time (s)")
    plt.ylabel("Onset Strength")
    plt.title(f"Onset Envelope for {instrument}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Tuned onset detection parameters
    onset_kwargs = {
        "sr": sr,
        "backtrack": True,
        "pre_max": 20,
        "post_max": 20,
        "pre_avg": 100,
        "post_avg": 100,
        "delta": 0.3,
        "wait": 10
    }
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, **onset_kwargs)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"[{instrument}] Detected onset frames: {onset_frames}")
    print(f"[{instrument}] Detected onset times: {onset_times}")
    print(f"[{instrument}] Number of detected onsets: {len(onset_times)}")

    pm = pretty_midi.PrettyMIDI()
    drum_instr = pretty_midi.Instrument(program=0, is_drum=True)

    # Map instrument name to General MIDI note
    inst_lower = instrument.lower()
    if inst_lower == "kick":
        note_pitch = 36
    elif inst_lower == "snare":
        note_pitch = 38
    elif inst_lower in ["tom", "toms"]:
        note_pitch = 45
    elif inst_lower in ["hihat"]:
        note_pitch = 42
    elif inst_lower == "ride":
        note_pitch = 51
    elif inst_lower == "crash":
        note_pitch = 49
    else:
        note_pitch = 38  # Fallback to snare

    if len(onset_times) == 0:
        print(f"[{instrument}] Warning: No onsets detected in {audio_path}!")

    for onset in onset_times:
        print(f"[{instrument}] Processing onset at: {onset:.3f}")
        start_sample = int(onset * sr)
        window = y[start_sample:start_sample + int(0.05 * sr)]
        amplitude = np.max(np.abs(window)) if len(window) > 0 else 0.1
        velocity = int(np.clip(amplitude * 127 / np.max(np.abs(y)), 20, 127))
        note = pretty_midi.Note(velocity=velocity, pitch=note_pitch, start=onset, end=onset + 0.1)
        drum_instr.notes.append(note)

    print(f"[{instrument}] Total MIDI notes added: {len(drum_instr.notes)}")
    pm.instruments.append(drum_instr)
    try:
        pm.write(output_midi_path)
        print(f"[{instrument}] MIDI file written to: {output_midi_path}")
    except Exception as e:
        print(f"[{instrument}] Error writing MIDI file: {e}")


def find_separated_instrument_files(root_dir, instrument):
    """
    Recursively search for .wav files in root_dir that match the given instrument.
    For hi-hat, search for both "hihat" and "hh".
    Returns a list of absolute file paths.
    """
    found_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            f_lower = f.lower()
            if f_lower.endswith(".wav"):
                if instrument.lower() == "hihat":
                    if "hihat" in f_lower or "hh" in f_lower:
                        found_files.append(os.path.join(dirpath, f))
                else:
                    if instrument.lower() in f_lower:
                        found_files.append(os.path.join(dirpath, f))
    return found_files


def convert_all_separated_to_midi(separation_dir, output_dir):
    """
    Iterate through the separation results directory, find separated instrument audio files,
    and convert each to a MIDI file using General MIDI drum note assignments.
    The MIDI files are saved in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define the instruments we expect:
    instruments = ["kick", "snare", "tom", "hihat", "ride", "crash"]

    for inst in instruments:
        print(f"Searching for files for instrument: {inst}")
        inst_files = find_separated_instrument_files(separation_dir, inst)
        if not inst_files:
            print(f"No files found for instrument: {inst}")
        else:
            for file_path in inst_files:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                midi_filename = f"{base_name}_{inst}.mid"
                output_midi_path = os.path.join(output_dir, midi_filename)
                print(f"Converting {file_path} to MIDI as {midi_filename}")
                convert_separated_audio_to_midi(file_path, output_midi_path, instrument=inst)


if __name__ == "__main__":
    # Set the path to your separation results folder.
    separation_results_dir = r"C:\Users\goldw\PycharmProjects\DrumTracksAI_Data\separation_results"

    # Set the output directory for MIDI files.
    midi_output_dir = os.path.join(os.getcwd(), "separated_MIDI_converted")
    os.makedirs(midi_output_dir, exist_ok=True)

    # Run conversion on all separated instrument files.
    convert_all_separated_to_midi(separation_results_dir, midi_output_dir)

