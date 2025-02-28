import os
import streamlit as st
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
import librosa
import time


# -----------------------------
# Placeholder for Real Drum Extraction (6 Stems)
# -----------------------------
def extract_real_drum_midi(audio_path):
    """
    Extract the real drum track from the original audio by separating it into 6 stems
    (e.g., Kick, Snare, HiHat, Ride, Crash, Tom) and converting them to MIDI.
    This is a placeholder adapted from your Drummer_Style_App extraction process.
    In your production system, replace this with the actual inference and conversion code.
    """
    with st.spinner("Extracting real drum stems (6 stems) and converting to MIDI..."):
        # Simulate processing delay
        time.sleep(3)
        pm = pretty_midi.PrettyMIDI()
        drum = pretty_midi.Instrument(program=0, is_drum=True)
        # Dummy assignments for six stems:
        # Kick: 36, Snare: 38, HiHat: 42, Ride: 51, Crash: 49, Tom: 45
        dummy_pitches = {"Kick": 36, "Snare": 38, "HiHat": 42, "Ride": 51, "Crash": 49, "Tom": 45}
        # For demonstration, create 10 notes per instrument, spaced one second apart.
        for instr_name, pitch in dummy_pitches.items():
            for i in range(10):
                start = i * 1.0
                note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.1)
                drum.notes.append(note)
        pm.instruments.append(drum)
        return pm


# -----------------------------
# Functions for MIDI Drum Note Extraction & Grouping
# -----------------------------
def extract_drum_notes(midi_file_path):
    """
    Load a MIDI file and extract drum notes from drum instruments.
    Returns a list of dictionaries with keys: start, end, pitch, velocity.
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_file_path)
    except Exception as e:
        st.error(f"Error loading MIDI file: {e}")
        return []
    drum_notes = []
    for instr in pm.instruments:
        if instr.is_drum:
            for note in instr.notes:
                drum_notes.append({
                    "start": note.start,
                    "end": note.end,
                    "pitch": note.pitch,
                    "velocity": note.velocity
                })
    return drum_notes


def group_drum_notes_by_instrument(notes):
    """
    Group drum notes into categories:
      - Kick: MIDI notes 35 and 36.
      - Snare: MIDI notes 37-40.
      - HiHat: MIDI notes 42, 44, 46.
      - Cymbals: MIDI notes 49, 51, 53, 57, 59.
      Others are grouped into "Other".
    Returns a dictionary mapping instrument names to lists of note dictionaries.
    """
    groups = {"Kick": [], "Snare": [], "HiHat": [], "Cymbals": [], "Other": []}
    for note in notes:
        pitch = note["pitch"]
        if pitch in [35, 36]:
            groups["Kick"].append(note)
        elif pitch in [37, 38, 39, 40]:
            groups["Snare"].append(note)
        elif pitch in [42, 44, 46]:
            groups["HiHat"].append(note)
        elif pitch in [49, 51, 53, 57, 59]:
            groups["Cymbals"].append(note)
        else:
            groups["Other"].append(note)
    return groups


def plot_drum_comparison_by_instrument(ref_notes, gen_notes):
    """
    Create a 2x2 grid plot comparing the reference (red) and generated (blue) drum note onsets
    for Kick, Snare, HiHat, and Cymbals.
    """
    ref_groups = group_drum_notes_by_instrument(ref_notes)
    gen_groups = group_drum_notes_by_instrument(gen_notes)
    instruments = ["Kick", "Snare", "HiHat", "Cymbals"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, instr in enumerate(instruments):
        ax = axes[i]
        if ref_groups.get(instr):
            ref_times = [n["start"] for n in ref_groups[instr]]
            ref_pitches = [n["pitch"] for n in ref_groups[instr]]
            ax.scatter(ref_times, ref_pitches, color="red", label="Reference", alpha=0.7)
        if gen_groups.get(instr):
            gen_times = [n["start"] for n in gen_groups[instr]]
            gen_pitches = [n["pitch"] for n in gen_groups[instr]]
            ax.scatter(gen_times, gen_pitches, color="blue", label="Generated", alpha=0.7)
        ax.set_title(instr)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI Pitch")
        ax.legend()
    plt.tight_layout()
    return fig


# -----------------------------
# Main UI with Tabs
# -----------------------------
def main():
    st.title("Drum Track & Groove Analysis Comparison")
    tabs = st.tabs(["Extract Real Drums", "Compare Drum Tracks"])

    # Tab 1: Extraction of Real Drums
    with tabs[0]:
        st.header("Extract Real Drum MIDI from Original Audio")
        st.write(
            "Upload an original audio file (e.g., the YouTube download) to extract the real drum track using 6-stem separation.")
        orig_audio = st.file_uploader("Upload Original Audio (wav/mp3)", type=["wav", "mp3"], key="orig_audio")
        if orig_audio:
            orig_temp = "original_temp.wav"
            with open(orig_temp, "wb") as f:
                f.write(orig_audio.getbuffer())
            st.success("Original audio uploaded.")
            with st.spinner("Extracting real drum MIDI (6 stems)..."):
                ref_drum_pm = extract_real_drum_midi(orig_temp)
            ref_midi_path = "extracted_real_drum.mid"
            ref_drum_pm.write(ref_midi_path)
            st.success("Real drum MIDI extracted.")
            st.download_button("Download Extracted Real Drum MIDI", data=open(ref_midi_path, "rb").read(),
                               file_name=ref_midi_path)
            try:
                os.remove(orig_temp)
            except Exception as e:
                st.warning(f"Could not remove temporary file: {e}")

    # Tab 2: Comparison of Drum Tracks
    with tabs[1]:
        st.header("Compare Drum Tracks by Instrument")
        st.write(
            "Upload the reference drum MIDI file (extracted from the original) and the generated drum MIDI file to compare their timing and articulation.")
        ref_midi_file = st.file_uploader("Upload Reference Drum MIDI", type=["mid", "midi"], key="ref_midi_compare")
        gen_midi_file = st.file_uploader("Upload Generated Drum MIDI", type=["mid", "midi"], key="gen_midi_compare")
        if ref_midi_file and gen_midi_file:
            ref_temp = "ref_temp.mid"
            gen_temp = "gen_temp.mid"
            with open(ref_temp, "wb") as f:
                f.write(ref_midi_file.getbuffer())
            with open(gen_temp, "wb") as f:
                f.write(gen_midi_file.getbuffer())
            ref_notes = extract_drum_notes(ref_temp)
            gen_notes = extract_drum_notes(gen_temp)
            if not ref_notes:
                st.error("No drum notes extracted from the reference file.")
            elif not gen_notes:
                st.error("No drum notes extracted from the generated file.")
            else:
                fig = plot_drum_comparison_by_instrument(ref_notes, gen_notes)
                st.pyplot(fig)
            try:
                os.remove(ref_temp)
                os.remove(gen_temp)
            except Exception as e:
                st.warning(f"Could not remove temporary files: {e}")
        else:
            st.write("Please upload both reference and generated drum MIDI files.")


if __name__ == "__main__":
    main()
