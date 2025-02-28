import os
import streamlit as st
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
import librosa


# -----------------------------
# Functions for MIDI Comparison
# -----------------------------
def extract_drum_notes(midi_file_path):
    """Load a MIDI file and extract drum notes (all notes from drum instruments)."""
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
    Group drum notes by instrument category:
      - Kick: MIDI 35,36
      - Snare: MIDI 37-40
      - HiHat: MIDI 42,44,46
      - Cymbals: MIDI 49,51,53,57,59
      Other notes are placed in "Other".
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
    Create a 2x2 grid plot comparing reference (red) and generated (blue) drum note onsets,
    separately for Kick, Snare, and HiHat, Cymbals.
    """
    ref_groups = group_drum_notes_by_instrument(ref_notes)
    gen_groups = group_drum_notes_by_instrument(gen_notes)
    instruments = ["Kick", "Snare", "HiHat", "Cymbals"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, instr in enumerate(instruments):
        ax = axes[i]
        # Plot reference notes in red.
        if ref_groups.get(instr):
            ref_times = [n["start"] for n in ref_groups[instr]]
            ref_pitches = [n["pitch"] for n in ref_groups[instr]]
            ax.scatter(ref_times, ref_pitches, color="red", label="Reference", alpha=0.7)
        # Plot generated notes in blue.
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
# Functions for Groove Extraction Testing
# -----------------------------
def analyze_guide_track(audio_path, trim_silence=True, top_db=30, hop_length=256):
    """
    Load an audio file and perform beat tracking using librosa.
    Returns a dictionary with tempo, beat_times, and groove_metric.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=top_db)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(tempo)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    intervals = np.diff(beat_times)
    groove_metric = np.std(intervals) if len(intervals) > 0 else 0.0
    return {"tempo": tempo, "beat_times": beat_times.tolist(), "groove_metric": float(groove_metric)}


def plot_beat_times(beat_times):
    """
    Plot vertical lines at each beat time to visualize the groove extraction.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    for bt in beat_times:
        ax.axvline(x=bt, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_title("Extracted Beat Times from Guide Track")
    return fig


# -----------------------------
# Main UI with Tabs
# -----------------------------
def main():
    st.title("Drum Track and Groove Analysis Tester")

    tab1, tab2 = st.tabs(["MIDI Comparison", "Groove Extraction Test"])

    with tab1:
        st.header("MIDI Comparison by Instrument")
        st.write("""
            Upload a reference drum MIDI file (from the real performance) and a generated drum MIDI file.
            This will display separate plots for Kick, Snare, HiHat, and Cymbals.
        """)
        ref_midi_file = st.file_uploader("Upload Reference Drum MIDI", type=["mid", "midi"], key="ref_midi")
        gen_midi_file = st.file_uploader("Upload Generated Drum MIDI", type=["mid", "midi"], key="gen_midi")
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

    with tab2:
        st.header("Groove Extraction Test")
        st.write("""
            Upload a guide track audio file (e.g., an isolated bass track) to test the groove extraction process.
            The system will perform beat tracking and display the extracted beat times along with the tempo and groove metric.
        """)
        guide_audio_file = st.file_uploader("Upload Guide Track Audio (wav or mp3)", type=["wav", "mp3"],
                                            key="guide_audio")
        if guide_audio_file:
            guide_temp = "guide_temp.wav"
            with open(guide_temp, "wb") as f:
                f.write(guide_audio_file.getbuffer())
            with st.spinner("Extracting groove from guide track..."):
                groove_result = analyze_guide_track(guide_temp)
            if groove_result:
                st.write("Groove Extraction Result:")
                st.json(groove_result)
                beat_times = groove_result.get("beat_times", [])
                if beat_times:
                    fig2 = plot_beat_times(beat_times)
                    st.pyplot(fig2)
                else:
                    st.error("No beat times extracted.")
            try:
                os.remove(guide_temp)
            except Exception as e:
                st.warning(f"Could not remove temporary guide file: {e}")
        else:
            st.write("Please upload a guide track audio file.")


if __name__ == "__main__":
    main()
