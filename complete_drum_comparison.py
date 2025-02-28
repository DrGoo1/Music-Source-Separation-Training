import numpy as np

if not hasattr(np, 'int'):
    np.int = int

import os
import time
import re
import subprocess
import streamlit as st
import yt_dlp as youtube_dl
import pretty_midi
import matplotlib.pyplot as plt
import librosa

# Set global directories (using your DrumTracksAI_Data folder)
DATA_FOLDER = r"C:\Users\goldw\PycharmProjects\DrumTracksAI_Data"
AUDIO_DOWNLOAD_DIR = os.path.join(DATA_FOLDER, "downloaded_audio")
SEPARATED_RESULTS_DIR = os.path.join(DATA_FOLDER, "separation_results")
os.makedirs(AUDIO_DOWNLOAD_DIR, exist_ok=True)
os.makedirs(SEPARATED_RESULTS_DIR, exist_ok=True)


# -----------------------------
# YouTube Download Function
# -----------------------------
def download_youtube_audio(url, output_template="%(title)s.%(ext)s"):
    st.info(f"Downloading audio from YouTube URL: {url}")
    abs_output_template = os.path.join(os.getcwd(), output_template)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": abs_output_template,
        "download_sections": [{"section": "*0-180"}],  # First 3 minutes
        "-x": True,
        "--audio-format": "wav",
        "--audio-quality": "192",
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        st.error(f"Error downloading YouTube audio: {e}")
        return None
    # Find the downloaded .wav file in the current directory
    for file in os.listdir(os.getcwd()):
        if file.endswith(".wav"):
            downloaded_path = os.path.join(os.getcwd(), file)
            st.success(f"Audio downloaded to {downloaded_path}")
            return downloaded_path
    st.error("Downloaded file not found.")
    return None


# -----------------------------
# Groove Analysis Function
# -----------------------------
def analyze_guide_track(audio_path, trim_silence=True, top_db=30, hop_length=256):
    """
    Analyze the guide track using librosa to extract tempo, beat times, and a groove metric.
    Returns a dictionary with keys: tempo, beat_times, and groove_metric.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None
    st.write(f"Audio loaded: Duration = {len(y) / sr:.2f} sec, Sample rate = {sr} Hz")
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=top_db)
        st.write(f"After trimming silence (top_db={top_db}): Duration = {len(y) / sr:.2f} sec")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    tempo = float(tempo)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    st.write(f"Detected tempo: {tempo:.2f} BPM, Beats: {len(beats)}")
    intervals = np.diff(beat_times)
    groove_metric = np.std(intervals) if len(intervals) > 0 else 0.0
    return {"tempo": tempo, "beat_times": beat_times.tolist(), "groove_metric": float(groove_metric)}


# -----------------------------
# Run Drum Separation Inference with Status Bar & Stem Selection
# -----------------------------
def run_drum_sep_inference_with_progress(input_file):
    st.info("Running drum separation inference for Drums...")
    input_basename = os.path.splitext(os.path.basename(input_file))[0]

    # Copy input file to AUDIO_DOWNLOAD_DIR if not already there.
    dest_path = os.path.join(AUDIO_DOWNLOAD_DIR, os.path.basename(input_file))
    if not os.path.exists(dest_path):
        try:
            subprocess.check_call(f'copy "{input_file}" "{dest_path}"', shell=True)
        except Exception as e:
            st.error(f"Error copying file: {e}")
            return None

    # Set paths for config, checkpoint, and inference script.
    config_path = os.path.join("C:\\Users\\goldw\\PycharmProjects\\DrumTracksAI\\Music-Source-Separation-Training",
                               "configs", "config_drumsep_mdx23c.yaml")
    checkpoint_path = os.path.join("C:\\Users\\goldw\\PycharmProjects\\DrumTracksAI\\Music-Source-Separation-Training",
                                   "pretrained", "drumsep_mdx23c.ckpt")
    inference_script = os.path.join("C:\\Users\\goldw\\PycharmProjects\\DrumTracksAI\\Music-Source-Separation-Training",
                                    "inference.py")

    command = [
        "python",
        "-u",
        inference_script,
        "--model_type", "mdx23c",
        "--config_path", config_path,
        "--start_check_point", checkpoint_path,
        "--input_folder", AUDIO_DOWNLOAD_DIR,
        "--store_dir", SEPARATED_RESULTS_DIR,
        "--device_ids", "0"
    ]

    progress_bar = st.progress(0)
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        progress_regex = r"Processing (?:audio )?chunks?:\s*(\d+)%"
        for line in iter(process.stdout.readline, ""):
            # Only update the progress bar (do not print each line)
            match = re.search(progress_regex, line)
            if match:
                try:
                    percent = int(match.group(1))
                    progress_bar.progress(percent)
                except Exception:
                    pass
        process.stdout.close()
        process.wait()
        if process.returncode == 0:
            progress_bar.progress(100)
            st.success("Drum separation completed successfully!")
        else:
            st.error(f"Drum separation failed with return code: {process.returncode}")
    except Exception as e:
        st.error(f"Error during drum separation: {e}")
        return None

    # Expected output folder: SEPARATED_RESULTS_DIR\<input_basename>
    separated_folder = os.path.join(SEPARATED_RESULTS_DIR, input_basename)
    # Originally expected file "drums.wav"; if not found, list available files.
    expected_path = os.path.join(separated_folder, "drums.wav")
    if os.path.exists(expected_path):
        return expected_path
    else:
        st.error("Separated drum track not found. Check the inference output.")
        if os.path.exists(separated_folder):
            files = os.listdir(separated_folder)
            st.write("Files in the expected output folder:", files)
            chosen = st.selectbox("Select the separated drum stem to use as the guide track:", files)
            chosen_path = os.path.join(separated_folder, chosen)
            return chosen_path
        return None


# -----------------------------
# Guide Track Extraction (Updated)
# -----------------------------
def extract_guide_track(input_file_path, source_type, target_instrument="Drums"):
    st.info(f"Extracting guide track for {target_instrument} from {source_type} input...")
    if target_instrument.lower() == "drums":
        separated_path = run_drum_sep_inference_with_progress(input_file_path)
        return separated_path
    else:
        return input_file_path


# -----------------------------
# Drum Note Extraction & Grouping (Comparison)
# -----------------------------
def extract_drum_notes(midi_file_path):
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
        ax.set_ylabel("MIDI Note")
        ax.legend()
    plt.tight_layout()
    return fig


# -----------------------------
# Main UI with Tabs
# -----------------------------
def main():
    st.title("Complete Drum Extraction & Comparison Module")
    tabs = st.tabs(["Extract Real Drums", "Compare Drum Tracks"])

    # Tab 1: Extraction of Real Drums
    with tabs[0]:
        st.header("Extract Real Drum MIDI")
        st.write("""
            Upload an original audio file (wav/mp3) or enter a YouTube URL.
            The system will extract the drum track using a 6-stem separation process from your Music-Source-Separation-Training folder
            and perform groove analysis.
        """)
        input_choice = st.radio("Choose Input Type:", options=["Upload Audio File", "YouTube URL"])
        input_path = None
        if input_choice == "Upload Audio File":
            uploaded_audio = st.file_uploader("Upload Original Audio", type=["wav", "mp3"], key="orig_audio")
            if uploaded_audio:
                input_path = "original_audio.wav"
                with open(input_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                st.success("Original audio uploaded.")
        else:
            youtube_url = st.text_input("Enter YouTube URL:")
            if youtube_url:
                input_path = download_youtube_audio(youtube_url)

        if input_path:
            guide_track = extract_guide_track(input_path, source_type=input_choice, target_instrument="Drums")
            st.write("Guide track path:", guide_track)
            if guide_track:
                with st.spinner("Performing groove analysis on guide track..."):
                    groove_data = analyze_guide_track(guide_track)
                if groove_data:
                    st.write("Groove Analysis Result:")
                    st.json(groove_data)
                else:
                    st.error("Groove analysis failed.")
            if st.button("Extract Real Drum MIDI"):
                if guide_track:
                    with st.spinner("Extracting real drum MIDI..."):
                        ref_drum_pm = extract_real_drum_midi(guide_track)
                        ref_midi_path = "extracted_real_drum.mid"
                        ref_drum_pm.write(ref_midi_path)
                    st.success("Real drum MIDI extracted.")
                    st.download_button("Download Extracted Real Drum MIDI", data=open(ref_midi_path, "rb").read(),
                                       file_name=ref_midi_path)
                else:
                    st.error("Guide track not available for extraction.")

    # Tab 2: Comparison of Drum Tracks
    with tabs[1]:
        st.header("Compare Drum Tracks by Instrument")
        st.write(
            "Upload a reference drum MIDI file and a generated drum MIDI file to compare their timing and articulation.")
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
