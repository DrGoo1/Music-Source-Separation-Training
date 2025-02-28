# Drummer_Style_App.py
# ----------------------------------------------------------------
# Set environment variables early to disable Intel SVML
import os

os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
os.environ["DISABLE_INTEL_SVML"] = "1"

# Patch torch distributed ProcessGroup to work around the missing Options attribute.
import torch

if not hasattr(torch._C._distributed_c10d.ProcessGroup, 'Options'):
    torch._C._distributed_c10d.ProcessGroup.Options = None

print("Environment patch applied. Using GPU:", torch.cuda.is_available())

# ----------------------------------------------------------------
# Continue with the rest of your imports and app code
import streamlit as st
import requests
import pretty_midi
import numpy as np
import pandas as pd
import joblib
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import json
import random
import yt_dlp as youtube_dl  # Use yt_dlp instead of youtube_dl
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import io
import re
import shutil
import time
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Debug Message: App Load
# -----------------------------
st.write("Debug test: App loaded!")

# -----------------------------
# Configuration & API Tokens from Streamlit Secrets
# -----------------------------
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

# PayPal API Integration (placeholders; integration to be added)
PAYPAL_CLIENT_ID = st.secrets["PAYPAL_CLIENT_ID"]
PAYPAL_SECRET = st.secrets["PAYPAL_SECRET"]
PAYPAL_API_BASE = "https://api-m.paypal.com"  # Use sandbox for testing

# -----------------------------
# Data Folder and Paths Setup
# -----------------------------
# All our data output folders will reside in DrumTracksAI_Data.
DATA_FOLDER = r"C:\Users\goldw\PycharmProjects\DrumTracksAI_Data"
AUDIO_DOWNLOAD_DIR = os.path.join(DATA_FOLDER, "downloaded_audio")
SEPARATED_RESULTS_DIR = os.path.join(DATA_FOLDER, "separation_results")
MIDI_CONVERTED_DIR = os.path.join(DATA_FOLDER, "separated_MIDI_converted")

# Ensure these directories exist
for folder in [AUDIO_DOWNLOAD_DIR, SEPARATED_RESULTS_DIR, MIDI_CONVERTED_DIR]:
    os.makedirs(folder, exist_ok=True)

# Other folders (for uploads, training samples, etc.) remain in the project folder.
DRUM_SAMPLE_REPO = "drum_samples/"  # Admin-uploaded drum samples
USER_UPLOADS_DIR = "user_uploads/"  # Used for style transfer
TRAINING_MIDI_DIR = "drummer_midi_samples"  # Training MIDI samples (e.g., "John Bonham.mid")

# -----------------------------
# Predefined Famous Drummers Dictionary
# -----------------------------
FAMOUS_DRUMMERS = {
    "John Bonham": ["Led Zeppelin"],
    "Neil Peart": ["Rush"],
    "Buddy Rich": ["Various Ensembles"],
    "Stewart Copeland": ["The Police"],
    "Keith Moon": ["The Who"],
    "Chad Smith": ["Red Hot Chili Peppers"],
    "Ginger Baker": ["Cream"],
    "Lars Ulrich": ["Metallica"],
    "Carter Beauford": ["Dave Matthews Band"],
    "Steve Gadd": ["Steely Dan"],
    "Phil Collins": ["Genesis"],
    "Danny Carey": ["Tool"],
    "Mike Portnoy": ["Dream Theater"],
    "Terry Bozzio": ["Missing Persons"],
    "Jojo Mayer": ["The Digital Intervention"],
    "Questlove": ["The Roots"]
}

# -----------------------------
# Global Drummer Selection
# -----------------------------
drummer_name = st.selectbox("Select Drummer", list(FAMOUS_DRUMMERS.keys()))
st.write("Selected Drummer:", drummer_name)

# -----------------------------
# Helper Function: Clean Filename
# -----------------------------
def clean_filename(filename):
    while filename.count(".wav") > 1:
        filename = filename.replace(".wav", "", 1)
    if not filename.endswith(".wav"):
        filename += ".wav"
    return filename

# -----------------------------
# New Function: Compute Groove Metric
# -----------------------------
def compute_groove_metric(pm, resolution=0.05):
    """
    Compute a groove metric for a PrettyMIDI object.
    For each drum note, quantize the start time to the nearest multiple of 'resolution' (in seconds)
    and compute the absolute deviation. Return the average deviation.
    """
    deviations = []
    for inst in pm.instruments:
        if inst.is_drum:
            for note in inst.notes:
                quantized_time = round(note.start / resolution) * resolution
                deviations.append(abs(note.start - quantized_time))
    if deviations:
        return np.mean(deviations)
    else:
        return None

# -----------------------------
# New Function: Apply Groove to Existing MIDI
# -----------------------------
def apply_groove_to_midi(input_midi_path, output_midi_path, groove_metric):
    """
    Modify a fully quantized MIDI file by adding random timing offsets
    drawn from a normal distribution with a standard deviation equal to the groove_metric.
    This simulates the human "groove" of a drummer.
    """
    pm = pretty_midi.PrettyMIDI(input_midi_path)
    for inst in pm.instruments:
        if inst.is_drum:
            for note in inst.notes:
                offset = np.random.normal(loc=0, scale=groove_metric)
                note.start += offset
                note.end += offset
    pm.write(output_midi_path)

# -----------------------------
# Helper Functions: Web Searches & Downloads
# -----------------------------
def search_youtube_videos(query):
    st.info("Searching YouTube with query: " + query)
    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": 50,
        "videoDuration": "short",  # Only short videos (<4 minutes)
        "key": YOUTUBE_API_KEY
    }
    response = requests.get(search_url, params=params)
    st.info("YouTube API URL: " + response.url)
    st.info("YouTube API status: " + str(response.status_code))
    if response.status_code == 200:
        data = response.json()
        video_links = [{
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        } for item in data.get("items", [])]
        return video_links
    else:
        st.error("YouTube API call failed.")
        return []

def search_youtube_drum_tracks(drummer_name, bands=None):
    video_results = []
    if bands:
        for band in bands:
            query = f"{band} {drummer_name}"
            video_results.extend(search_youtube_videos(query))
    else:
        bands = FAMOUS_DRUMMERS.get(drummer_name, [])
        if not bands:
            video_results.extend(search_youtube_videos(drummer_name))
        else:
            for band in bands:
                query = f"{band} {drummer_name}"
                video_results.extend(search_youtube_videos(query))
    unique_results = {video['url']: video for video in video_results}
    return list(unique_results.values())

def download_youtube_audio(youtube_url, output_path):
    st.info("Downloading audio from: " + youtube_url)
    # Limit download to first 3 minutes using yt_dlp's download_sections
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "download_sections": [{"section": "00:00:00-00:03:00"}],
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        st.success("Audio downloaded to: " + output_path)
    except Exception as e:
        st.error("Error during download: " + str(e))

# -----------------------------
# Helper Function: Drum Separation with Progress Bar
# -----------------------------
def run_drum_sep_inference_with_progress(model_type, config_path, checkpoint_path, input_folder, store_dir):
    command = [
        "python",
        "-u",
        os.path.join("Music-Source-Separation-Training", "inference.py"),
        "--model_type", model_type,
        "--config_path", config_path,
        "--start_check_point", checkpoint_path,
        "--input_folder", input_folder,
        "--store_dir", store_dir,
        "--device_ids", "0"
    ]
    st.info("Starting drum separation on GPU. Command: " + " ".join(command))
    progress_bar = st.progress(0)
    env = os.environ.copy()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        progress_regex = r"Processing (?:audio )?chunks?:\s*(\d+)%"
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
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
            st.error("Drum separation failed with return code: " + str(process.returncode))
    except Exception as e:
        st.error("Error during drum separation: " + str(e))

# -----------------------------
# Helper Functions: MIDI Style Transfer & Analysis
# -----------------------------
def get_drum_type(note_pitch):
    drum_map = {
        "kick": [35, 36],
        "snare": [38, 40],
        "hihat": [42, 44, 46],
        "crash": [49, 57],
        "ride": [51, 59]
    }
    for drum_type, pitches in drum_map.items():
        if note_pitch in pitches:
            return drum_type
    return "other"

def extract_drum_stats_by_type(pm):
    drum_map = {
        "kick": [35, 36],
        "snare": [38, 40],
        "hihat": [42, 44, 46],
        "crash": [49, 57],
        "ride": [51, 59]
    }
    stats_by_type = {}
    for drum_type, pitches in drum_map.items():
        onset_devs = []
        velocities = []
        for inst in pm.instruments:
            if inst.is_drum:
                for note in inst.notes:
                    if note.pitch in pitches:
                        quantization = 0.25
                        quantized_time = round(note.start / quantization) * quantization
                        deviation = note.start - quantized_time
                        onset_devs.append(deviation)
                        velocities.append(note.velocity)
        if onset_devs:
            stats_by_type[drum_type] = {
                "onset_mean": np.mean(onset_devs),
                "onset_std": np.std(onset_devs),
                "velocity_mean": np.mean(velocities),
                "velocity_std": np.std(velocities)
            }
        else:
            stats_by_type[drum_type] = None
    return stats_by_type

def apply_drum_style_to_pm(pm, stats_by_type):
    for inst in pm.instruments:
        if inst.is_drum:
            for note in inst.notes:
                drum_type = get_drum_type(note.pitch)
                if drum_type in stats_by_type and stats_by_type[drum_type]:
                    stats = stats_by_type[drum_type]
                    random_offset = np.random.normal(loc=stats["onset_mean"], scale=stats["onset_std"])
                    note.start += random_offset
                    note.end += random_offset
                    new_velocity = int(
                        np.clip(np.random.normal(loc=stats["velocity_mean"], scale=stats["velocity_std"]), 1, 127))
                    note.velocity = new_velocity
    return pm

def transfer_drum_style(source_midi_path, target_midi_path, output_midi_path):
    st.info("Transferring drum style...")
    source_pm = pretty_midi.PrettyMIDI(source_midi_path)
    stats_by_type = extract_drum_stats_by_type(source_pm)
    target_pm = pretty_midi.PrettyMIDI(target_midi_path)
    modified_pm = apply_drum_style_to_pm(target_pm, stats_by_type)
    modified_pm.write(output_midi_path)
    st.info("Drum style transferred successfully to: " + output_midi_path)

def classify_drummer(audio_path):
    st.info("Classifying drummer style for audio: " + audio_path)
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfccs, axis=1).reshape(1, -1)
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    prediction = model.predict(features)
    predicted_drummer = le.inverse_transform(prediction)[0]
    st.info("Classification complete. Predicted drummer: " + predicted_drummer)
    return predicted_drummer

# -----------------------------
# New Helper Functions: MIDI Conversion for Separated Instruments
# (with hi-hat open/closed classification)
# -----------------------------
def classify_hihat_segment(segment, sr):
    """
    Classify a hi-hat segment as open or closed.
    Uses a simple RMS decay heuristic: compares the RMS of the first 30ms to the next 70ms.
    If the later energy is at least 70% of the initial, classify as "open"; otherwise "closed".
    """
    n_samples_30ms = int(0.03 * sr)
    n_samples_70ms = int(0.07 * sr)
    if len(segment) < n_samples_30ms + 1:
        return "closed"
    initial = segment[:n_samples_30ms]
    later = segment[n_samples_30ms:n_samples_30ms + n_samples_70ms]
    rms_initial = np.sqrt(np.mean(initial ** 2))
    rms_later = np.sqrt(np.mean(later ** 2)) if len(later) > 0 else 0
    ratio = rms_later / (rms_initial + 1e-6)
    if ratio > 0.7:
        return "open"
    else:
        return "closed"

def convert_separated_audio_to_midi_for_instrument(audio_path, output_midi_path, instrument):
    """
    Convert a separated audio file for a specific drum instrument to MIDI.

    For hi-hat (or "hh"), it further classifies each onset as open (MIDI note 46)
    or closed (MIDI note 42) using a simple RMS decay heuristic.
    For other instruments, fixed General MIDI notes are used:
      - Kick: 36, Snare: 38, Tom(s): 45, Ride: 51, Crash: 49.
    """
    y, sr = librosa.load(audio_path)
    print(f"[{instrument}] Loaded audio shape: {y.shape}")
    print(f"[{instrument}] Duration (seconds): {len(y) / sr:.2f}")

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = np.arange(len(onset_env)) / sr
    plt.figure(figsize=(10, 4))
    plt.plot(times, onset_env, label="Onset Strength")
    plt.xlabel("Time (s)")
    plt.ylabel("Onset Strength")
    plt.title(f"Onset Envelope for {instrument}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    onset_kwargs = {"sr": sr}
    if instrument.lower() in ["kick", "snare"]:
        onset_kwargs.update({
            "pre_max": 20,
            "post_max": 20,
            "pre_avg": 100,
            "post_avg": 100,
            "delta": 0.3,
            "wait": 10
        })
    elif instrument.lower() in ["tom", "toms"]:
        onset_kwargs.update({
            "pre_max": 30,
            "post_max": 30,
            "pre_avg": 120,
            "post_avg": 120,
            "delta": 0.4,
            "wait": 20
        })
    elif instrument.lower() in ["hihat", "hh"]:
        onset_kwargs.update({
            "pre_max": 10,
            "post_max": 10,
            "pre_avg": 30,
            "post_avg": 30,
            "delta": 0.03,
            "wait": 2
        })
    else:
        onset_kwargs.update({
            "pre_max": 15,
            "post_max": 15,
            "pre_avg": 80,
            "post_avg": 80,
            "delta": 0.2,
            "wait": 10
        })

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, **onset_kwargs)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"[{instrument}] Detected onsets: {onset_times}")

    pm = pretty_midi.PrettyMIDI()
    drum_instr = pretty_midi.Instrument(program=0, is_drum=True)

    if instrument.lower() not in ["hihat", "hh"]:
        if instrument.lower() == "kick":
            fixed_pitch = 36
        elif instrument.lower() == "snare":
            fixed_pitch = 38
        elif instrument.lower() in ["tom", "toms"]:
            fixed_pitch = 45
        elif instrument.lower() == "ride":
            fixed_pitch = 51
        elif instrument.lower() == "crash":
            fixed_pitch = 49
        else:
            fixed_pitch = 38
    if len(onset_times) == 0:
        print(f"[{instrument}] Warning: No onsets detected in {audio_path}!")

    for onset in onset_times:
        print(f"[{instrument}] Processing onset at: {onset:.3f}")
        start_sample = int(onset * sr)
        window = y[start_sample:start_sample + int(0.1 * sr)]
        amplitude = np.max(np.abs(window)) if len(window) > 0 else 0.1
        velocity = int(np.clip(amplitude * 127 / np.max(np.abs(y)), 20, 127))
        if instrument.lower() in ["hihat", "hh"]:
            classification = classify_hihat_segment(window, sr)
            if classification == "open":
                current_pitch = 46
            else:
                current_pitch = 42
            print(f"[{instrument}] Onset at {onset:.3f} classified as {classification}")
        else:
            current_pitch = fixed_pitch
        note = pretty_midi.Note(velocity=velocity, pitch=current_pitch, start=onset, end=onset + 0.1)
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
    For hi-hat, the conversion function further distinguishes open vs. closed.
    The MIDI files are saved in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    instruments = ["kick", "snare", "tom", "hihat", "ride", "crash"]

    for inst in instruments:
        print(f"Searching for files for instrument: {inst}")
        inst_files = find_separated_instrument_files(separation_dir, inst)
        if not inst_files:
            print(f"No files found for instrument: {inst}")
        else:
            for file_path in inst_files:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                midi_filename = f"{base_name}.mid"
                output_midi_path = os.path.join(output_dir, midi_filename)
                print(f"Converting {file_path} to MIDI as {midi_filename}")
                convert_separated_audio_to_midi_for_instrument(file_path, output_midi_path, instrument=inst)

# -----------------------------
# New Function: Automated Groove Analysis
# -----------------------------
def automated_groove_analysis(drummer_name, video_type, num_tracks, selected_band="All"):
    """
    Automatically search YouTube for a specified number of drum tracks for the given drummer.
    The search query uses the drummer's band names along with the video_type keyword ("live", "studio", or "isolated").
    If a specific band is selected, only that band's name is used.
    """
    # Determine which bands to search:
    if selected_band and selected_band != "All":
        bands = [selected_band]
    else:
        bands = FAMOUS_DRUMMERS.get(drummer_name, [])
        if not bands:
            bands = [drummer_name]

    search_queries = []
    for band in bands:
        # Build query based on the video_type selected
        if video_type.lower() == "live":
            query = f"{band} {drummer_name} live performance"
        elif video_type.lower() == "studio":
            query = f"{band} {drummer_name} studio recording"
        elif video_type.lower() == "isolated":
            query = f"{band} {drummer_name} isolated drum track"
        else:
            query = f"{band} {drummer_name}"
        search_queries.append(query)

    all_videos = []
    for query in search_queries:
        st.info("Searching YouTube with query: " + query)
        videos = search_youtube_videos(query)
        all_videos.extend(videos)
    # Deduplicate videos by URL
    unique_videos = list({v['url']: v for v in all_videos}.values())
    # Limit to the desired number of tracks
    selected_videos = unique_videos[:num_tracks]

    all_groove_metrics = []
    inst_groove = {}  # instrument-specific metrics

    for i, video in enumerate(selected_videos):
        st.write(f"Processing track {i + 1}: {video['title']}")
        # Create a unique filename for the track audio
        raw_filename = f"{drummer_name.replace(' ', '_')}_track_{i + 1}.wav"
        full_audio_filename = clean_filename(raw_filename)
        full_audio_path = os.path.join(AUDIO_DOWNLOAD_DIR, full_audio_filename)

        # Download the audio (limited to 3 minutes)
        download_youtube_audio(video['url'], full_audio_path)

        # Run drum separation on this file
        run_drum_sep_inference_with_progress(
            model_type="mdx23c",
            config_path=os.path.join(
                "C:\\Users\\goldw\\PycharmProjects\\DrumTracksAI\\Music-Source-Separation-Training", "configs",
                "config_drumsep_mdx23c.yaml"),
            checkpoint_path=os.path.join(
                "C:\\Users\\goldw\\PycharmProjects\\DrumTracksAI\\Music-Source-Separation-Training", "pretrained",
                "drumsep_mdx23c.ckpt"),
            input_folder=AUDIO_DOWNLOAD_DIR,
            store_dir=SEPARATED_RESULTS_DIR
        )
        # Assume the separation results for this track are stored in a subfolder named after the audio file (without extension)
        track_folder = os.path.join(SEPARATED_RESULTS_DIR, os.path.splitext(full_audio_filename)[0])
        if not os.path.exists(track_folder):
            st.write(f"Separation folder not found for track {i + 1}. Skipping.")
            continue

        # Convert all separated stems for this track to MIDI in a temporary folder
        temp_midi_folder = os.path.join(MIDI_CONVERTED_DIR, f"{drummer_name.replace(' ', '_')}_track_{i + 1}")
        os.makedirs(temp_midi_folder, exist_ok=True)
        convert_all_separated_to_midi(track_folder, temp_midi_folder)

        # Compute groove metrics from each MIDI file in the temporary folder
        for midi_file in os.listdir(temp_midi_folder):
            if midi_file.endswith(".mid"):
                midi_path = os.path.join(temp_midi_folder, midi_file)
                try:
                    pm = pretty_midi.PrettyMIDI(midi_path)
                    gm = compute_groove_metric(pm, resolution=0.05)
                    if gm is not None:
                        all_groove_metrics.append(gm)
                        # Determine instrument from the midi filename
                        for inst in ["kick", "snare", "tom", "hihat", "ride", "crash"]:
                            if inst in midi_file.lower():
                                inst_groove.setdefault(inst, []).append(gm)
                except Exception as e:
                    st.write(f"Error processing MIDI {midi_file}: {e}")

    if all_groove_metrics:
        global_groove = np.mean(all_groove_metrics)
    else:
        global_groove = None

    instrument_metrics = {inst: np.mean(vals) for inst, vals in inst_groove.items() if vals}

    analysis_result = {
        "drummer": drummer_name,
        "selected_band": selected_band,
        "video_type": video_type,
        "num_tracks": num_tracks,
        "global_groove": global_groove,
        "instrument_groove": instrument_metrics
    }

    # Update the drummer characteristics database
    db_file = "drummer_characteristics_db.json"
    if os.path.exists(db_file):
        with open(db_file, "r") as f:
            db = json.load(f)
    else:
        db = {}
    db[drummer_name] = analysis_result
    with open(db_file, "w") as f:
        json.dump(db, f, indent=4)

    return analysis_result

# -----------------------------
# UI Section: Apply Drum Style to MIDI
# -----------------------------
st.subheader("4. Apply Drum Style to MIDI")
uploaded_midi = st.file_uploader("Upload your MIDI file", type=["mid", "midi"])
if uploaded_midi is not None:
    target_midi_path = os.path.join(USER_UPLOADS_DIR, uploaded_midi.name)
    with open(target_midi_path, "wb") as f:
        f.write(uploaded_midi.getbuffer())
    st.info("MIDI file uploaded: " + uploaded_midi.name)
    if st.button("Apply Drum Style"):
        source_midi_path = os.path.join(TRAINING_MIDI_DIR, f"{drummer_name}.mid")
        if not os.path.exists(source_midi_path):
            st.error("Training MIDI sample for the selected drummer not found.")
        else:
            output_midi_path = os.path.join(USER_UPLOADS_DIR, f"styled_{uploaded_midi.name}")
            try:
                transfer_drum_style(source_midi_path, target_midi_path, output_midi_path)
                st.success("Drum style transferred successfully! Download your modified MIDI file below.")
                with open(output_midi_path, "rb") as f:
                    st.download_button("Download Modified MIDI", f, file_name=f"styled_{uploaded_midi.name}")
            except Exception as e:
                st.error(f"Error applying drum style: {e}")

# -----------------------------
# Section 5: Classify Drummer Style from Audio
# -----------------------------
st.subheader("5. Classify Drummer Style from Audio")
uploaded_audio = st.file_uploader("Upload an audio file for classification", type=["wav", "mp3"], key="audio_classify")
if uploaded_audio is not None:
    audio_path = os.path.join(USER_UPLOADS_DIR, uploaded_audio.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.getbuffer())
    st.info("Audio file uploaded: " + uploaded_audio.name)
    if st.button("Classify Drummer Style"):
        try:
            predicted_drummer = classify_drummer(audio_path)
            st.success(f"Predicted Drummer: {predicted_drummer}")
        except Exception as e:
            st.error(f"Error during classification: {e}")

# -----------------------------
# Section 6: Analyze Drummer Characteristics & Groove
# -----------------------------
st.subheader("6. Analyze Drummer Characteristics & Groove")
st.write(
    "Upload multiple separated drum MIDI files (e.g., converted from audio) representing different songs from the drummer's bands. The app will compute onset/velocity statistics and groove metrics.")
uploaded_midi_files = st.file_uploader("Upload separated MIDI files", type=["mid", "midi"], accept_multiple_files=True,
                                       key="analysis")
if uploaded_midi_files:
    if st.button("Analyze Drummer Characteristics"):
        profiles = {}
        groove_values = []
        for midi_file in uploaded_midi_files:
            try:
                midi_data = midi_file.read()
                pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_data))
                stats = extract_drum_stats_by_type(pm)
                for drum in stats:
                    if stats[drum] is not None:
                        if drum not in profiles:
                            profiles[drum] = {"onset_means": [], "onset_stds": [], "velocity_means": [],
                                              "velocity_stds": []}
                        profiles[drum]["onset_means"].append(stats[drum]["onset_mean"])
                        profiles[drum]["onset_stds"].append(stats[drum]["onset_std"])
                        profiles[drum]["velocity_means"].append(stats[drum]["velocity_mean"])
                        profiles[drum]["velocity_stds"].append(stats[drum]["velocity_std"])
                groove = compute_groove_metric(pm, resolution=0.05)
                if groove is not None:
                    groove_values.append(groove)
                    st.write(f"Groove metric for {midi_file.name}: {groove:.4f} sec deviation")
            except Exception as e:
                st.error(f"Error processing {midi_file.name}: {e}")
        aggregated = {}
        for drum, data in profiles.items():
            aggregated[drum] = {
                "onset_mean": np.mean(data["onset_means"]) if data["onset_means"] else None,
                "onset_std": np.mean(data["onset_stds"]) if data["onset_stds"] else None,
                "velocity_mean": np.mean(data["velocity_means"]) if data["velocity_means"] else None,
                "velocity_std": np.mean(data["velocity_stds"]) if data["velocity_stds"] else None,
            }
        if groove_values:
            overall_groove = np.mean(groove_values)
            st.write(f"Overall average groove deviation: {overall_groove:.4f} sec")
        db_file = "drummer_characteristics_db.json"
        if os.path.exists(db_file):
            with open(db_file, "r") as f:
                db = json.load(f)
        else:
            db = {}
        db[drummer_name] = {"characteristics": aggregated, "global_groove": overall_groove if groove_values else None}
        with open(db_file, "w") as f:
            json.dump(db, f, indent=4)
        st.success(f"Drummer characteristics for {drummer_name} have been updated in the database!")
        st.json(db[drummer_name])

# -----------------------------
# Section 7: Convert Audio to MIDI
# -----------------------------
st.subheader("7. Convert Audio to MIDI from YouTube Download")
audio_source_option = st.radio("Select audio source to convert:", options=["Full Audio", "Separated Drum Stem"])
if audio_source_option == "Full Audio":
    full_audio_filename = clean_filename(f"{drummer_name.replace(' ', '_')}_full.wav")
    full_audio_path = os.path.join(AUDIO_DOWNLOAD_DIR, full_audio_filename)
    st.info("Looking for full audio file: " + full_audio_path)
    if os.path.exists(full_audio_path):
        if st.button("Convert Full Audio to MIDI"):
            midi_output_path = os.path.join(MIDI_CONVERTED_DIR, f"{drummer_name.replace(' ', '_')}_drum_track.mid")
            try:
                convert_audio_to_midi(full_audio_path, midi_output_path, separate_instruments=False)
                st.success("Conversion complete!")
                with open(midi_output_path, "rb") as f:
                    st.download_button("Download Converted MIDI", f, file_name=os.path.basename(midi_output_path))
            except Exception as e:
                st.error("Error during conversion: " + str(e))
    else:
        st.warning("Full audio file not found. Please download audio from YouTube first.")
else:
    st.write("Selected conversion mode: Separated Drum Stem")
    st.write("Available files in separation results folder:")
    separated_files = []
    for root, dirs, files in os.walk(SEPARATED_RESULTS_DIR):
        for f in files:
            if f.lower().endswith(".wav"):
                separated_files.append(os.path.join(root, f))
    st.write(separated_files)
    instrument_options = ["kick", "snare", "tom", "hihat", "ride", "crash"]
    selected_instrument = st.selectbox("Select Instrument", instrument_options)
    filtered_files = find_separated_instrument_files(SEPARATED_RESULTS_DIR, selected_instrument)
    st.write("Files found for", selected_instrument, ":", filtered_files)
    if filtered_files:
        selected_file = st.selectbox("Select file for conversion", filtered_files)
        if st.button("Convert Selected Stem to MIDI"):
            midi_output_path = os.path.join(MIDI_CONVERTED_DIR,
                                            f"{os.path.splitext(os.path.basename(selected_file))[0]}.mid")
            try:
                temp_stem_path = os.path.join(os.getcwd(), "temp_stem.wav")
                copy_command = f'copy "{selected_file}" "{temp_stem_path}"'
                st.info("Copying file using command: " + copy_command)
                subprocess.check_call(copy_command, shell=True)
                st.info("File copied to temporary location for processing.")
                convert_separated_audio_to_midi_for_instrument(temp_stem_path, midi_output_path,
                                                               instrument=selected_instrument)
                st.success("Conversion complete!")
                with open(midi_output_path, "rb") as f:
                    st.download_button("Download Converted MIDI", f, file_name=os.path.basename(midi_output_path))
            except Exception as e:
                st.error("Error during conversion: " + str(e))
    else:
        st.warning("No files found for the selected instrument. Please ensure the separated drum stems are correctly labeled.")

# -----------------------------
# Section 8: Apply Groove to Existing MIDI
# -----------------------------
st.subheader("8. Apply Groove to Existing MIDI")
st.write(
    "Upload a fully quantized MIDI file and apply a drummer's groove (using the global groove metric) to modify its timing.")
uploaded_quantized_midi = st.file_uploader("Upload Quantized MIDI File", type=["mid", "midi"], key="quantized_midi")
if uploaded_quantized_midi is not None:
    quantized_midi_path = os.path.join(USER_UPLOADS_DIR, uploaded_quantized_midi.name)
    with open(quantized_midi_path, "wb") as f:
        f.write(uploaded_quantized_midi.getbuffer())
    st.info("Quantized MIDI file uploaded: " + uploaded_quantized_midi.name)
    try:
        with open("global_groove_analysis.json", "r") as f:
            groove_data = json.load(f)
        groove_metric = groove_data.get("global_groove", 0)
        st.write(f"Global groove metric for {groove_data.get('drummer', 'Unknown')}: {groove_metric:.4f} sec")
    except Exception as e:
        st.error("Error reading global groove analysis: " + str(e))
        groove_metric = 0
    if st.button("Apply Groove to MIDI"):
        grooved_midi_path = os.path.join(USER_UPLOADS_DIR, f"grooved_{uploaded_quantized_midi.name}")
        try:
            apply_groove_to_midi(quantized_midi_path, grooved_midi_path, groove_metric)
            st.success("Groove applied successfully!")
            with open(grooved_midi_path, "rb") as f:
                st.download_button("Download Grooved MIDI", f, file_name=os.path.basename(grooved_midi_path))
        except Exception as e:
            st.error("Error applying groove: " + str(e))

# -----------------------------
# Section 9: Automated Groove Analysis for Drummer
# -----------------------------
st.subheader("9. Automated Groove Analysis for Drummer")
st.write("Using the global selected drummer, choose video type (Live, Studio, or Isolated), "
         "select a band (or All) to narrow the search, and specify the number of tracks (1 to 50) for analysis.")

st.write("Drummer Selected:", drummer_name)
drummer_bands = FAMOUS_DRUMMERS.get(drummer_name, [])
selected_band = st.selectbox("Select Band:", ["All"] + drummer_bands)
video_type = st.radio("Select Video Type:", options=["Live", "Studio", "Isolated"])
num_tracks = st.slider("Number of Tracks to Analyze:", min_value=1, max_value=50, value=5)

if st.button("Run Automated Groove Analysis"):
    st.info(f"Running automated analysis for {drummer_name} ({video_type}, Band: {selected_band}) on {num_tracks} tracks...")
    analysis_result = automated_groove_analysis(drummer_name, video_type, num_tracks, selected_band=selected_band)
    if analysis_result:
        st.success("Automated groove analysis completed!")
        st.json(analysis_result)
    else:
        st.error("Automated groove analysis failed or returned no data.")

# -----------------------------
# Note on Monetization & Plugin Integration
# -----------------------------
st.markdown("---")
st.write("Note: Monetization via PayPal and integration into a VST/AU plugin are not handled directly in this app.")
