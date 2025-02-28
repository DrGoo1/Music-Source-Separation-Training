# drum_groove_generator.py

import os

# Disable Intel SVML and force the correct FFmpeg binary to be used
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
os.environ["DISABLE_INTEL_SVML"] = "1"

# Set ffmpeg_path to your alternative FFmpeg build (update this if necessary)
# For example, if you've downloaded a build that avoids SVML issues, extract it to a folder (e.g., C:\ffmpeg_btbn)
ffmpeg_path = r"C:\ffmpeg_btbn\bin"
# Prepend this to PATH so that subprocesses find the correct ffmpeg
os.environ["PATH"] = ffmpeg_path + ";" + os.environ.get("PATH", "")

import streamlit as st
import yt_dlp as youtube_dl
import pretty_midi
import numpy as np
import pandas as pd
import librosa
import io
import subprocess

# -----------------------------
# Debug: Check current FFmpeg version
# -----------------------------
try:
    ffmpeg_ver = subprocess.check_output(["ffmpeg", "-version"], env=os.environ, text=True)
    st.info("Current FFmpeg version:\n" + ffmpeg_ver)
except Exception as e:
    st.error(f"Error checking FFmpeg version: {e}")


# -----------------------------
# Download Audio from YouTube
# -----------------------------
def download_youtube_audio(url, output_template):
    """
    Download the first 3 minutes of audio from a YouTube video using yt-dlp.
    The output file will be saved using an absolute path based on the provided template.
    """
    st.info(f"Downloading audio from URL: {url}")
    abs_output_template = os.path.join(os.getcwd(), output_template)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": abs_output_template,  # e.g., "youtube_audio.%(ext)s"
        "ffmpeg_location": ffmpeg_path,  # Force usage of our chosen FFmpeg binary.
        "download_sections": [{"section": "*0-180"}],  # First 180 seconds (3 minutes)
        "-x": True,  # Extract audio
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192"
        }]
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        st.error(f"yt-dlp download error: {e}")

    final_output = os.path.join(os.getcwd(), "youtube_audio.wav")
    if not os.path.exists(final_output):
        raise FileNotFoundError(f"Downloaded file not found at {final_output}")

    st.info(f"File downloaded successfully to {final_output}")
    return final_output


# -----------------------------
# Analyze Groove (Audio or MIDI)
# -----------------------------
def analyze_groove(file_path, target_instrument="Drums (default)"):
    """
    Analyze the groove of a file.
    If the file is a MIDI file, load it via pretty_midi.
    Otherwise, assume it's audio and use librosa.
    Returns a dictionary with tempo and a dummy groove signature.
    (Replace with your actual analysis code.)
    """
    if file_path.lower().endswith(('.mid', '.midi')):
        try:
            pm = pretty_midi.PrettyMIDI(file_path)
            groove = compute_groove_metric(pm)
            tempo = 120  # Dummy tempo; replace with actual extraction if available.
            return {"tempo": tempo, "groove_signature": [groove],
                    "num_notes": sum(len(inst.notes) for inst in pm.instruments if inst.is_drum)}
        except Exception as e:
            raise Exception(f"Error analyzing MIDI groove: {e}")
    else:
        try:
            y, sr = librosa.load(file_path, sr=None)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            groove_signature = [0.01, 0.012, 0.011, 0.013]  # Dummy signature.
            return {"tempo": tempo, "groove_signature": groove_signature, "num_beats": len(beats)}
        except Exception as e:
            raise Exception(f"Error analyzing audio groove: {e}")


# -----------------------------
# Compute Groove Metric (Helper)
# -----------------------------
def compute_groove_metric(pm, resolution=0.05):
    """
    Compute the average deviation of drum note start times from their nearest quantized value.
    """
    deviations = []
    for inst in pm.instruments:
        if inst.is_drum:
            for note in inst.notes:
                quantized_time = round(note.start / resolution) * resolution
                deviations.append(abs(note.start - quantized_time))
    return np.mean(deviations) if deviations else None


# -----------------------------
# Generate Drum MIDI from Groove and Arrangement (Placeholder)
# -----------------------------
def generate_drum_midi(groove_data, arrangement):
    """
    Generate a drum MIDI file using the groove data and arrangement details.
    This is a placeholder implementation that creates a simple kick pattern.
    Replace this with your actual drum pattern generation logic.
    """
    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True)
    beat_duration = 60.0 / arrangement["tempo"]
    current_time = 0.0
    # For each section, generate a simple pattern (4 beats per bar)
    for section, bars in arrangement["sections"].items():
        for _ in range(int(bars)):
            for beat in range(4):
                start_time = current_time + beat * beat_duration
                end_time = start_time + beat_duration * 0.5  # Note lasts half a beat
                velocity = 80 + int(10 * groove_data["groove_signature"][beat % len(groove_data["groove_signature"])])
                note = pretty_midi.Note(velocity=velocity, pitch=36, start=start_time, end=end_time)
                drum.notes.append(note)
            current_time += 4 * beat_duration
    pm.instruments.append(drum)
    return pm


# -----------------------------
# Main UI
# -----------------------------
def main():
    st.title("Drum Groove Generator")
    st.write("Analyze a song's groove and generate a drum part based on your arrangement specifications.")

    # --- Source Selection ---
    source_option = st.radio("Select Audio Source:", options=["Upload Audio File", "Upload MIDI File", "YouTube URL"])
    input_file_path = None

    if source_option == "Upload Audio File":
        uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"], key="audio_input")
        if uploaded_audio is not None:
            input_file_path = "uploaded_audio.wav"
            with open(input_file_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            st.success("Audio file uploaded.")
    elif source_option == "Upload MIDI File":
        uploaded_midi = st.file_uploader("Upload a MIDI file", type=["mid", "midi"], key="midi_input")
        if uploaded_midi is not None:
            input_file_path = "uploaded_input.mid"
            with open(input_file_path, "wb") as f:
                f.write(uploaded_midi.getbuffer())
            st.success("MIDI file uploaded.")
    else:
        youtube_url = st.text_input("Enter YouTube URL:")
        if youtube_url:
            input_file_path = download_youtube_audio(youtube_url, "youtube_audio.%(ext)s")

    # --- Instrument Choice for Groove Analysis ---
    instrument_choice = st.selectbox("Select Instrument for Groove Analysis:",
                                     options=["Drums (default)", "Bass", "Guitar", "Vocals", "Other"])
    st.write("Instrument selected for groove analysis:", instrument_choice)

    # --- Arrangement Details ---
    st.write("### Arrangement Details")
    bpm = st.number_input("Enter Tempo (BPM):", value=120)
    time_signature = st.selectbox("Select Time Signature:", options=["4/4", "3/4", "6/8"])
    arrangement_sections = {
        "intro": st.number_input("Number of bars for Intro:", value=4),
        "verse": st.number_input("Number of bars for Verse:", value=8),
        "chorus": st.number_input("Number of bars for Chorus:", value=4),
        "breakdown": st.number_input("Number of bars for Breakdown:", value=2),
        "fill": st.number_input("Number of bars for Fills:", value=1),
        "outro": st.number_input("Number of bars for Outro/Rideout:", value=4)
    }

    if st.button("Analyze and Generate Drum Part"):
        if input_file_path is None:
            st.error("Please select an input source (upload a file or provide a YouTube URL).")
        else:
            try:
                st.info("Analyzing groove...")
                groove_data = analyze_groove(input_file_path, target_instrument=instrument_choice)
                st.write("Extracted Groove Data:")
                st.json(groove_data)

                arrangement = {
                    "tempo": bpm,
                    "time_signature": time_signature,
                    "sections": arrangement_sections
                }
                st.write("Arrangement Details:")
                st.json(arrangement)

                st.info("Generating drum MIDI...")
                generated_pm = generate_drum_midi(groove_data, arrangement)
                output_midi_path = "generated_drum_part.mid"
                generated_pm.write(output_midi_path)
                st.success("Drum part generated successfully!")
                with open(output_midi_path, "rb") as f:
                    st.download_button("Download Generated Drum MIDI", f, file_name=output_midi_path)
            except Exception as e:
                st.error(f"Error during drum part generation: {e}")


if __name__ == '__main__':
    main()
