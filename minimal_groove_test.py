# test_bass_analysis.py

import os
import streamlit as st
import librosa
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import io


def analyze_bass(audio_path, apply_filter=True, trim_silence=True, top_db=20):
    """
    Analyze the groove of an isolated bass track.

    Loads the audio file, optionally trims leading/trailing silence,
    optionally applies a low-pass filter to approximate bass extraction,
    then performs beat tracking and computes a groove metric (the standard deviation of intervals between beats).
    """
    # Load the audio file at its original sampling rate.
    y, sr = librosa.load(audio_path, sr=None)
    original_duration = len(y) / sr
    st.write(f"Original audio duration: {original_duration:.2f} seconds, Sample rate: {sr} Hz")

    # Optionally trim silence at the beginning and end.
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=top_db)
        trimmed_duration = len(y) / sr
        st.write(f"Trimmed audio duration: {trimmed_duration:.2f} seconds (top_db={top_db})")
    else:
        st.write("No silence trimming applied.")

    # Optionally apply a low-pass filter to focus on bass frequencies.
    if apply_filter:
        cutoff = 150  # Hz, adjust as needed for bass extraction.
        nyquist = sr / 2
        b, a = scipy.signal.butter(4, cutoff / nyquist, btype='low')
        y_proc = scipy.signal.filtfilt(b, a, y)
        st.write("Low-pass filter applied (cutoff = 150 Hz).")
    else:
        y_proc = y
        st.write("No filtering applied; using full audio signal.")

    # Plot the onset envelope for debugging.
    onset_env = librosa.onset.onset_strength(y=y_proc, sr=sr)
    times = librosa.frames_to_time(range(len(onset_env)), sr=sr)
    fig, ax = plt.subplots()
    ax.plot(times, onset_env, label="Onset Strength")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Onset Strength")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf)
    plt.close(fig)

    # Perform beat tracking on the processed signal.
    tempo, beats = librosa.beat.beat_track(y=y_proc, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    st.write(f"Number of beats detected: {len(beats)}")
    if len(beats) > 0:
        st.write("Beat times:", beat_times)
    else:
        st.write("No beats detected.")

    # Compute a dummy groove metric as the standard deviation of the intervals between beats.
    intervals = np.diff(beat_times)
    groove_metric = np.std(intervals) if len(intervals) > 0 else None

    return {
        "tempo": float(tempo),
        "beat_times": beat_times.tolist(),
        "groove_metric": float(groove_metric) if groove_metric is not None else None
    }


def main():
    st.title("Isolated Bass Groove Analysis Test")
    st.write("Upload an isolated bass track (wav or mp3) to test groove analysis.")

    uploaded_file = st.file_uploader("Upload Bass Track", type=["wav", "mp3"])

    # Option to toggle low-pass filtering and silence trimming.
    apply_filter = st.checkbox("Apply Bass (Low-pass) Filter", value=True)
    trim_silence = st.checkbox("Trim Leading/Trailing Silence", value=True)
    top_db = st.number_input("Silence threshold (top_db) for trimming:", value=20)

    if uploaded_file is not None:
        temp_audio_path = "temp_bass.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Bass track uploaded successfully.")

        if st.button("Analyze Bass Groove"):
            try:
                result = analyze_bass(temp_audio_path, apply_filter=apply_filter, trim_silence=trim_silence,
                                      top_db=top_db)
                st.write("Analysis Result:")
                st.json(result)
            except Exception as e:
                st.error(f"Error during analysis: {e}")


if __name__ == '__main__':
    main()
