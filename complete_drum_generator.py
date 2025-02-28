import os
import streamlit as st
import yt_dlp as youtube_dl
import librosa
import pretty_midi
import numpy as np


# -----------------------------
# Input Source & YouTube Download
# -----------------------------
def download_youtube_audio(url, output_template="youtube_audio.%(ext)s"):
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
    final_output = os.path.join(os.getcwd(), "youtube_audio.wav")
    if not os.path.exists(final_output):
        st.error("Downloaded file not found.")
        return None
    st.success(f"Audio downloaded to {final_output}")
    return final_output


# -----------------------------
# Guide Track Extraction (Placeholder)
# -----------------------------
def extract_guide_track(input_file_path, source_type, target_instrument="Bass"):
    st.info(f"Extracting guide track for {target_instrument} from {source_type} input...")
    # Placeholder: Replace with source separation or MIDI track selection.
    return input_file_path


# -----------------------------
# Groove Analysis with Spinner & Formatting Fix
# -----------------------------
def analyze_guide_track(audio_path, trim_silence=True, top_db=30, hop_length=256):
    with st.spinner("Analyzing guide track groove..."):
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
        groove_metric = np.std(intervals) if len(intervals) > 0 else 0
        return {"tempo": tempo, "beat_times": beat_times.tolist(), "groove_metric": float(groove_metric)}


# -----------------------------
# Drumming Style Selection (Dummy Data)
# -----------------------------
def select_drum_style():
    # Dummy style parameters from E-GMD; can be replaced with real database lookups.
    styles = {
        "Funk": {"swing": 0.05, "dynamics": 0.1},
        "Rock": {"swing": 0.03, "dynamics": 0.15},
        "Jazz": {"swing": 0.08, "dynamics": 0.2},
        "Pop": {"swing": 0.02, "dynamics": 0.1}
    }
    selected = st.selectbox("Select Drumming Style:", list(styles.keys()))
    st.write("Selected style:", selected)
    return styles[selected]


# -----------------------------
# Arrangement Specification
# -----------------------------
def get_arrangement():
    st.write("### Arrangement Specification")
    measures_intro = st.number_input("Measures for Intro:", value=4)
    measures_verse = st.number_input("Measures for Verse:", value=8)
    measures_chorus = st.number_input("Measures for Chorus:", value=4)
    measures_breakdown = st.number_input("Measures for Breakdown:", value=2)
    measures_outro = st.number_input("Measures for Outro:", value=4)
    song_tempo = st.number_input("Desired Song Tempo (BPM):", value=120)
    time_signature = st.selectbox("Time Signature:", options=["4/4", "3/4", "6/8"])
    arrangement = {
        "measures_intro": measures_intro,
        "measures_verse": measures_verse,
        "measures_chorus": measures_chorus,
        "measures_breakdown": measures_breakdown,
        "measures_outro": measures_outro,
        "song_tempo": song_tempo,
        "time_signature": time_signature
    }
    st.write("Arrangement:", arrangement)
    return arrangement


# -----------------------------
# Articulation & Output Mode Options
# -----------------------------
def get_articulation_options():
    st.write("### Articulation Options")
    output_mode = st.selectbox("Select Output Mode for Drum Articulations:", options=["MIDI notes", "MIDI CC"])
    # For MIDI notes mode, we use default note assignments.
    # For MIDI CC mode, the user provides a CC number (example: hi-hat is often CC #4).
    if output_mode == "MIDI notes":
        hi_hat_note = 42  # default hi-hat note (closed)
        snare_notes = {"center": 38, "outer": 40, "rim": 37, "side_stick": 39}
        ride_note = 51  # default ride note
        st.write("Using MIDI note articulations.")
        return {"mode": "MIDI notes", "hi_hat": hi_hat_note, "snare": snare_notes, "ride": ride_note}
    else:
        hi_hat_cc = st.number_input("Hi-Hat MIDI CC number:", value=4)
        snare_cc = st.number_input("Snare MIDI CC number:", value=5)
        ride_cc = st.number_input("Ride MIDI CC number:", value=6)
        st.write("Using MIDI CC articulations.")
        return {"mode": "MIDI CC", "hi_hat": hi_hat_cc, "snare": snare_cc, "ride": ride_cc}


# -----------------------------
# Basic Rhythm Template Selection
# -----------------------------
def select_basic_rhythm():
    st.write("### Basic Rhythm Template")
    rhythms = {
        "Standard 4/4": {
            "kick": [0, 2],  # kick on beat 1 and 3 (0-indexed)
            "snare": [1, 3],  # snare on beat 2 and 4
            "hi_hat": "off",  # hi-hat on off-beats (eighth notes)
        },
        "Rock": {
            "kick": [0, 2],
            "snare": [1, 3],
            "hi_hat": "quarter",  # hi-hat on every beat
        },
        "Funk Shuffle": {
            "kick": [0],
            "snare": [2],
            "hi_hat": "shuffle",  # a swung hi-hat pattern (placeholder)
        }
    }
    selected_rhythm = st.selectbox("Select Basic Rhythm Template:", list(rhythms.keys()))
    st.write("Selected basic rhythm:", selected_rhythm)
    return rhythms[selected_rhythm]


# -----------------------------
# Drum Track Generation with Improved Timing and Articulations
# -----------------------------
def generate_drum_track(guide_analysis, drum_style, arrangement, articulations, basic_rhythm, groove_weight):
    """
    Generate a drum MIDI track based on the guide analysis and user parameters.
    - The basic rhythm template defines which beats get which instruments.
    - The groove factor is applied to alter note timing based on the groove_weight.
    - In the chorus section, the user can choose to use the ride cymbal instead of hi-hat.
    - The output_mode (MIDI notes or MIDI CC) is determined by the articulations dictionary.
    """
    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True)

    song_tempo = arrangement.get("song_tempo", 120)
    beats_per_measure = 4  # assuming 4/4
    measure_duration = (60.0 / song_tempo) * beats_per_measure

    # Total measures: sum of all sections.
    total_measures = (arrangement.get("measures_intro", 0) +
                      arrangement.get("measures_verse", 0) +
                      arrangement.get("measures_chorus", 0) +
                      arrangement.get("measures_breakdown", 0) +
                      arrangement.get("measures_outro", 0))

    groove_metric = guide_analysis.get("groove_metric", 0)
    # Scale the groove metric by the user-selected weight.
    effective_groove = groove_metric * groove_weight

    current_time = 0.0
    for measure in range(total_measures):
        # For this example, assume the chorus section is the middle third.
        if measure >= total_measures // 3 and measure < 2 * total_measures // 3:
            section = "chorus"
        else:
            section = "other"

        for beat in range(beats_per_measure):
            beat_duration = measure_duration / beats_per_measure
            beat_time = current_time + beat * beat_duration

            # Apply basic rhythm for kick.
            if beat in basic_rhythm["kick"]:
                offset = np.random.normal(0, effective_groove)
                note_time = beat_time + offset
                kick = pretty_midi.Note(velocity=100, pitch=36, start=note_time, end=note_time + 0.1)
                drum.notes.append(kick)
            # Apply basic rhythm for snare.
            if beat in basic_rhythm["snare"]:
                offset = np.random.normal(0, effective_groove)
                note_time = beat_time + offset
                # In chorus, choose a random articulation from snare options; else use center.
                if section == "chorus":
                    snare_art = np.random.choice(list(articulations["snare"].keys()))
                    snare_pitch = articulations["snare"][snare_art] if articulations["mode"] == "MIDI notes" else 38
                else:
                    snare_pitch = articulations["snare"]["center"] if articulations["mode"] == "MIDI notes" else 38
                snare = pretty_midi.Note(velocity=100, pitch=snare_pitch, start=note_time, end=note_time + 0.1)
                drum.notes.append(snare)

            # Apply hi-hat or ride based on basic rhythm.
            # For hi-hat, if basic_rhythm indicates "off" (off-beats), schedule at mid-beat.
            if basic_rhythm["hi_hat"] == "off":
                offset = np.random.normal(0, effective_groove)
                # For chorus, optionally use ride cymbal if selected by style (simulate with 50% chance).
                if section == "chorus" and np.random.rand() < 0.5:
                    # Use ride articulation.
                    ride_pitch = articulations["ride"] if articulations["mode"] == "MIDI notes" else 51
                    note_time = beat_time + (beat_duration * 0.5) + offset
                    note = pretty_midi.Note(velocity=90, pitch=ride_pitch, start=note_time, end=note_time + 0.1)
                    drum.notes.append(note)
                else:
                    hi_hat_pitch = articulations["hi_hat"] if articulations["mode"] == "MIDI notes" else 42
                    note_time = beat_time + (beat_duration * 0.5) + offset
                    note = pretty_midi.Note(velocity=90, pitch=hi_hat_pitch, start=note_time, end=note_time + 0.05)
                    drum.notes.append(note)
            elif basic_rhythm["hi_hat"] == "quarter":
                # Hi-hat on every beat.
                offset = np.random.normal(0, effective_groove)
                hi_hat_pitch = articulations["hi_hat"] if articulations["mode"] == "MIDI notes" else 42
                note_time = beat_time + offset
                note = pretty_midi.Note(velocity=90, pitch=hi_hat_pitch, start=note_time, end=note_time + 0.05)
                drum.notes.append(note)
            elif basic_rhythm["hi_hat"] == "shuffle":
                # For shuffle, assume hi-hat hits on an uneven pattern.
                offset = np.random.normal(0, effective_groove)
                hi_hat_pitch = articulations["hi_hat"] if articulations["mode"] == "MIDI notes" else 42
                # This is a placeholder: hit at 0.4 of beat and 0.9 of beat.
                note_time1 = beat_time + (beat_duration * 0.4) + offset
                note_time2 = beat_time + (beat_duration * 0.9) + offset
                note1 = pretty_midi.Note(velocity=90, pitch=hi_hat_pitch, start=note_time1, end=note_time1 + 0.05)
                note2 = pretty_midi.Note(velocity=90, pitch=hi_hat_pitch, start=note_time2, end=note_time2 + 0.05)
                drum.notes.extend([note1, note2])
        current_time += measure_duration

    # If using MIDI CC output mode, add a control change message for hi-hat articulations at each measure.
    if articulations["mode"] == "MIDI CC":
        # For simplicity, add a hi-hat CC at the start of each measure.
        cc_number = st.number_input("Hi-Hat CC to send:", value=articulations["hi_hat"], key="hi_hat_cc")
        current_time = 0.0
        for m in range(total_measures):
            cc_value = int(64 + (effective_groove * 10))  # Dummy value based on groove.
            cc = pretty_midi.ControlChange(number=cc_number, value=cc_value, time=current_time)
            drum.control_changes.append(cc)
            current_time += measure_duration

    pm.instruments.append(drum)
    return pm


# -----------------------------
# Optional MIDI to Audio Rendering (Placeholder)
# -----------------------------
def render_midi_to_audio(midi_path, sample_repo="drum_samples"):
    st.info("Rendering MIDI to audio (placeholder).")
    audio_output_path = "drum_track_audio.wav"
    # Insert your actual rendering code here.
    return audio_output_path


# -----------------------------
# Main UI
# -----------------------------
def main():
    st.title("Complete Drum Track Generator")
    st.write("""
    This module lets you create a drum track by:
      1. Uploading or selecting a guide song (audio, MIDI, or YouTube).
      2. Extracting a guide track (e.g., Bass) and analyzing its groove.
      3. Selecting a drumming style.
      4. Specifying an arrangement (measures for intro, verse, chorus, breakdown, outro, song tempo, time signature).
      5. Selecting a basic rhythm template.
      6. Adjusting the amount of groove factor with a slider.
      7. Specifying articulation output mode (MIDI notes vs. MIDI CC) and entering CC numbers.
      8. Generating a drum track with the groove factor applied.
      9. Downloading the output as MIDI (or rendering to audio).
    """)

    # Step 1: Input Source
    source_option = st.radio("Select Input Source:", options=["Upload Audio File", "Upload MIDI File", "YouTube URL"])
    input_file_path = None
    if source_option == "Upload Audio File":
        uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"], key="audio_input")
        if uploaded_audio:
            input_file_path = "input_audio.wav"
            with open(input_file_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            st.success("Audio file uploaded.")
    elif source_option == "Upload MIDI File":
        uploaded_midi = st.file_uploader("Upload a MIDI file", type=["mid", "midi"], key="midi_input")
        if uploaded_midi:
            input_file_path = "input_midi.mid"
            with open(input_file_path, "wb") as f:
                f.write(uploaded_midi.getbuffer())
            st.success("MIDI file uploaded.")
    else:
        youtube_url = st.text_input("Enter YouTube URL:")
        if youtube_url:
            input_file_path = download_youtube_audio(youtube_url)

    if not input_file_path:
        st.error("Please provide an input source.")
        return

    # Step 2: Guide Track Extraction
    target_instrument = st.selectbox("Select instrument for guide track extraction:",
                                     options=["Bass", "Guitar", "Vocals", "Drums (default)"])
    guide_track = extract_guide_track(input_file_path, source_type=source_option, target_instrument=target_instrument)
    st.write("Guide track path:", guide_track)

    # Step 3: Groove Analysis
    st.write("### Guide Track Groove Analysis")
    guide_analysis = analyze_guide_track(guide_track)
    if not guide_analysis:
        st.error("Groove analysis failed.")
        return
    st.write("Groove Analysis Result:")
    st.json(guide_analysis)

    # Step 4: Drumming Style Selection
    st.write("### Drumming Style Selection")
    drum_style = select_drum_style()

    # Step 5: Arrangement Specification
    st.write("### Arrangement Specification")
    arrangement = get_arrangement()

    # Step 6: Basic Rhythm Template Selection
    st.write("### Basic Rhythm Template")
    basic_rhythm = select_basic_rhythm()

    # Step 7: Groove Factor Weight
    groove_weight = st.slider("Groove Factor Weight (0 = no timing variation, 1 = full groove effect):", min_value=0.0,
                              max_value=1.0, value=0.5, step=0.05)

    # Step 8: Articulation & Output Mode Options
    art_options = get_articulation_options()

    # Optional: Commercial VST Template (Placeholder)
    use_vst = st.checkbox("Use Commercial VST Template (Placeholder)", value=False)
    if use_vst:
        vst_template = st.selectbox("Select VST Template:",
                                    options=["Superior Drummer Template A", "Template B", "Template C"])
        st.write("Using VST template:", vst_template)
        # In a full implementation, this would override articulation settings.

    # Step 9: Drum Track Generation
    if st.button("Generate Drum Track"):
        if not guide_analysis:
            st.error("No groove analysis available.")
        else:
            drum_pm = generate_drum_track(guide_analysis, drum_style, arrangement, art_options, basic_rhythm,
                                          groove_weight)
            midi_output_path = "generated_drum_track.mid"
            drum_pm.write(midi_output_path)
            st.success("Drum MIDI track generated successfully!")
            st.download_button("Download Drum MIDI", data=open(midi_output_path, "rb").read(),
                               file_name=midi_output_path)

            # Step 10: Optional Audio Rendering
            if st.checkbox("Render Drum Track to Audio"):
                audio_output_path = render_midi_to_audio(midi_output_path)
                st.success("Drum audio generated successfully!")
                st.download_button("Download Drum Audio", data=open(audio_output_path, "rb").read(),
                                   file_name=audio_output_path)


if __name__ == "__main__":
    main()
