import os
import streamlit as st


# Cache the directory-scanning functions to avoid re-running them unnecessarily.
@st.cache_data
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
                    style = parts[1]  # e.g. "funk-groove1"
                    try:
                        bpm = int(parts[2])
                    except ValueError:
                        bpm = None
                    time_signature = parts[4]  # e.g. "4-4"
                    midi_db[base] = {
                        "midi_path": os.path.join(dirpath, file),
                        "style": style,
                        "bpm": bpm,
                        "time_signature": time_signature
                    }
                else:
                    st.warning(f"Filename does not match expected pattern: {file}")
    return midi_db


@st.cache_data
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


@st.cache_data
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


def organize_by_style(combined_db):
    """
    Organize the combined database by grouping entries by style.
    Returns a dictionary mapping a style category (e.g. "Funk") to a sub-dictionary of entries.
    The category is derived by splitting the style string at '-' and capitalizing the first part.
    """
    organized = {}
    for base, entry in combined_db.items():
        style = entry["midi"].get("style", "Unknown")
        category = style.split('-')[0].capitalize()
        if category in organized:
            organized[category][base] = entry
        else:
            organized[category] = {base: entry}
    return organized


def main():
    st.title("Combined E‑GMD Database Viewer (Cached)")
    st.write("""
        This tool scans your E‑GMD database for MIDI and corresponding audio files,
        then organizes them by drumming style. The scanning process is cached to speed up repeated tests.
        If the page becomes unresponsive, consider using a smaller directory for testing.
    """)

    midi_root = st.text_input("Enter the MIDI root directory:",
                              value=r"G:\E-GMD Dataset\e-gmd-v1.0.0-midi\e-gmd-v1.0.0")
    audio_root = st.text_input("Enter the Audio root directory:",
                               value=r"G:\E-GMD Dataset\e-gmd-v1.0.0\e-gmd-v1.0.0")

    if st.button("Build Combined Database"):
        if not os.path.exists(midi_root):
            st.error("MIDI root directory not found!")
            return
        if not os.path.exists(audio_root):
            st.error("Audio root directory not found!")
            return

        with st.spinner("Scanning directories, please wait..."):
            combined_db = build_combined_database(midi_root, audio_root)
        if not combined_db:
            st.error("No matching MIDI and audio files found.")
            return

        st.write("Combined Database (raw):")
        st.json(combined_db)

        organized_db = organize_by_style(combined_db)
        st.write("Organized by Style:")
        st.json(organized_db)

        # Dropdown to select a category.
        categories = list(organized_db.keys())
        selected_category = st.selectbox("Select Category:", categories)
        styles_in_category = list(organized_db[selected_category].keys())
        selected_entry_key = st.selectbox("Select an entry from " + selected_category, styles_in_category)
        selected_entry = organized_db[selected_category][selected_entry_key]
        st.write(f"Details for {selected_entry_key}:")
        st.json(selected_entry)

        # Option to play the corresponding audio.
        if st.button("Play Audio Snippet"):
            audio_path = selected_entry.get("audio")
            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path)
            else:
                st.error("Audio file not found.")


if __name__ == "__main__":
    main()

