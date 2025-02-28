# egmd_database_viewer.py

import os
import streamlit as st


def build_egmd_database(root_path):
    """
    Recursively scan the root_path for MIDI files that follow the naming convention:
      <id>_<style>_<bpm>_beat_<time signature>_<version>.midi
    Returns a dictionary mapping style names to a list of dictionaries with details.
    """
    database = {}
    for dirpath, dirnames, filenames in os.walk(root_path):
        for file in filenames:
            if file.lower().endswith((".midi", ".mid")):
                base_name = os.path.splitext(file)[0]
                parts = base_name.split("_")
                # Ensure we have enough parts
                if len(parts) >= 5:
                    # Extract style, BPM, and time signature
                    style = parts[1]  # e.g. "funk-groove1"
                    try:
                        bpm = int(parts[2])
                    except ValueError:
                        bpm = None
                    time_signature = parts[4]  # e.g. "4-4"
                    file_path = os.path.join(dirpath, file)
                    entry = {"file_path": file_path, "bpm": bpm, "time_signature": time_signature}
                    if style in database:
                        database[style].append(entry)
                    else:
                        database[style] = [entry]
                else:
                    st.warning(f"Filename does not conform to expected pattern: {file}")
    return database


def main():
    st.title("E-GMD Database Viewer")
    st.write(
        "This tool scans the E-GMD database folder and displays the available drumming styles and associated MIDI files.")

    # Let the user specify the root directory of the E-GMD database.
    # Example: G:\E-GMD Dataset\e-gmd-v1.0.0-midi\e-gmd-v1.0.0\drummer1\eval_session
    root_dir = st.text_input("Enter the root directory for the E-GMD database:",
                             value=r"G:\E-GMD Dataset\e-gmd-v1.0.0-midi\e-gmd-v1.0.0")

    if root_dir:
        if st.button("Build and View Database"):
            if not os.path.exists(root_dir):
                st.error("The specified directory does not exist.")
            else:
                db = build_egmd_database(root_dir)
                if db:
                    st.write("E-GMD Database:")
                    st.json(db)

                    # Optionally, allow selection of a style.
                    style_options = list(db.keys())
                    selected_style = st.selectbox("Select a style to view details:", style_options)
                    st.write(f"Details for {selected_style}:")
                    st.json(db[selected_style])
                else:
                    st.error("No MIDI files found or database is empty.")


if __name__ == "__main__":
    main()
