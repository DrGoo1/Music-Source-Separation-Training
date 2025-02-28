import os
import json
import streamlit as st


@st.cache_data
def load_combined_database(json_path):"C:\Users\goldw\PycharmProjects\DrumTracksAI\egmd_combined_db.json"
    """Load the preprocessed combined E‑GMD database from a JSON file."""
    if not os.path.exists(json_path):
        st.error("Database file not found.")
        return {}
    with open(json_path, "r") as f:
        db = json.load(f)
    return db


def organize_by_style(combined_db):
    """
    Organize the combined database by grouping entries by style.
    The category is defined as the first part of the style string (split by '-') capitalized.
    Returns a dictionary mapping categories to sub-dictionaries of entries.
    """
    organized = {}
    for base, entry in combined_db.items():
        style = entry.get("midi", {}).get("style", "Unknown")
        category = style.split('-')[0].capitalize()
        organized.setdefault(category, {})[base] = entry
    return organized


def main():
    st.title("E‑GMD Style Browser (Optimized)")
    st.write("""
        This tool loads the preprocessed E‑GMD database and organizes it by drumming style.
        Instead of dumping the entire JSON (which can cause memory issues), it shows summary information.
    """)

    json_path = st.text_input("Enter the path to the combined E‑GMD database JSON:",
                              value="egmd_combined_db.json")

    if st.button("Load Database"):
        combined_db = load_combined_database(json_path)
        if not combined_db:
            st.error("Database is empty or could not be loaded.")
            return

        organized_db = organize_by_style(combined_db)

        # Instead of dumping the entire object, show a summary by category.
        summary = {cat: len(entries) for cat, entries in organized_db.items()}
        st.write("**Database Summary by Style Category:**")
        st.write(summary)

        # Let the user choose a category.
        categories = list(organized_db.keys())
        selected_category = st.selectbox("Select Category:", categories)
        category_entries = organized_db[selected_category]
        st.write(f"**{selected_category}** category contains {len(category_entries)} entries.")

        # Display a table summary for this category.
        entry_data = [{"Base": base,
                       "BPM": entry.get("midi", {}).get("bpm", "N/A"),
                       "Time Signature": entry.get("midi", {}).get("time_signature", "N/A")}
                      for base, entry in category_entries.items()]
        st.table(entry_data)

        # Let the user select a specific entry.
        selected_entry_key = st.selectbox("Select an entry:", list(category_entries.keys()))
        selected_entry = category_entries[selected_entry_key]
        st.write("**Details for Selected Entry:**")
        st.json(selected_entry)

        # Play the corresponding audio snippet.
        if st.button("Play Audio Snippet"):
            audio_path = selected_entry.get("audio")
            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path)
            else:
                st.error("Audio file not found.")


if __name__ == "__main__":
    main()
