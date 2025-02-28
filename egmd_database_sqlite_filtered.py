import os
import sqlite3
import ijson
import streamlit as st


# -----------------------------
# Build and Cache the Database
# -----------------------------
@st.cache_data
def create_database(db_path, json_path):
    """
    Create a SQLite database from the preprocessed JSON file.
    The table 'egmd' has columns: base, midi_path, style, bpm, time_signature, and audio_path.
    Data is inserted incrementally using ijson.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS egmd (
            base TEXT PRIMARY KEY,
            midi_path TEXT,
            style TEXT,
            bpm INTEGER,
            time_signature TEXT,
            audio_path TEXT
        )
    ''')
    conn.commit()

    # Only rebuild if the table is empty.
    c.execute("SELECT COUNT(*) FROM egmd")
    count = c.fetchone()[0]
    if count > 0:
        conn.close()
        return

    with open(json_path, "r", encoding="utf-8") as f:
        parser = ijson.kvitems(f, '')
        batch = []
        batch_size = 1000
        for base, entry in parser:
            midi = entry.get("midi", {})
            audio = entry.get("audio", None)
            style = midi.get("style", "Unknown")
            try:
                bpm = int(midi.get("bpm", 0))
            except Exception:
                bpm = None
            time_sig = midi.get("time_signature", None)
            batch.append((base, midi.get("midi_path"), style, bpm, time_sig, audio))
            if len(batch) >= batch_size:
                c.executemany(
                    "INSERT INTO egmd (base, midi_path, style, bpm, time_signature, audio_path) VALUES (?,?,?,?,?,?)",
                    batch)
                conn.commit()
                batch = []
        if batch:
            c.executemany(
                "INSERT INTO egmd (base, midi_path, style, bpm, time_signature, audio_path) VALUES (?,?,?,?,?,?)",
                batch)
            conn.commit()
    conn.close()


def load_database(db_path):
    """Open a new SQLite database connection (with check_same_thread=False) each time."""
    if not os.path.exists(db_path):
        st.error("Database file not found.")
        return None
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn


# -----------------------------
# Query with Filters (Exact or Nearest BPM)
# -----------------------------
def query_database(conn, selected_category, desired_bpm=None, time_signature=None):
    """
    Query the database for entries whose style (lowercased) starts with the selected category.
    If desired_bpm is provided, first try exact matches; if none, order results by the difference.
    Optionally filter by time signature.
    Returns a list of tuples.
    """
    c = conn.cursor()
    base_query = "SELECT base, midi_path, style, bpm, time_signature, audio_path FROM egmd WHERE LOWER(style) LIKE ?"
    params = [selected_category.lower() + "%"]
    if time_signature and time_signature.lower() != "all":
        base_query += " AND LOWER(time_signature) = ?"
        params.append(time_signature.lower())

    if desired_bpm is not None:
        # Try exact match first.
        exact_query = base_query + " AND bpm = ?"
        exact_params = params + [desired_bpm]
        c.execute(exact_query, exact_params)
        results = c.fetchall()
        if not results:
            # If no exact match, order by absolute difference.
            ordered_query = base_query + " ORDER BY ABS(bpm - ?)"
            ordered_params = params + [desired_bpm]
            c.execute(ordered_query, ordered_params)
            results = c.fetchall()
    else:
        c.execute(base_query, params)
        results = c.fetchall()
    return results


@st.cache_data
def get_distinct_categories(db_path):
    """
    Query the SQLite database for distinct style categories.
    The category is derived as the first part of the style string (split by '-') capitalized.
    Returns a sorted list of unique categories.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT DISTINCT style FROM egmd")
    rows = c.fetchall()
    conn.close()
    categories = set()
    for row in rows:
        style = row[0] if row[0] else "Unknown"
        if '-' in style:
            category = style.split('-')[0].capitalize()
        else:
            category = style.capitalize()
        categories.add(category)
    return sorted(list(categories))


@st.cache_data
def build_combined_database(midi_root, audio_root):
    """
    Build a combined database by matching MIDI and audio files based on the base filename.
    Returns a dictionary mapping base filename to a dict with keys "midi" and "audio".
    """

    def build_midi_database(midi_root):
        midi_db = {}
        for dirpath, _, filenames in os.walk(midi_root):
            for file in filenames:
                if file.lower().endswith((".midi", ".mid")):
                    base = os.path.splitext(file)[0]
                    parts = base.split('_')
                    if len(parts) >= 5:
                        style = parts[1]
                        try:
                            bpm = int(parts[2])
                        except ValueError:
                            bpm = None
                        time_signature = parts[4]
                        midi_db[base] = {
                            "midi_path": os.path.join(dirpath, file),
                            "style": style,
                            "bpm": bpm,
                            "time_signature": time_signature
                        }
        return midi_db

    def build_audio_database(audio_root):
        audio_db = {}
        for dirpath, _, filenames in os.walk(audio_root):
            for file in filenames:
                if file.lower().endswith(('.wav', '.mp3')):
                    base = os.path.splitext(file)[0]
                    audio_db[base] = os.path.join(dirpath, file)
        return audio_db

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


# -----------------------------
# Main UI
# -----------------------------
def main():
    st.title("E‑GMD Database Browser (Filtered, Paginated, Limited)")
    st.write("""
        This tool loads a preprocessed E‑GMD database (via SQLite) and lets you filter entries by style, desired BPM, and time signature.
        Only the Style, BPM, and Time Signature columns are displayed.
        Results are paginated to show 5 entries per page, with an option to load more.
    """)

    json_path = st.text_input("Path to preprocessed JSON database:", value="egmd_combined_db.json")
    db_path = st.text_input("Path for SQLite database file:", value="egmd.db")

    if st.button("Build Database"):
        if not os.path.exists(json_path):
            st.error("JSON database file not found!")
            return
        with st.spinner("Building SQLite database, please wait..."):
            create_database(db_path, json_path)
        st.success("Database built successfully!")

    # Use drop-down for style category selection.
    categories = get_distinct_categories(db_path)
    if not categories:
        st.error("No categories found. Build the database first.")
        return
    selected_category = st.selectbox("Select Style Category:", categories)

    st.write("### Filter Options")
    desired_bpm = st.number_input("Enter Desired BPM:", min_value=0, value=120)
    time_signature = st.selectbox("Time Signature:", options=["All", "4-4", "3-4", "6-8"])

    if st.button("Query Database"):
        conn = load_database(db_path)
        results = query_database(conn, selected_category, desired_bpm=desired_bpm, time_signature=time_signature)
        if results:
            st.session_state.results = results
            st.session_state.offset = 0
        else:
            st.error("No entries found for the given filters.")

    if "results" in st.session_state and st.session_state.results:
        results = st.session_state.results
        offset = st.session_state.get("offset", 0)
        limit = 5
        current_results = results[offset: offset + limit]
        st.write(f"Displaying entries {offset + 1} to {min(offset + limit, len(results))} of {len(results)}")

        # Instead of a selectbox, display each entry with a "Play" button.
        for entry in current_results:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"Style: {entry[2]}")
            with col2:
                st.write(f"BPM: {entry[3]}")
            with col3:
                st.write(f"TS: {entry[4]}")
            with col4:
                if st.button("Play", key=f"play_{entry[0]}"):
                    audio_path = entry[5]
                    if audio_path and os.path.exists(audio_path):
                        st.audio(audio_path)
                    else:
                        st.error("Audio file not found for entry " + entry[0])

        # Pagination control.
        if st.session_state.offset + limit < len(results):
            if st.button("Load More"):
                st.session_state.offset += limit
                try:
                    if hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()
                    else:
                        st.write("Please refresh the page to load more entries.")
                except Exception as e:
                    st.write("Please refresh the page to load more entries.")
        else:
            st.write("No more entries.")
    else:
        st.write("No query results to display yet.")


if __name__ == "__main__":
    main()
