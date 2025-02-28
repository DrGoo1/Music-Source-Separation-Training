import os
import sqlite3
import ijson
import streamlit as st


def create_database(db_path, json_path):
    """
    Create a SQLite database (if not exists) and load data from the JSON file incrementally.
    The table 'egmd' will have columns for base, midi_path, style, bpm, time_signature, and audio_path.
    """
    conn = sqlite3.connect(db_path)
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

    # Check if table is empty
    c.execute("SELECT COUNT(*) FROM egmd")
    count = c.fetchone()[0]
    if count > 0:
        conn.close()
        return  # Database already built

    # Incrementally parse the JSON and insert entries
    with open(json_path, "r", encoding="utf-8") as f:
        # The top-level JSON is assumed to be a dictionary mapping base -> entry.
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


def query_database_by_category(db_path, category):
    """
    Query the SQLite database for entries whose style (case-insensitive) starts with the given category.
    Returns a list of tuples.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    query = "SELECT base, midi_path, style, bpm, time_signature, audio_path FROM egmd WHERE LOWER(style) LIKE ?"
    param = category.lower() + "%";
    c.execute(query, (param,))
    results = c.fetchall()
    conn.close()
    return results


def main():
    st.title("E‑GMD Database Browser (SQLite)")
    st.write("""
        This tool loads a preprocessed E‑GMD database from a JSON file and stores it in a SQLite database.
        It then allows you to query the database by a style category.
        This method avoids loading the entire database into memory at once.
    """)

    json_path = st.text_input("Path to preprocessed JSON database:", value="egmd_combined_db.json")
    db_path = st.text_input("Path for SQLite database file:", value="egmd.db")

    if st.button("Build Database"):
        if not os.path.exists(json_path):
            st.error("JSON database file not found!")
            return
        with st.spinner("Building SQLite database (this may take a while)..."):
            create_database(db_path, json_path)
        st.success("Database built successfully!")

    category = st.text_input("Enter style category to query (e.g., funk, jazz):", value="funk")
    if st.button("Query Database"):
        results = query_database_by_category(db_path, category)
        if results:
            st.write(f"Found {len(results)} entries for category '{category}':")
            # Display as a table with selected columns.
            table_data = [{"Base": r[0], "Style": r[2], "BPM": r[3], "Time Signature": r[4]} for r in results]
            st.table(table_data)
            # Let the user select an entry to view details.
            bases = [r[0] for r in results]
            selected_base = st.selectbox("Select an entry for details:", bases)
            for r in results:
                if r[0] == selected_base:
                    selected_entry = {
                        "base": r[0],
                        "midi_path": r[1],
                        "style": r[2],
                        "bpm": r[3],
                        "time_signature": r[4],
                        "audio_path": r[5]
                    }
                    st.json(selected_entry)
                    # Play audio snippet if available.
                    if st.button("Play Audio Snippet"):
                        if selected_entry["audio_path"] and os.path.exists(selected_entry["audio_path"]):
                            st.audio(selected_entry["audio_path"])
                        else:
                            st.error("Audio file not found.")
                    break
        else:
            st.error("No entries found for that category.")


if __name__ == "__main__":
    main()
