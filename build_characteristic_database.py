# build_characteristic_database.py
import os
import pandas as pd
import pretty_midi
import numpy as np


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


def extract_drum_stats(pm, start_time, end_time):
    """
    Extract basic statistics from the drum notes within a time window.
    Returns mean and standard deviation of onset deviation (with a 0.25 sec grid),
    mean and standard deviation of note velocity, and note count.
    """
    onset_devs = []
    velocities = []
    quantization = 0.25
    for inst in pm.instruments:
        if inst.is_drum:
            for note in inst.notes:
                # Only consider notes in the time window.
                if start_time <= note.start < end_time:
                    quantized_time = round(note.start / quantization) * quantization
                    onset_devs.append(note.start - quantized_time)
                    velocities.append(note.velocity)
    if onset_devs:
        return {
            "mean_onset_dev": np.mean(onset_devs),
            "std_onset_dev": np.std(onset_devs),
            "mean_velocity": np.mean(velocities),
            "std_velocity": np.std(velocities),
            "num_notes": len(velocities)
        }
    else:
        return {
            "mean_onset_dev": None,
            "std_onset_dev": None,
            "mean_velocity": None,
            "std_velocity": None,
            "num_notes": 0
        }


def segment_midi(pm):
    """
    Segment a MIDI drum track into estimated arrangement parts.
    This heuristic divides the track into fixed portions:
      - For tracks longer than 60 seconds, we define:
          intro: first 20%
          verse: next 30%
          chorus: next 20%
          breakdown: next 10%
          verse2: next 10%
          outro: final 10%
      - For shorter tracks, we simply divide into three equal segments.
    Returns a list of tuples (segment_label, start_time, end_time).
    """
    total_duration = pm.get_end_time()
    segments = []
    if total_duration > 60:
        intro_end = 0.2 * total_duration
        verse_end = intro_end + 0.3 * total_duration
        chorus_end = verse_end + 0.2 * total_duration
        breakdown_end = chorus_end + 0.1 * total_duration
        verse2_end = breakdown_end + 0.1 * total_duration
        segments = [
            ("intro", 0, intro_end),
            ("verse", intro_end, verse_end),
            ("chorus", verse_end, chorus_end),
            ("breakdown", chorus_end, breakdown_end),
            ("verse2", breakdown_end, verse2_end),
            ("outro", verse2_end, total_duration)
        ]
    else:
        num_segments = 3
        segment_length = total_duration / num_segments
        segments = [("segment_" + str(i + 1), i * segment_length, (i + 1) * segment_length)
                    for i in range(num_segments)]
    return segments


def main():
    # Update these paths as needed.
    # For example, pick one representative MIDI file from E-GMD:
    midi_file = r"G:\E-GMD Dataset\e-gmd-v1.0.0-midi\e-gmd-v1.0.0\drummer1\eval_session\1_funk-groove1_138_beat_4-4_1.midi"  # Change to an actual file path
    # Alternatively, you could loop over multiple files.

    # Load the MIDI file.
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        print(f"Loaded MIDI: {midi_file}")
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return

    total_duration = pm.get_end_time()
    print(f"Total duration: {total_duration:.2f} seconds")

    # Segment the MIDI into arrangement parts.
    segments = segment_midi(pm)
    print("Identified segments:")
    for seg in segments:
        print(seg)

    # Analyze each segment.
    results = []
    for label, start, end in segments:
        stats = extract_drum_stats(pm, start, end)
        # Also compute a groove metric for the segment (by filtering the notes)
        # We create a temporary PrettyMIDI object that only contains notes in the segment.
        segment_pm = pretty_midi.PrettyMIDI()
        drum_instr = pretty_midi.Instrument(program=0, is_drum=True)
        for inst in pm.instruments:
            if inst.is_drum:
                for note in inst.notes:
                    if start <= note.start < end:
                        drum_instr.notes.append(note)
        segment_pm.instruments.append(drum_instr)
        groove_metric = compute_groove_metric(segment_pm)
        result = {
            "segment": label,
            "start_time": start,
            "end_time": end,
            "groove_metric": groove_metric,
            "mean_onset_dev": stats["mean_onset_dev"],
            "std_onset_dev": stats["std_onset_dev"],
            "mean_velocity": stats["mean_velocity"],
            "std_velocity": stats["std_velocity"],
            "num_notes": stats["num_notes"]
        }
        results.append(result)
        print(f"Segment {label}: groove_metric={groove_metric}, num_notes={stats['num_notes']}")

    # Save the analysis data to a CSV file.
    df = pd.DataFrame(results)
    output_csv = "egmd_characteristics_database.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nAnalysis data saved to {output_csv}")


if __name__ == '__main__':
    main()
