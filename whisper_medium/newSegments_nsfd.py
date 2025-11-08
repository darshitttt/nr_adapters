# %%
import os
import librosa
import pandas as pd
import numpy as np

# %%
dff = pd.read_csv('/ceph/dpandya/notsofar/train_set/240825.1_train/train.csv')

# %%
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm

def segment_multiple_audios(audio_files, csv_files, output_dir, output_csv, min_duration=10, max_duration=20):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the final DataFrame for all metadata
    all_segments = []

    # Process each audio and its corresponding CSV file
    for audio_file, csv_file in zip(audio_files, csv_files):
        # Load the audio file
        audio = AudioSegment.from_file(audio_file)

        # Load the CSV file
        df = pd.read_json(csv_file)

        # Initialize variables
        segments = []
        current_text = ""
        current_start = None
        segment_start = None
        segment_duration = 0

        # Iterate through the rows of the CSV
        for _, row in df.iterrows():
            start = row['start_time'] * 1000  # Convert to milliseconds
            stop = row['end_time'] * 1000   # Convert to milliseconds
            text = row['text']
            duration = stop - start
            print(duration/1000, start/1000, stop/1000)
            if current_start is None:
                # Initialize the first segment
                current_start = start
                segment_start = start
                current_text = text
                segment_duration = duration
            else:
                # Check if adding this segment keeps duration within limits
                if segment_duration + duration <= max_duration * 1000:
                    segment_duration += duration
                    current_text += " " + text
                else:
                    # Finalize the current segment
                    segments.append((segment_start, current_start + segment_duration))
                    all_segments.append({
                        "original_audio_file": os.path.basename(audio_file),
                        "segmented_audio_file": f"{os.path.splitext(os.path.basename(audio_file))[0]}_segment_{len(segments)}.wav",
                        "segmented_text": current_text.strip()
                    })

                    # Start a new segment
                    current_start = start
                    segment_start = start
                    current_text = text
                    segment_duration = duration

                # If segment is too short, combine it
                if segment_duration < min_duration * 1000:
                    continue
            print(segments)

        # Save the last segment if it hasn't been added
        if segment_duration >= min_duration * 1000:
            segments.append((segment_start, segment_start + segment_duration))
            all_segments.append({
                "original_audio_file": os.path.basename(audio_file),
                "segmented_audio_file": f"{os.path.splitext(os.path.basename(audio_file))[0]}_segment_{len(segments)}.wav",
                "segmented_text": current_text.strip()
            })

        # Save the audio segments
        for i, (start, stop) in enumerate(segments):
            segment_filename = all_segments[-len(segments) + i]['segmented_audio_file']
            segment = audio[start:stop]
            #segment.export(os.path.join(output_dir, segment_filename), format="wav")

    # Save the consolidated CSV file
    #final_df = pd.DataFrame(all_segments)
    #final_df.to_csv(output_csv, index=False)

    print(f"Processed {len(all_segments)} segments across all files.")
    print(f"All segments saved in: {output_dir}")
    print(f"Consolidated metadata saved in: {output_csv}")


# Example usage
'''audio_files = ["path/to/audio1.wav", "path/to/audio2.wav", "path/to/audio3.wav"]
csv_files = ["path/to/transcription1.csv", "path/to/transcription2.csv", "path/to/transcription3.csv"]
output_directory = "output_segments"
consolidated_csv = "all_segments_metadata.csv"'''

# %%
audio_files = list(dff['sc_files'])[0:1]
json_files = list(dff['gt_transcription_files'])[0:1]
output_dir = '/ceph/dpandya/notsofar/dev_set/240825.1_dev1/new_segmented_audios/'
consolidated_csv = '/ceph/dpandya/notsofar/dev_set/240825.1_dev1/new_segmented_audios/new_segmented_audios.csv'
segment_multiple_audios(audio_files, json_files, output_dir, consolidated_csv)


