import pandas as pd
import os
from tqdm import tqdm

# Load your CSV file
csv_file = "ar-dataset/validated.csv"
df = pd.read_csv(csv_file, dtype=str)

# Define output directory for data lists and text files
output_dir = "espnet_data"
txt_output_dir = "espnet_data/text"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(txt_output_dir, exist_ok=True)

# Create wav.scp, text, utt2spk, and individual sentence text files
with open(os.path.join(output_dir, 'wav.scp'), 'w') as wav_scp, \
     open(os.path.join(output_dir, 'text'), 'w') as text_file, \
     open(os.path.join(output_dir, 'utt2spk'), 'w') as utt2spk:

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        utterance_id = row['sentence_id']  # Unique ID for each utterance
        wav_path = row['path']             # Path to the audio file
        transcription = row['sentence']    # Transcription text
        speaker_id = row['client_id']      # Use client_id as the speaker ID

        # Write to wav.scp: <utterance_id> <path_to_audio_file>
        wav_scp.write(f"{utterance_id} {wav_path}\n")

        # Write to text: <utterance_id> <transcription>
        text_file.write(f"{utterance_id} {transcription}\n")

        # Write to utt2spk: <utterance_id> <speaker_id>
        utt2spk.write(f"{utterance_id} {speaker_id}\n")

        # Create individual txt file named after the audio file (e.g., voice.wav.txt)
        audio_filename = os.path.basename(wav_path)  # Extracts 'voice.wav' from the path
        txt_filename = os.path.splitext(audio_filename)[0]  # Removes the '.wav' extension
        
        # Full path to save the transcription as a txt file
        txt_filepath = os.path.join(txt_output_dir, f"{txt_filename}.txt")

        # Write the sentence to the corresponding txt file
        with open(txt_filepath, 'w') as sentence_file:
            sentence_file.write(transcription)

print("Data lists and sentence text files created successfully!")
