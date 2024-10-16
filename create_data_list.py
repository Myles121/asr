import pandas as pd
import os

# Load the transcriptions
df = pd.read_csv("espnet_data/data/transcriptions.csv")

# Create data directories
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/dev", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# Create wav.scp, text, and utt2spk
with open("data/train/wav.scp", "w") as wav_scp, \
     open("data/train/text", "w") as text_file, \
     open("data/train/utt2spk", "w") as utt2spk:

    for index, row in df.iterrows():
        utterance_id = row['sentence_id']
        wav_path = row['path']  # Adjust this if necessary
        transcription = row['sentence']
        speaker_id = row['client_id']

        # Write to wav.scp
        wav_scp.write(f"{utterance_id} {wav_path}\n")
        # Write to text
        text_file.write(f"{utterance_id} {transcription}\n")
        # Write to utt2spk
        utt2spk.write(f"{utterance_id} {speaker_id}\n")
