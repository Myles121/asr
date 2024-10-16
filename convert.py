from pydub import AudioSegment
import os
from tqdm import tqdm

def convert_audio_to_wav(input_folder, output_folder):
    """
    Convert all audio files in the input folder to mono, 16kHz WAV format and save in the output folder.
    Also deletes the original input file after processing or skipping.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    files = os.listdir(input_folder)
    for audio_file in tqdm(files, total=len(files)):
    # for audio_file in files:
        audio_path = os.path.join(input_folder, audio_file)
        
        # Check if the file is a valid audio file
        if os.path.isfile(audio_path):
            file_extension = os.path.splitext(audio_file)[1].lower()
            
            # Construct the output file path
            output_file_name = os.path.splitext(audio_file)[0] + ".wav"
            output_path = os.path.join(output_folder, output_file_name)

            # Check if the output file already exists
            if os.path.exists(output_path):
                print(f"Output file already exists, skipping: {output_path}")
                os.remove(audio_path)
                continue  # Skip to the next file
            
            try:
                # Load the audio file with pydub
                audio = AudioSegment.from_file(audio_path)
                
                # Check if the audio needs conversion (to WAV, mono, 16kHz)
                if file_extension != ".wav" or audio.channels != 1 or audio.frame_rate != 16000:
                    # Convert to mono and set frame rate to 16kHz
                    audio = audio.set_channels(1).set_frame_rate(16000)
                    
                    # Export the converted file to WAV format
                    audio.export(output_path, format="wav")
                    # print(f"Converted audio saved to {output_path}")
                    
                    # Remove the original file
                    os.remove(audio_path)
                else:
                    print(f"Audio is already in the correct format: {audio_path}")
                    os.remove(audio_path)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

# Example usage
input_folder = "ar-dataset/clips"  # Path to your folder containing audio files
output_folder = "ar-dataset/clips/converted"  # Path to save converted audio files
convert_audio_to_wav(input_folder, output_folder)