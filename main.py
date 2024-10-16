import tensorflow as tf
import librosa
import numpy as np
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

DIR_PATH = os.getenv('DIR_PATH')
MODEL_NAME = os.getenv('MODEL_NAME')
LABEL_ENCODER_NAME = os.getenv('LABEL_ENCODER_NAME')

# Load the model
model = tf.keras.models.load_model(f'{DIR_PATH}/{MODEL_NAME}.keras')

# Load the label encoder
label_encoder = joblib.load(f'{DIR_PATH}/{LABEL_ENCODER_NAME}.pkl')

# Function to extract MFCC features
def extract_features(file_path, fixed_length=100):
    audio, sample_rate = librosa.load(file_path, sr=16000)  
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)  
    delta_mfcc = librosa.feature.delta(mfccs)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    features = np.concatenate((mfccs, delta_mfcc, delta2_mfcc), axis=0)  # Combine MFCC and its deltas
    
    # Transpose and pad to fixed length
    features = features.T
    if features.shape[0] < fixed_length:
        features = np.pad(features, ((0, fixed_length - features.shape[0]), (0, 0)), mode='constant')
    else:
        features = features[:fixed_length, :]
    
    return features

# Function for predicting transcription
def predict_audio(file_path):
    features = extract_features(file_path)
    prediction = model.predict(np.array([features]))  # Reshape to (1, 100, 13)
    predicted_label = np.argmax(prediction, axis=1)  # Get the predicted character index
    return label_encoder.inverse_transform(predicted_label)  # Decode back to original transcription

# Example usage
file_path = 'dataset/arabic-ghaizer/asr/data/test/test_record_010.wav'  # Replace with the path to your audio file
predicted_transcription = predict_audio(file_path)
print(f'Predicted Transcription: {predicted_transcription}')