import os
import joblib
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm  # For progress bar
from tensorflow.keras import layers, Input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv

load_dotenv()

DIR_PATH = os.getenv('DIR_PATH')
MODEL_NAME = os.getenv('MODEL_NAME')
EXTRACTED_DATA_NAME = os.getenv('EXTRACTED_DATA_NAME')
LABEL_ENCODER_NAME = os.getenv('LABEL_ENCODER_NAME')
DATA_FILE_PATH = os.getenv('DATA_FILE_PATH')
FEATURES_NAME = os.getenv('FEATURES_NAME')
LABELS_NAME = os.getenv('LABELS_NAME')

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

if not os.path.exists(f"{DIR_PATH}"):
    os.makedirs(f"{DIR_PATH}")

# Load or extract features
if os.path.exists(f'{DIR_PATH}/{EXTRACTED_DATA_NAME}.npz') and os.path.exists(f'{DIR_PATH}/{LABEL_ENCODER_NAME}.pkl'):
    print("Loading pre-extracted features and labels from disk...")
    
    # Load pre-extracted features and labels
    data = np.load(f'{DIR_PATH}/{EXTRACTED_DATA_NAME}.npz')
    features = data['features']
    labels = data['labels']
    
    # Load label encoder
    label_encoder = joblib.load(f'{DIR_PATH}/{LABEL_ENCODER_NAME}.pkl')
    labels_encoded = label_encoder.transform(labels)
    
else:
    print("Extracting features (this may take a while)...")
    
    file_extension = os.path.splitext(DATA_FILE_PATH)[1].lower()
    
    # Load dataset
    if file_extension == '.csv':
        data = pd.read_csv(f'{DATA_FILE_PATH}')
    elif file_extension == '.json':
        data = pd.read_json(f'{DATA_FILE_PATH}')
    else:
        raise ValueError("Invalid data file format. Please provide a CSV or JSON file.")
    # data = pd.read_csv(f'{DATA_FILE_PATH}') 
    features, labels = [], []
    
    # Extract features with a progress bar
    for index, row in tqdm(data.iterrows(), total=len(data)):
        features.append(extract_features(row[f'{FEATURES_NAME}']))
        labels.append(row[f'{LABELS_NAME}'])
    
    # Convert features to NumPy array
    features = np.array(features)
    
    # Encode the labels into integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Save extracted features and label encoder to disk
    np.savez_compressed(f'{DIR_PATH}/{EXTRACTED_DATA_NAME}.npz', features=features, labels=labels)
    joblib.dump(label_encoder, f'{DIR_PATH}/{LABEL_ENCODER_NAME}.pkl')
    
    print("Features and labels saved.")

# Split data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define positional encoding layer
class PositionalEncoding(layers.Layer):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.get_positional_encoding(max_len, embed_dim)
    
    def get_positional_encoding(self, max_len, embed_dim):
        angle_rads = self.get_angles(np.arange(max_len)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]
    
# Define Transformer Block
def transformer_block(x, embed_dim, num_heads, ff_dim, rate=0.1):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attention_output = layers.Dropout(rate)(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)  # Skip connection
    
    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = layers.Dropout(rate)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)  # Skip connection

# Transformer Model Using the Keras Transformer module
input_shape = (100, 39)
embed_dim = 128
num_heads = 4
ff_dim = 512
num_transformer_blocks = 4

inputs = Input(shape=input_shape)

# Apply CNN layers first to process the features
x = layers.Conv1D(128, kernel_size=3, strides=1, activation='relu', padding='same')(inputs)
x = layers.MaxPooling1D(pool_size=2)(x)

x = layers.Conv1D(128, kernel_size=3, strides=1, activation='relu', padding='same')(x)
x = layers.MaxPooling1D(pool_size=2)(x)

x = layers.Conv1D(64, kernel_size=3, strides=1, activation='relu', padding='same')(x)
x = layers.MaxPooling1D(pool_size=2)(x)

# Add Positional Encoding to the features
x = PositionalEncoding(embed_dim=64)(x)

# Use Transformer Blocks from Keras (tf.keras.layers.Transformer)
for _ in range(num_transformer_blocks):
    x = transformer_block(x, embed_dim=64, num_heads=4, ff_dim=ff_dim)

# Global pooling and final classification
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(len(label_encoder.classes_), activation='softmax')(x)

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model with optimizer and learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model with early stopping and learning rate scheduling
history = model.fit(np.array(X_train), np.array(y_train), epochs=50, validation_data=(np.array(X_val), np.array(y_val)), callbacks=[early_stopping])

# Evaluate on test set
test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test))
print(f'Test accuracy: {test_acc}')

# Save model and label encoder
model.save(f'{DIR_PATH}/{MODEL_NAME}.keras')