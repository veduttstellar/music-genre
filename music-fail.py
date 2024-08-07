import os
import glob

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization


def pad_or_truncate(feature, max_len):
    if feature.shape[1] < max_len:
        pad_width = max_len - feature.shape[1]
        feature = np.pad(feature, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        feature = feature[:, :max_len]
    return feature


def extract_features(file_path, max_len=130):
    try:
        audio, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        # zcr = librosa.feature.zero_crossing_rate(y=audio)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        rms = librosa.feature.rms(y=audio)
        # spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        # tempo = librosa.feature.rhythm.tempo(y=audio, sr=sr)  # Updated line

        # Pad or truncate features to ensure they have the same length
        mfccs = pad_or_truncate(mfccs, max_len)
        mel_spectrogram = pad_or_truncate(mel_spectrogram, max_len)
        # zcr = pad_or_truncate(zcr, max_len)
        spectral_contrast = pad_or_truncate(spectral_contrast, max_len)
        rms = pad_or_truncate(rms, max_len)
        # spectral_rolloff = pad_or_truncate(spectral_rolloff, max_len)
        # spectral_bandwidth = pad_or_truncate(spectral_bandwidth, max_len)

        num_rows = max(
            mfccs.shape[0],
            mel_spectrogram.shape[0],
            # zcr.shape[0],
            spectral_contrast.shape[0],
            rms.shape[0],
            # spectral_rolloff.shape[0],
            # spectral_bandwidth.shape[0],
        )
        mfccs = np.pad(mfccs, ((0, num_rows - mfccs.shape[0]), (0, 0)), mode="constant")
        mel_spectrogram = np.pad(
            mel_spectrogram,
            ((0, num_rows - mel_spectrogram.shape[0]), (0, 0)),
            mode="constant",
        )
        # zcr = np.pad(zcr, ((0, num_rows - zcr.shape[0]), (0, 0)), mode='constant')
        spectral_contrast = np.pad(
            spectral_contrast,
            ((0, num_rows - spectral_contrast.shape[0]), (0, 0)),
            mode="constant",
        )
        rms = np.pad(rms, ((0, num_rows - rms.shape[0]), (0, 0)), mode="constant")
        # spectral_rolloff = np.pad(spectral_rolloff, ((0, num_rows - spectral_rolloff.shape[0]), (0, 0)), mode='constant')
        # spectral_bandwidth = np.pad(spectral_bandwidth, ((0, num_rows - spectral_bandwidth.shape[0]), (0, 0)), mode='constant')

        # Stack features along the third dimension
        features = np.stack(
            [
                mfccs,
                mel_spectrogram,
                #  zcr,
                spectral_contrast,
                rms,
                #  spectral_rolloff,
                #  spectral_bandwidth
            ],
            axis=-1,
        )
        return features

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def load_data(directory, max_len=130):
    X, y = [], []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                genre = file_name.split(".")[0]  # Extract genre from the file name
                print(f"Processing file: {file_name}, Genre: {genre}")
                features = extract_features(file_path, max_len)
                if features is not None:
                    X.append(features)
                    y.append(genre)

    return np.array(X), np.array(y)


# Directory containing the music files
directory = "genres_original"

X, y = load_data(directory)
print(f"Loaded data shape: {X.shape}")
print(f"Loaded labels: {y}")

if len(set(y)) <= 1:
    raise ValueError("The number of classes has to be greater than one; got 1 class")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize the data
X = (X - np.mean(X)) / np.std(X)

# Split the data into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of the original data
)

# Build the CNN model
model = Sequential(
    [
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(X.shape[1], X.shape[2], X.shape[3]),
        ),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.6),
        Dense(len(set(y)), activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# from tensorflow.keras.callbacks import EarlyStopping
# 
#for reducing unnecessary epochs

# early_stopping = EarlyStopping(
#     monitor="val_loss", patience=5, restore_best_weights=True
# )

# Train the model
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    # callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
