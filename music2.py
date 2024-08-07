import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def extract_features(input_data, sr=None):
    try:
        if isinstance(input_data, str):
            # Load the audio file
            audio, sr = librosa.load(input_data)
        else:
            # Use the provided audio array and sample rate
            audio = input_data
        
        # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        # Extract Chroma features
        # chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # Extract Mel-scaled spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        
        # Extract Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        # Extract Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        
        # Extract Spectral Roll-off
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        # Combine all features into a single feature vector
        features = np.hstack([
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(zcr.T, axis=0),
            np.mean(rolloff.T, axis=0)
        ])
        
        return features
    except Exception as e:
        print(f"Error processing file {input_data}: {e}")
        return None

def augment_audio(audio, sr):
    # Pitch shifting
    pitch_shifted = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=4)
    
    # Time stretching
    time_stretched = librosa.effects.time_stretch(y=audio, rate=1.25)
    
    # Adding noise
    noise = np.random.randn(len(audio))
    audio_with_noise = audio + 0.005 * noise
    
    return [pitch_shifted, time_stretched, audio_with_noise]

def load_data(directory):
    X, y = [], []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                genre = os.path.basename(root)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(genre)
                    
                    # Data augmentation
                    audio, sr = librosa.load(file_path)
                    augmented_audios = augment_audio(audio, sr)
                    for augmented_audio in augmented_audios:
                        augmented_features = extract_features(augmented_audio, sr)
                        if augmented_features is not None:
                            X.append(augmented_features)
                            y.append(genre)
    return np.array(X), np.array(y)

# Directory containing the music files
directory = 'genres_original'

X, y = load_data(directory)
print(f"Loaded data shape: {X.shape}")
print(f"Loaded labels: {y}")

# Check if there is more than one class
if len(set(y)) <= 1:
    raise ValueError("The number of classes has to be greater than one; got 1 class")

# Print a few samples of the features and labels
print("Sample features:", X[:5])
print("Sample labels:", y[:5])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define the parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Initialize Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)

# Predict on the test set using the best estimator
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")