import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score





# Directory containing the music files
directory = "genres_original"

X, y = load_data(directory)

# print(np.array(X), np.array(y))

print(f"Loaded data shape: {X.shape}")
print(f"Loaded labels: {y}")

if len(set(y)) <= 1:
    raise ValueError("The number of classes has to be greater than one; got 1 class")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set

y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")






def load_data(directory):
    X, y = [], []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                genre = file_name.split(".")[0]  # Extract genre from the file name
                print(f"Processing file: {file_name}, Genre: {genre}")
                try:
                    audio, sr = librosa.load(file_path)
                    print(audio, sr)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    mfccs = np.mean(mfccs.T, axis=0)
                    X.append(mfccs)
                    y.append(genre)
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
            # print()      
    return np.array(X), np.array(y)




