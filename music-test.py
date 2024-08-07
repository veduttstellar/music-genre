import os
import glob

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler







def extract_features(file_path):
	try:
		audio, sr = librosa.load(file_path)
		mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
		# chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
		mel = librosa.feature.melspectrogram(y=audio, sr=sr)
		contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

		max_len = 100  # Define the maximum length
		num_rows = max(mfccs.shape[0], mel.shape[0], contrast.shape[0])  # Calculate the maximum number of rows

		# mfccs = pad_or_truncate(mfccs, max_len)
		# mel = pad_or_truncate(mel, max_len)
		# # chroma = pad_or_truncate(chroma, max_len)
		# contrast = pad_or_truncate(contrast, max_len)
		# s = pad_or_truncate(rms, max_len)

		# num_rows = max(mfccs.shape[0], mel_spectrogram.shape[0], zcr.shape[0], spectral_contrast.shape[0], rms.shape[0], spectral_rolloff.shape[0], spectral_bandwidth.shape[0])
		mfccs = np.pad(mfccs, ((0, num_rows - mfccs.shape[0]), (0, 0)), mode='constant')
		mel = np.pad(mel, ((0, num_rows - mel.shape[0]), (0, 0)), mode='constant')
		# zcr = np.pad(zcr, ((0, num_rows - zcr.shape[0]), (0, 0)), mode='constant')
		contrast = np.pad(contrast, ((0, num_rows - contrast.shape[0]), (0, 0)), mode='constant')
		# rms = np.pad(rms, ((0, num_rows - rms.shape[0]), (0, 0)), mode='constant')
		# spectral_rolloff = np.pad(spectral_rolloff, ((0, num_rows - spectral_rolloff.shape[0]), (0, 0)), mode='constant')
		# spectral_bandwidth = np.pad(spectral_bandwidth, ((0, num_rows - spectral_bandwidth.shape[0]), (0, 0)), mode='constant')

		features = np.hstack([
			np.mean(mfccs.T, axis=0),
			np.mean(mel.T, axis=0),
			# np.mean(chroma.T, axis=0),
			np.mean(contrast.T, axis=0)
		])
		return features

	except Exception as e:
		print(f"Error processing file {file_path}: {e}")
		return None




# def pad_or_truncate(feature, max_len):
#     if feature.shape[1] > max_len:
#         return feature[:, :max_len]
#     else:
#         pad_width = max_len - feature.shape[1]
#         return np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
# 


def load_data(directory):
	X, y = [], []
	for root, _, files in os.walk(directory):
		for file_name in files:
			if file_name.endswith(".wav"):
				file_path = os.path.join(root, file_name)
				genre = file_name.split(".")[0]  # Extract genre from the file name
				print(f"Processing file: {file_name}, Genre: {genre}")
				features = extract_features(file_path)
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

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning for RandomForestClassifier
param_grid = {
	'n_estimators': [100, 200, 300],
	'max_depth': [None, 10, 20, 30],
	'min_samples_split': [2, 5, 10],
	'min_samples_leaf': [1, 2, 4]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

# Predict on the test set
y_pred = best_clf.predict(X_test)

# Calculate accuracyis th
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")