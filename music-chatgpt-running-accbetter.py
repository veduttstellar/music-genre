import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def extract_features(file_name, n_mfcc=20):
    print(f"Extracting features from {file_name}...")
    try:
        y, sr = librosa.load(file_name, mono=True)
        print(f"Loaded {file_name} successfully.")

        # Extract features
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Print shapes of features
        print(f"RMSE shape: {rmse.shape}")
        print(f"Spectral Centroid shape: {spec_cent.shape}")
        print(f"Spectral Bandwidth shape: {spec_bw.shape}")
        print(f"Spectral Rolloff shape: {rolloff.shape}")
        print(f"Zero Crossing Rate shape: {zcr.shape}")
        print(f"MFCC shape: {mfcc.shape}")

        # Mean of features
        rmse_mean = np.mean(rmse)
        spec_cent_mean = np.mean(spec_cent)
        spec_bw_mean = np.mean(spec_bw)
        rolloff_mean = np.mean(rolloff)
        zcr_mean = np.mean(zcr)

        mfcc_means = [np.mean(mfcc[i]) for i in range(n_mfcc)]
        if len(mfcc_means) < n_mfcc:
            mfcc_means.extend([0] * (n_mfcc - len(mfcc_means)))  # Padding if needed

        features = np.array(
            [rmse_mean, spec_cent_mean, spec_bw_mean, rolloff_mean, zcr_mean]
            + mfcc_means
        )

        print(f"Feature vector shape: {features.shape}")
        return features

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None


# Main script
dataset_dir = "genres_original"
features_list = []
labels_list = []

print("Starting to process dataset...")

if not os.path.isdir(dataset_dir):
    print(f"Directory {dataset_dir} does not exist.")
else:
    for root, _, files in os.walk(dataset_dir):
        if files:
            genre = os.path.basename(root)
            print(f"Processing genre: {genre}")
            for file in files:
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                features = extract_features(file_path)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(genre)
                else:
                    print(f"Failed to extract features from: {file_path}")

    if len(features_list) == 0:
        print(
            "No features were extracted. Please check the dataset directory and file paths."
        )
    else:
        columns = ["rmse", "spec_cent", "spec_bw", "rolloff", "zcr"] + [
            f"mfcc{i}" for i in range(1, 21)
        ]
        features_df = pd.DataFrame(features_list, columns=columns)
        features_df["label"] = labels_list

        print(f"Extracted features from {len(features_df)} files.")
        print("Sample features:")
        print(features_df.head())

        X = features_df.iloc[:, :-1].astype(float)
        y = features_df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)

        # Define parameter grid for hyperparameter tuning
        param_grid = {
            "n_estimators": list(
                range(100, 1001, 50)
            ),  # Adding values from 100 to 1000 with a step of 50
            "max_depth": [None]
            + list(
                range(10, 101, 10)
            ),  # Adding values from 10 to 100 with a step of 10
            "min_samples_split": [
                2,
                5,
                10,
                20,
            ],  # Adding more values for min_samples_split
            "min_samples_leaf": [
                1,
                2,
                4,
                10,
            ],  # Adding more values for min_samples_leaf
            "bootstrap": [True, False],  # Adding bootstrap parameter
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
        )

        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)

        # Get best parameters and best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print("Best parameters found: ", best_params)
        print("Best score found: ", best_score)

        # Use the best model to make predictions
        best_clf = grid_search.best_estimator_

        y_pred = best_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy * 100:.2f}%")
