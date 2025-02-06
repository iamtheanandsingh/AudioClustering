# Import libraries
import numpy as np
import pandas as pd
import librosa
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode
from art2 import ART2
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

# Define paths to audio files with labels for each class
audio_files = {
    'class_0': 'trimmed_crowd_talking.mp3',
    'class_1': 'trimmed_motor_riding.mp3',
    'class_2': 'trimmed_water_flowing.mp3'
}

# Function to load and process audio
def load_and_process_audio(file_path, frame_size_ms=30, overlap=0.1, n_mels=40, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=48000)
    frame_size = int(sr * frame_size_ms / 1000)
    hop_length = int(frame_size * (1 - overlap))
    mfsc = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mfsc = librosa.power_to_db(mfsc, ref=np.max).T
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T
    features = np.hstack((mfsc, mfcc))
    timestamps = librosa.frames_to_time(np.arange(features.shape[0]), sr=sr, hop_length=hop_length)
    return features, timestamps

# Function to generate synthetic time series
def generate_time_series_combinations(feature_files, num_series=5, min_segment=15, max_segment=30, total_duration=120):
    series_list = []
    class_ids = [0, 1, 2]
    for _ in range(num_series):
        current_time = 0
        segments = []
        while current_time < total_duration:
            selected_class = random.choice(class_ids)
            segment_duration = min(random.randint(min_segment, max_segment), total_duration - current_time)
            df = feature_files[selected_class]
            segment_df = df[df['timestamp'] <= segment_duration].copy()
            segment_df['timestamp'] += current_time
            segment_df['class_id'] = selected_class
            segments.append(segment_df)
            current_time += segment_duration
        combined_series = pd.concat(segments, ignore_index=True)
        series_list.append(combined_series)
    return series_list

# Function to map class IDs to sequential labels
def map_class_ids(true_labels):
    unique_classes = sorted(set(true_labels))
    class_map = {original: idx for idx, original in enumerate(unique_classes)}
    mapped_labels = np.array([class_map[label] for label in true_labels])
    return mapped_labels

def map_labels_to_true(predicted_labels, true_labels):
    labels = np.unique(predicted_labels)
    true_classes = np.unique(true_labels)
    cost_matrix = np.zeros((len(labels), len(true_classes)), dtype=int)
    for i, label in enumerate(labels):
        for j, cls in enumerate(true_classes):
            cost_matrix[i, j] = -np.sum((predicted_labels == label) & (true_labels == cls))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {labels[row]: true_classes[col] for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, -1) for label in predicted_labels])

# Function to smooth labels using majority voting
def smooth_labels(predicted_labels, window=5):
    smoothed_labels = np.copy(predicted_labels)
    for i in range(len(predicted_labels)):
        window_slice = predicted_labels[max(i - window // 2, 0):min(i + window // 2 + 1, len(predicted_labels))]
        smoothed_labels[i] = mode(window_slice, keepdims=True)[0][0]
    return smoothed_labels

# Run ART2 clustering
def run_art2_clustering(features, true_labels, vigilance_values, max_clusters):
    best_labels = None
    best_accuracy = 0
    for vigilance in vigilance_values:
        art2 = ART2(num_features=features.shape[1], vigilance=vigilance, learning_rate=0.1, max_epochs=10, max_clusters=max_clusters)
        art2.train(features)
        predicted_labels = art2.predict(features)
        predicted_labels_mapped = map_labels_to_true(predicted_labels, true_labels)
        smoothed_labels = smooth_labels(predicted_labels_mapped)
        accuracy = accuracy_score(true_labels, smoothed_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_labels = smoothed_labels
    return best_labels

# Evaluate clustering and apply PCA
def evaluate_clustering_with_pca(time_series_data, vigilance_values, max_clusters):
    results = []
    for idx, series_df in enumerate(time_series_data):
        features = series_df.iloc[:, :-2].values
        true_labels = map_class_ids(series_df['class_id'].values)
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
        pca = PCA(n_components=10)
        features_pca = pca.fit_transform(features)
        smoothed_labels = run_art2_clustering(features_pca, true_labels, vigilance_values, max_clusters)
        accuracy = accuracy_score(true_labels, smoothed_labels)
        conf_matrix = confusion_matrix(true_labels, smoothed_labels)
        
        print(f"Time Series {idx + 1}:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix, "\n")

        results.append({
            'time_series': idx + 1,
            'accuracy': accuracy,
            'conf_matrix': conf_matrix,
            'timestamps': series_df['timestamp'].values,
            'true_labels': true_labels,
            'smoothed_labels': smoothed_labels,
        })
    return results

# Plot Gantt charts
def plot_discrete_gantt(clustering_results, save_path="gantt_chart"):
    for result in clustering_results:
        timestamps = result['timestamps']
        true_labels = result['true_labels']
        smoothed_labels = result['smoothed_labels']
        fig, ax = plt.subplots(1, figsize=(15, 5))
        ax.step(timestamps, true_labels, where='post', label="True", color="black", alpha=0.6)
        ax.step(timestamps, smoothed_labels, where='post', label="Predicted", color="blue", alpha=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Class")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}_series_{result['time_series']}.png")
        plt.close(fig)

# Main Execution
feature_files = []
for class_id in audio_files.keys():
    features, timestamps = load_and_process_audio(audio_files[class_id])
    df = pd.DataFrame(features, columns=[f'Feature_{i}' for i in range(features.shape[1])])
    df['timestamp'] = timestamps
    df['class_id'] = int(class_id[-1])
    feature_files.append(df)
    df.to_csv(f"singh_progress_class_{class_id}.csv", index=False)

time_series_data = generate_time_series_combinations(feature_files)
clustering_results = evaluate_clustering_with_pca(time_series_data, vigilance_values=[0.5, 0.6, 0.7], max_clusters=4)
plot_discrete_gantt(clustering_results)
