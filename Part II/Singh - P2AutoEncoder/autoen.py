import numpy as np
import pandas as pd
import librosa
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# Paths to audio files labeled by class
audio_files = {
    'class_0': 'trimmed_crowd_talking.mp3',
    'class_1': 'trimmed_motor_riding.mp3',
    'class_2': 'trimmed_water_flowing.mp3'
}

# Function to load an audio file and extract MFSC and MFCC features
def load_and_process_audio(file_path, frame_size_ms=30, overlap=0.1, n_mels=40, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=48000)
    frame_size = int(sr * frame_size_ms / 1000)
    hop_length = int(frame_size * (1 - overlap))
    
    # Extract MFSC and MFCC features
    mfsc = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mfsc = librosa.power_to_db(mfsc, ref=np.max).T
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length).T

    # Combine MFSC and MFCC features horizontally
    features = np.hstack((mfsc, mfcc))
    timestamps = librosa.frames_to_time(np.arange(features.shape[0]), sr=sr, hop_length=hop_length)
    
    return features, timestamps

# Function to generate synthetic time series data
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

# Function to smooth predictions using a sliding window
def smooth_labels(predicted_labels, window=5):
    smoothed_labels = np.copy(predicted_labels)
    for i in range(0, len(predicted_labels), window):
        segment = predicted_labels[i:i+window]
        majority_label = mode(segment, keepdims=True)[0][0]
        smoothed_labels[i:i+window] = majority_label
    return smoothed_labels

# Build and train Auto-Encoder
def train_autoencoder(features, encoding_dim):
    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    input_dim = features_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    # Decoder
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Auto-Encoder Model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    # Encoder Model
    encoder = Model(inputs=input_layer, outputs=encoded)
    
    # Compile and train
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    autoencoder.fit(features_scaled, features_scaled, epochs=50, batch_size=256, shuffle=True, verbose=0)
    
    # Get encoded features
    features_encoded = encoder.predict(features_scaled)
    return features_encoded, scaler

# Map predicted labels to true labels
def best_map(true_labels, pred_labels):
    D = max(true_labels.max(), pred_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(true_labels.size):
        w[true_labels[i], pred_labels[i]] += 1
    ind_row, ind_col = linear_sum_assignment(-w)
    mapping = {col: row for row, col in zip(ind_row, ind_col)}
    mapped_labels = np.array([mapping[label] for label in pred_labels])
    return mapped_labels

# Evaluate clustering performance using Auto-Encoders
def evaluate_clustering_with_autoencoder_and_print(time_series_data, encoding_dim=20):
    results = []
    
    for idx, series_df in enumerate(time_series_data):
        features = series_df.iloc[:, :-2].values
        true_labels = series_df['class_id'].values
        
        # Train Auto-Encoder
        features_encoded, scaler = train_autoencoder(features, encoding_dim)
        
        # Clustering using K-Means
        kmeans = KMeans(n_clusters=5, random_state=42)
        predicted_labels = kmeans.fit_predict(features_encoded)
        
        # Smoothing
        smoothed_labels = smooth_labels(predicted_labels, window=10)
        
        # Map labels
        mapped_labels = best_map(true_labels, smoothed_labels)
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, mapped_labels)
        conf_matrix = confusion_matrix(true_labels, mapped_labels)
        
        # Print results
        print(f"Time Series {idx + 1}:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(conf_matrix, "\n")
        
        results.append({
            'time_series': idx + 1,
            'true_labels': true_labels,
            'timestamps': series_df['timestamp'].values,
            'predicted_labels': mapped_labels,
            'accuracy': accuracy,
            'conf_matrix': conf_matrix
        })
        
    return results

# Plot Gantt-style visualization of clustering results
def plot_discrete_gantt(clustering_results):
    for result in clustering_results:
        timestamps = result['timestamps']
        true_labels = result['true_labels']
        predicted_labels = result['predicted_labels']
        time_series_number = result['time_series']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # True label visualization
        ax1.step(timestamps, true_labels, where='post', color="black", linewidth=5, alpha=0.3)
        ax1.set_title(f"Time Series {time_series_number} - True Clusters")
        ax1.set_ylabel("Cluster")
        
        # Predicted clusters
        ax2.step(timestamps, predicted_labels, where='post', color="green", linewidth=5, alpha=0.6)
        ax2.set_title(f"Time Series {time_series_number} - Predicted Clusters")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Cluster")
        
        plt.tight_layout()
        # Save the plot to a file
        plt.savefig(f"time_series_{time_series_number}.png")
        plt.close(fig)  # Close the figure to free memory

# Load and process features for each audio file
feature_files = []
for class_id in audio_files.keys():
    features, timestamps = load_and_process_audio(audio_files[class_id])
    df = pd.DataFrame(features, columns=[f'MFSC_{i}' for i in range(40)] + [f'MFCC_{j}' for j in range(13)])
    df['timestamp'] = timestamps
    df['class_id'] = int(class_id[-1])
    feature_files.append(df)
    
    # Save each class's feature data to a CSV file
    df.to_csv(f"class_{class_id}.csv", index=False)

# Generate synthetic time series data
time_series_data = generate_time_series_combinations(feature_files)

# Evaluate clustering and display results
clustering_results = evaluate_clustering_with_autoencoder_and_print(time_series_data, encoding_dim=20)

# Plot results for each time series
plot_discrete_gantt(clustering_results)
