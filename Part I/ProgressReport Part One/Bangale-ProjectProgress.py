import numpy as np
import librosa
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


# Loads and processes audio file to extract MFSC and MFCC features
# Returns the processed features and corresponding timestamps
def load_and_process_audio(file_path, frame_size_ms, overlap=0.1):
    y, sr = librosa.load(file_path)
    frame_size = int(sr * frame_size_ms / 1000)
    hop_length = int(frame_size * (1 - overlap))
    mfsc = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfsc = mfsc.T
    mfcc = mfcc.T
    features = np.hstack((mfsc, mfcc))
    timestamps = librosa.frames_to_time(np.arange(1,features.shape[0]+1), sr=sr, hop_length=hop_length)
    return mfsc, mfcc, timestamps

# Writes extracted features to CSV files and returns combined features and timestamps
def write_to_csv(mfsc, mfcc, timestamps, class_id):
    df = pd.DataFrame()
    comb_feat = []
    comb_times = []
    for i, (MFSC, MFCC, timestamp) in enumerate(zip(mfsc, mfcc, timestamps)):
        new_row = {'class_id': class_id, 'timestamp': timestamp, 'MFSC': MFSC, 'MFCC': MFCC}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        comb_feat.append(np.hstack((MFSC, MFCC)))
        comb_times.append(timestamp)
    df.to_csv(f'Feature_{class_id}.csv', index=None)
    return comb_feat, comb_times

# Generates multiple time series combinations ensuring all classes are used
# Creates 2-minute sequences with random segment durations
def generate_time_series_combinations(fn_features, class_ids, num_combinations=5):
    combined_series = []
    for combo in range(num_combinations):
        series = []
        current_time = 0
        remaining_classes = class_ids.copy()
        while current_time < 120:
            if remaining_classes:
                selected_class = random.choice(remaining_classes)
                remaining_classes.remove(selected_class)
            else:
                selected_class = random.choice(class_ids)
            segment_duration = random.randint(15, 30)
            if current_time + segment_duration > 120:
                segment_duration = 120 - current_time
            df = fn_features[selected_class]
            segment = df[df['timestamp'] <= segment_duration].copy()
            segment['timestamp'] = segment['timestamp'] + current_time
            segment['class_id'] = selected_class
            series.append(segment)
            current_time += segment_duration
            remaining_time = 120 - current_time
            if remaining_time < 45 and remaining_classes:
                segment_duration = remaining_time / len(remaining_classes)
        combined_series.append(pd.concat(series, ignore_index=True))
    return combined_series

# Adds timestamps to series based on frame size
def add_timestamps(series):
    series['timestamp'] = series.index * (frm_sz / 1000)
    return series

# Implements ART-2 clustering algorithm on time series data
def run_art2_clustering(time_series, vigilance_threshold, creation_buffer_size, max_clusters=3):
    predictions = []
    num_clusters = 0
    cluster_weights = []
    cluster_creation_buffer = 0
    for sample in time_series:
        cluster_id, cluster_weights, num_clusters, cluster_creation_buffer = process_new_sample(
            sample, cluster_weights, num_clusters, cluster_creation_buffer, 
            vigilance_threshold, max_clusters, creation_buffer_size)
        predictions.append(cluster_id)
    return np.array(predictions), cluster_weights

# Processes each new sample for ART-2 clustering
def process_new_sample(input_features, cluster_weights, num_clusters, cluster_creation_buffer, 
                      vigilance_threshold, max_clusters, creation_buffer_size):
    if num_clusters == 0:
        cluster_weights.append(input_features.copy())
        return 0, cluster_weights, 1, 0
    distances = [np.sum(np.abs(input_features - weights)) for weights in cluster_weights]
    closest_cluster = np.argmin(distances)
    min_distance = distances[closest_cluster]
    if min_distance <= vigilance_threshold:
        cluster_weights[closest_cluster] = (cluster_weights[closest_cluster] + input_features) / 2
        return closest_cluster, cluster_weights, num_clusters, 0
    cluster_creation_buffer += 1
    if cluster_creation_buffer >= creation_buffer_size and num_clusters < max_clusters:
        cluster_weights.append(input_features.copy())
        num_clusters += 1
        return num_clusters - 1, cluster_weights, num_clusters, 0
    return closest_cluster, cluster_weights, num_clusters, cluster_creation_buffer

# Evaluates clustering performance on time series files
def evaluate_clustering(time_series_files, vigilance_threshold, creation_buffer_size):
    results = []
    for file_idx, file in enumerate(time_series_files):
        df = pd.read_csv(file)
        features = []
        for _, row in df.iterrows():
            mfsc = np.array([float(x) for x in row['MFSC'].strip('[]').split()])
            mfcc = np.array([float(x) for x in row['MFCC'].strip('[]').split()])
            combined = np.concatenate([mfsc, mfcc])
            features.append(combined)
        features = np.array(features)
        true_labels = df['class_id'].values
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        predicted_labels, cluster_weights = run_art2_clustering(features, vigilance_threshold, creation_buffer_size)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)
        results.append({
            'file': file,
            'confusion_matrix': conf_matrix,
            'accuracy': accuracy,
            'predictions': predicted_labels,
            'cluster_weights': cluster_weights
        })
    return results

# Saves Clustering Results to the CSV file

def save_clustering_results(clustering_results, time_series_files):
    # Process each time series separately
    for idx, file in enumerate(time_series_files):
        # Load original time series data
        df = pd.read_csv(file)
        
        # Get predictions for this time series
        result = clustering_results[idx]
        predictions = result['predictions']
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'class_id': df['class_id'],
            'timestamp': df['timestamp'],
            'MFSC': df['MFSC'],
            'MFCC': df['MFCC'],
            'cluster': predictions
        })
        
        # Save to CSV
        output_file = f'clustering_results_series_{idx+1}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Results for Time Series {idx+1} saved to {output_file}")
        
# Plots the graph for results of each of the time series.
        
def plot_clustering_comparison(time_series_files, clustering_results):
    for idx, (file, result) in enumerate(zip(time_series_files, clustering_results)):
        # Load data
        df = pd.read_csv(file)
        timestamps = df['timestamp'].values
        true_labels = df['class_id'].values
        predicted_labels = result['predictions']
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Predicted Clusters vs Time
        ax1.set_title('Predicted: predicted_class_mfccs vs Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Predicted Class')
        
        # Create step plot for predicted clusters
        for cluster in range(max(predicted_labels) + 1):
            mask = predicted_labels == cluster
            if any(mask):  # Only plot if cluster exists
                ax1.fill_between(timestamps[mask], cluster, cluster + 1, 
                               color='black', alpha=1, step='post')
        
        ax1.set_ylim(-0.5, max(predicted_labels) + 0.5)
        ax1.grid(True)
        
        # Plot 2: Ground Truth Clusters vs Time
        ax2.set_title('Mapped Class ID vs Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Mapped Class')
        
        # Create step plot for true labels
        for cluster in range(max(true_labels) + 1):
            mask = true_labels == cluster
            if any(mask):  # Only plot if cluster exists
                ax2.fill_between(timestamps[mask], cluster, cluster + 1, 
                               color='black', alpha=1, step='post')
        
        ax2.set_ylim(-0.5, max(true_labels) + 0.5)
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'clustering_comparison_series_{idx+1}.png', dpi=300, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    
    # Initialize variables
    vigilance_threshold = 0.6  # Threshold for cluster matching
    creation_buffer_size = 7  # How many consecutive failed samples before creating new cluster
    num_clusters = 0  # Initial number of clusters
    cluster_weights = []  # Initialize empty cluster weights list
    cluster_creation_buffer = 0  # Initialize creation buffer counter
    

    audio_files = [af for af in os.listdir() if af.endswith(".mp3")]
    combined_features = []
    combined_timestamps = []
    frm_sz = 35 # Defining the Frame_Size !! Very important parameter
    sr = 22050 # The sampling rate of the given audio file
    
    # Process audio files
    for i, file in enumerate(audio_files):
        MFSC, MFCC, TIMESTAMPS = load_and_process_audio(file,frm_sz)
        feat, times = write_to_csv(MFSC, MFCC, TIMESTAMPS, i)
        combined_features.append(feat)
        combined_timestamps.append(times)
    
    # Load the CSV files
    feature_files = ['Feature_0.csv', 'Feature_1.csv', 'Feature_2.csv']
    features = [pd.read_csv(file) for file in feature_files]
    
    
    
    class_ids = list(range(len(audio_files)))  # Assuming class_id corresponds to file index
    
    series_list = generate_time_series_combinations(features, class_ids)
    
    # Save each time-series to a separate CSV file
    for i, series in enumerate(series_list):
        series.to_csv(f'Time_Series_{i+1}.csv', index=False)
        print(f"Saved Time Series {i+1}")
    
    print("All time-series combinations have been generated and saved.")
    
    
    
    # Generating Time-Series Combination
    time_series_files = [f'Time_Series_{i+1}.csv' for i in range(5)]
    
    
    #Running evaluation
    clustering_results = evaluate_clustering(time_series_files,vigilance_threshold,creation_buffer_size)
    
    # Saving the clustering evaluation results
    save_clustering_results(clustering_results, time_series_files)
    
    # Plotting and saving the clustering evaluation results
    plot_clustering_comparison(time_series_files, clustering_results)
    
    
