import librosa
import numpy as np
import math
import pandas as pd
from art2 import Art2

def load_audio(file_path):
    # Load audio file
    audio_signal, sample_rate = librosa.load(
        file_path, sr=None)  # Use original sample rate
    return audio_signal, sample_rate


def extract_features(audio_signal, sample_rate, n_fft):
    mfccs = librosa.feature.mfcc(
        y=audio_signal, sr=sample_rate, n_mfcc=13, n_fft=n_fft)
    mfsc = librosa.feature.melspectrogram(
        y=audio_signal, sr=sample_rate, n_mels=40, n_fft=n_fft)
    return mfccs, mfsc


def divide_into_frames(audio_signal, frame_size, sample_freq, overlap):
    # Convert frame_size ms to samples
    frame_length = int(frame_size * sample_freq / 1000)
    hop_size = math.floor(frame_length * (1 - overlap))
    # Create overlapping frames
    frames = librosa.util.frame(
        audio_signal, frame_length=frame_length, hop_length=hop_size)
    return frames.T  # Transpose to have frames as rows



def main():

    # Config variables
    mp3_file = "dataset/trimmed_crowd_talking.mp3"
    frame_size = 10  # Number of samples per frame
    overlap = 0.1     # Number of samples to move for the next frame

    n_fft = 256
    # Load audio and create frames
    audio_signal, sample_rate = load_audio(mp3_file)
    frames = divide_into_frames(audio_signal, frame_size, sample_rate, overlap)

    mfccs_list = []
    mfsc_list = []

    for frame in frames:
        mfccs, mfsc = extract_features(frame, sample_rate, n_fft)
        mfccs_list.append(mfccs.flatten())  # Flatten to 1D array
        mfsc_list.append(mfsc.flatten())    # Flatten to 1D array

    # Create a DataFrame
    index = np.arange(len(mfccs_list))
    # Calculate timestamps in seconds
    timestamps = (index * (len(frames[0]) / sample_rate)).round(2)
    df = pd.DataFrame(mfccs_list, index=index)
    df.columns = [f'MFCC_{i}' for i in range(df.shape[1])]

    # Add timestamps
    df.insert(0, 'Index', index)
    df.insert(1, 'Timestamp', timestamps)

    # Add MFSC as additional columns
    mfsc_df = pd.DataFrame(mfsc_list, index=index)
    mfsc_df.columns = [f'MFSC_{i}' for i in range(mfsc_df.shape[1])]
    combined_df = pd.concat([df, mfsc_df], axis=1)

    # Save to CSV
    #combined_df.to_csv("output.csv", index=False)

    print(combined_df.head())

if __name__ == "__main__":
    main()