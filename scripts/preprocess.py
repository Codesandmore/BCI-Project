import numpy as np
from mne.filter import notch_filter
from scipy.signal import butter, filtfilt

def notch_filter_trials(X, fs=250, freq=50):
    X_shape = X.shape
    X_flat = X.reshape(-1, X_shape[-1])
    X_filtered = notch_filter(X_flat, Fs=fs, freqs=freq, filter_length='auto')
    return X_filtered.reshape(X_shape)

def bandpass_filter_trials(X, lowcut=4, highcut=38, fs=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    X_shape = X.shape
    X_flat = X.reshape(-1, X_shape[-1])
    X_filtered = filtfilt(b, a, X_flat, axis=-1)
    return X_filtered.reshape(X_shape)

def standardize_trials(X):
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True) + 1e-6  # Add epsilon here
    return (X - mean) / std

def preprocess_bci_iii3a(s, events, types, labels, fs=250, trial_len_sec=4):
    # Event codes for motor imagery classes
    class_events = [769, 770, 771, 772]
    trial_len = int(trial_len_sec * fs)  # samples per trial

    X_trials = []
    y_trials = []

    for pos, typ in zip(events, types):
        if typ in class_events:
            start = int(pos)
            end = start + trial_len
            if end <= s.shape[0]:
                X_trials.append(s[start:end, :].T)  # shape: (channels, samples)
                y_trials.append(class_events.index(typ))  # 0,1,2,3 for classes

    X_trials = np.stack(X_trials)  # shape: (trials, channels, samples)
    y_trials = np.array(y_trials)
    return X_trials, y_trials

def crop_trials(X, y, window_size=500, step=25):
    # X: (trials, channels, samples)
    cropped_X, cropped_y = [], []
    for trial, label in zip(X, y):
        for start in range(0, trial.shape[-1] - window_size + 1, step):
            window = trial[:, start:start+window_size]
            cropped_X.append(window)
            cropped_y.append(label)
    return np.array(cropped_X), np.array(cropped_y)