from load_bci_iii3a import load_bci_iii3a_mat
from preprocess import preprocess_bci_iii3a, notch_filter_trials, bandpass_filter_trials, standardize_trials, crop_trials
import numpy as np
from sklearn.model_selection import train_test_split

subjects = ['k3b.mat', 'k6b.mat', 'l1b.mat']

for subj in subjects:
    print(f"\nProcessing {subj}...")
    s, events, types, labels = load_bci_iii3a_mat(f'data/BCICIII-3a-mat/{subj}')
    print("EEG shape:", s.shape)
    print("Events shape:", events.shape)
    print("Types shape:", types.shape)

    X_trials, y_trials = preprocess_bci_iii3a(s, events, types, labels)
    print("Trials shape:", X_trials.shape)
    print("Labels shape:", y_trials.shape)

    # Filtering and standardization
    X_notched = notch_filter_trials(X_trials, fs=250)
    X_band = bandpass_filter_trials(X_notched, lowcut=4, highcut=38, fs=250)
    X_std = standardize_trials(X_band)
    print("Preprocessed trials shape:", X_std.shape)

    # Cropping
    X_crop, y_crop = crop_trials(X_std, y_trials, window_size=500, step=25)
    print("Cropped shape:", X_crop.shape)
    print("Cropped labels shape:", y_crop.shape)

    np.save(f"data/X_{subj.replace('.mat','')}_crop.npy", X_crop)
    np.save(f"data/y_{subj.replace('.mat','')}_crop.npy", y_crop)

X_list, y_list = [], []
for subj in ['k3b', 'k6b', 'l1b']:
    X = np.load(f"data/X_{subj}_crop.npy")
    y = np.load(f"data/y_{subj}_crop.npy")
    X_list.append(X)
    y_list.append(y)
X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)

# Add the checks here:
print(np.isnan(X_all).sum(), np.isinf(X_all).sum())
print(np.isnan(y_all).sum(), np.isinf(y_all).sum())
print(np.unique(y_all))
print(X_all.shape, y_all.shape)

# Remove any samples with NaNs
mask = ~np.isnan(X_all).any(axis=(1,2))
X_all_clean = X_all[mask]
y_all_clean = y_all[mask]

print("After removing NaNs:", X_all_clean.shape, y_all_clean.shape)
print("NaNs left:", np.isnan(X_all_clean).sum())

# Split into train/validation and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all_clean, y_all_clean, test_size=0.2, random_state=123, stratify=y_all_clean
)

np.save("data/X_trainval.npy", X_trainval)
np.save("data/y_trainval.npy", y_trainval)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)