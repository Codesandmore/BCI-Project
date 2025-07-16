import numpy as np
import pickle
import re
from sklearn.metrics import accuracy_score
from scipy.signal import butter, lfilter

from mne.decoding import CSP

bands = [
    (4, 8), (8, 12), (12, 16), (16, 20),
    (20, 24), (24, 28), (28, 32), (32, 36)
]
fs = 250

def bandpass_filter(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data, axis=-1)

# Step 1: Find best seed from validation_log.txt
best_seed = None
best_acc = -1.0

with open("validation_log.txt", "r") as f:
    for line in f:
        match = re.search(r"Seed (\d+): Accuracy = ([0-9.]+)", line)
        if match:
            seed = int(match.group(1))
            acc = float(match.group(2))
            if acc > best_acc:
                best_acc = acc
                best_seed = seed

if best_seed is None:
    print("No valid model found in validation_log.txt.")
    exit()

print(f" Best model is from seed {best_seed} with validation accuracy {best_acc:.4f}")

# Step 2: Load the best model
model_path = f"fbcsp_svm_seed_{best_seed}.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

clf = model["svm_classifier"]
scaler = model["scaler"]
csp_list = model["csp_objects"]

# Step 3: Load test data
X_test = np.load('data/X_test.npy')         # shape: (n_trials, n_channels, n_samples)
y_test = np.load('data/y_test.npy')

# Step 4: Apply bandpass, CSP, and scaler
features_test = []

for i, (low, high) in enumerate(bands):
    X_test_filt = np.array([bandpass_filter(trial, low, high, fs) for trial in X_test])
    X_test_csp = csp_list[i].transform(X_test_filt)
    features_test.append(X_test_csp)

X_test_fbcsp = np.concatenate(features_test, axis=1)
X_test_fbcsp = scaler.transform(X_test_fbcsp)

# Step 5: Predict and evaluate
y_pred = clf.predict(X_test_fbcsp)
test_acc = accuracy_score(y_test, y_pred)

print(f"\n Test accuracy of best model (seed {best_seed}): {test_acc:.4f}")
