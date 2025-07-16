import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from scipy.signal import butter, lfilter
import pickle

bands = [
    (4, 8), (8, 12), (12, 16), (16, 20),
    (20, 24), (24, 28), (28, 32), (32, 36)
]
fs = 250

def bandpass_filter(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return lfilter(b, a, data, axis=-1)

X = np.load('data/X_trainval.npy')
y = np.load('data/y_trainval.npy')

split_seed = 46

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=split_seed, stratify=y
)

n_components = 4
features_train = []
features_val = []
csp_list = []

for low, high in bands:
    X_train_filt = np.array([bandpass_filter(trial, low, high, fs) for trial in X_train])
    X_val_filt = np.array([bandpass_filter(trial, low, high, fs) for trial in X_val])
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train_filt, y_train)
    X_val_csp = csp.transform(X_val_filt)
    features_train.append(X_train_csp)
    features_val.append(X_val_csp)
    csp_list.append(csp)

X_train_fbcsp = np.concatenate(features_train, axis=1)
X_val_fbcsp = np.concatenate(features_val, axis=1)

scaler = StandardScaler()
X_train_fbcsp = scaler.fit_transform(X_train_fbcsp)
X_val_fbcsp = scaler.transform(X_val_fbcsp)

clf = SVC(kernel='linear')
clf.fit(X_train_fbcsp, y_train)
y_pred = clf.predict(X_val_fbcsp)
acc = accuracy_score(y_val, y_pred)

model_filename = f"fbcsp_svm_seed_{split_seed}.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump({
        "svm_classifier": clf,
        "scaler": scaler,
        "csp_objects": csp_list,
        "accuracy": acc,
        "seed": split_seed
    }, f)

with open("validation_log.txt", 'a') as f:
    f.write(f"Seed {split_seed}: Accuracy = {acc:.4f}\n")

print(f"\nModel saved as {model_filename} with Accuracy: {acc:.4f}")
