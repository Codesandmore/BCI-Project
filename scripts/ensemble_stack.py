import torch
import torch.nn.functional as F
import numpy as np
from eegnet import EEGNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K = 5
seeds = [42, 43, 44, 45, 46]

X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

softmax_outputs = []
for seed in seeds:
    model = EEGNet(n_channels=60, n_samples=500, n_classes=4).to(device)
    with torch.no_grad():
        _ = model(X_test_tensor[:1])
    model.load_state_dict(torch.load(f"eegnet_tsgl_best_seed{seed}.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        softmax_outputs.append(probs)

X_stack = np.concatenate(softmax_outputs, axis=1)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_stack, y_test)
y_pred = clf.predict(X_stack)

acc = accuracy_score(y_test, y_pred)
print(f"Stacked Ensemble Accuracy: {acc:.4f}")

with open('stacking_regressor.pkl', 'wb') as f:
    pickle.dump(clf, f)