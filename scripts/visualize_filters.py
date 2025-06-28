import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import torch
import numpy as np
import matplotlib.pyplot as plt
from eegnet import EEGNet

SEEDS = [42, 43, 44, 45, 46]
SAMPLING_RATE = 250  # Hz

plt.figure(figsize=(12, 7))

for seed in SEEDS:
    model_path = f"eegnet_tsgl_best_seed{seed}.pth"
    model = EEGNet(n_channels=60, n_samples=500, n_classes=4)
    model.eval()
    dummy_input = torch.zeros(1, 60, 500)
    with torch.no_grad():
        _ = model(dummy_input)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    weights = model.conv1.weight.data.cpu().numpy()  # shape: (F1, 1, 1, 64)
    F1 = weights.shape[0]

    for i in range(F1):
        w_flat = weights[i].flatten()
        fft_vals = np.abs(np.fft.fft(w_flat))
        freqs = np.fft.fftfreq(len(w_flat), d=1/SAMPLING_RATE)
        plt.plot(freqs[:len(freqs)//2], fft_vals[:len(freqs)//2], 
                 label=f'Seed {seed} Filter {i+1}', alpha=0.7)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT of Temporal Filters (conv1) - All Base Learners')
plt.legend(fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("all_base_learners_fft.png")
print("Plot saved as all_base_learners_fft.png")