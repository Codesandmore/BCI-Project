BCI-Project: EEGNet Ensemble for EEG Classification

Welcome! This project is all about classifying EEG signals using deep learning and ensemble techniques.
We use the EEGNet architecture, a special loss function (TSGLoss), and combine several models to get the best results.

What’s Inside

Preprocessing: Clean and prepare your EEG data. Training: Train multiple neural networks (base learners) with different random seeds. Ensembling: Combine the predictions of all base learners for more robust results. Visualization: See what your models are learning with filter and frequency plots.

Project Structure

BCI-Project/ data/ (Your data goes here, not included in the repo) scripts/ eegnet.py tsgl_loss.py preprocess.py run_preprocessing.py train_eegnet.py ensemble_stack.py visualize_filters.py ... requirements.txt .gitignore README.md

Data

Note:
The data files (.npy, .mat, etc.) are not included in this repository.
Please preprocess your own EEG data and place the resulting files in the data/ folder.
If you need help with data preparation, just ask!

How to Use

Preprocess your data:
python run_preprocessing.py

Train the base learners (change the seed in train_eegnet.py for each run):
python train_eegnet.py

Run the ensemble stacking:
python ensemble_stack.py

Visualize what the models learned:
python visualize_filters.py

Why Seeds 42–46?

We use seeds 42–46 to make sure our models are diverse and our results are reproducible.
Seed 42 is a classic in machine learning, and using a range helps our ensemble work better.

Requirements

See requirements.txt for all dependencies.

License

This project is licensed under the MIT License (LICENSE).