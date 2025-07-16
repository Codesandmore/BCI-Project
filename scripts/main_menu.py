import os

def run_tsgl_eegnet():
    print("Running TSGL-EEGNet pipeline...")
    os.system("python scripts/train_eegnet.py")

def run_baseline_eegnet():
    print("Running baseline EEGNet pipeline...")
    os.system("python scripts/train_baseline_eegnet.py")

def run_fbcsp():
    print("Running FBCSP + SVM pipeline...")
    os.system("python scripts/fbcsp_pipeline.py")

if __name__ == "__main__":
    print("Select pipeline to run:")
    print("1. TSGL-EEGNet")
    print("2. Baseline EEGNet")
    print("3. FBCSP + SVM")
    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        run_tsgl_eegnet()
    elif choice == "2":
        run_baseline_eegnet()
    elif choice == "3":
        run_fbcsp()
    else:
        print("Invalid choice.")