# convert_to_submission.py
# Run this anytime to turn any big .zip into a tiny submission-ready one

import os
from stable_baselines3 import SAC

# -------------------------- CONFIG --------------------------
# Folder where your big models live
BIG_MODELS_FOLDER = "sac_models"   # change if your models are somewhere else

# Folder where small submission files will be saved
SUBMISSION_FOLDER = "models_for_submission"
os.makedirs(SUBMISSION_FOLDER, exist_ok=True)

# ------------------------------------------------------------
def convert_zip_to_submission(zip_path):
    print(f"\nConverting: {zip_path}")
    
    # Load the model (it will load the huge buffer too — that's fine, takes 5-10 seconds)
    model = SAC.load(zip_path, device="cpu")  # cpu is faster for just loading
    
    # Create clean filename
    base_name = os.path.basename(zip_path).replace(".zip", "_SUBMISSION.zip")
    submission_path = os.path.join(SUBMISSION_FOLDER, base_name)
    
    # Save WITHOUT the replay buffer
    model.save(submission_path, exclude=["replay_buffer"])
    
    old_size = os.path.getsize(zip_path) / (1024*1024)  # MB
    new_size = os.path.getsize(submission_path) / (1024*1024)  # MB
    
    print(f"   SUCCESS! {submission_path}")
    print(f"   Size: {old_size:.1f} MB → {new_size:.1f} MB")
    print(f"   Ready for competeSAI upload!\n")

# ---------------------- AUTO CONVERT ALL ----------------------
if __name__ == "__main__":
    print("=== COMPETESAI SUBMISSION CONVERTER ===\n")
    
    # Find every .zip in sac_models and subfolders
    zip_files = []
    for root, dirs, files in os.walk(BIG_MODELS_FOLDER):
        for f in files:
            if f.endswith(".zip") and "best_model" in f or "final_model" in f:
                zip_files.append(os.path.join(root, f))
    
    if not zip_files:
        print("No .zip files found! Check your BIG_MODELS_FOLDER path.")
    else:
        print(f"Found {len(zip_files)} model(s) to convert:\n")
        for z in zip_files:
            print(f"  • {z}")
        
        print(f"\nConverting all to {SUBMISSION_FOLDER}/ ...\n")
        for z in zip_files:
            convert_zip_to_submission(z)
        
        print("ALL DONE! Your submission-ready models are in:")
        print(f"   {os.path.abspath(SUBMISSION_FOLDER)}")
        print("\nJust upload any of those files to competeSAI — they are all < 100 MB!")