import os
import gdown

# Google Drive file ID from your link
FILE_ID = "1NgOvls3LBot3Ve5z9eQHiNETecKCbrth"
TARGET_PATH = "model/tf_model.h5"

os.makedirs("model", exist_ok=True)

url = f"https://drive.google.com/uc?id={FILE_ID}"
print("Downloading the model...")
gdown.download(url, TARGET_PATH, quiet=False)
print(f"Model downloaded to {TARGET_PATH}")
