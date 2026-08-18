import os
import sys
import warnings

# Monkey-patch 'requests' to globally disable SSL verification
import requests
original_request = requests.Session.request

def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False  # Force ignore SSL
    return original_request(self, method, url, **kwargs)

requests.Session.request = patched_request

# Now we can safely import huggingface_hub and it will use the patched requests
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir} (Bypassing SSL)...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            # Ignore some massive unnecessary files if present (like safetensors if bin exists)
            ignore_patterns=["*.msgpack", "*.h5", "rust_model.ot"]
        )
        print(f"Successfully downloaded {repo_id}!")
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_models_dir = os.path.join(base_dir, "local_models")
    os.makedirs(local_models_dir, exist_ok=True)
    
    download_model("thearod5/se-bert", os.path.join(local_models_dir, "sebert"))
    download_model("Jingye/BERT4RE", os.path.join(local_models_dir, "rebert"))
