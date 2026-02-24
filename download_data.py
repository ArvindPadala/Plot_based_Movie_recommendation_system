import os
from huggingface_hub import snapshot_download

def download_data():
    """
    Downloads the entire contents of the dataset repository from Hugging Face 
    into the local 'data' directory.
    """
    repo_id = "ArvindPadala/Movie-Recommendation-Data"
    
    # Path to the local data directory
    local_dir = os.path.join(os.path.dirname(__file__), "data")
    
    print(f"Downloading required models and data from Hugging Face Hub: {repo_id}")
    print(f"This will download ~500MB of data to {local_dir}...")
    print("Please wait, this may take a few minutes depending on your internet connection.\n")
    
    # Create the directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Download the repository snapshot
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False # Ensure actual files are downloaded, not just symlinks to cache
        )
        print("\n✅ Download completed successfully! You can now run the Streamlit app.")
    except Exception as e:
        print(f"\n❌ An error occurred during the download: {e}")
        print("\nPlease ensure you have an active internet connection and try again.")

if __name__ == "__main__":
    download_data()
