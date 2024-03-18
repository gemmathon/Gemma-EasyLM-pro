from huggingface_hub import HfApi

api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path="./gemma-ko-2b-dev",
    repo_id="gemmathon/gemma-ko-2b-dev",
    repo_type="model",
)