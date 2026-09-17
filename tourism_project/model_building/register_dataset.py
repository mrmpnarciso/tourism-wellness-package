
import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "mrmpnarciso"
HF_DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-package-dataset"
LOCAL_RAW_PATH = "tourism_project/data/tourism.csv"

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj=LOCAL_RAW_PATH,
    path_in_repo="tourism.csv",
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN,
)
print(f"Raw dataset pushed to: https://huggingface.co/datasets/{HF_DATASET_REPO}")
