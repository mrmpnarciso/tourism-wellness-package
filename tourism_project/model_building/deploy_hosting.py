
import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "mrmpnarciso"
HF_SPACE_REPO = f"{HF_USERNAME}/tourism-wellness-package-app"

api = HfApi(token=HF_TOKEN)
api.create_repo(
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    space_sdk="gradio",
    space_hardware="zero-a10g",
    exist_ok=True,
)
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    token=HF_TOKEN,
    ignore_patterns=["Dockerfile"],
)
print(f"Space live at: https://huggingface.co/spaces/{HF_SPACE_REPO}")
