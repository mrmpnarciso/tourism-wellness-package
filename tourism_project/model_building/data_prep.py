
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi

# ---- Config: update before running ----
HF_USERNAME = "mrmpnarciso"
HF_DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-package-dataset"
LOCAL_RAW_PATH = "tourism_project/data/tourism.csv"

HF_TOKEN = os.getenv("HF_TOKEN")


def load_raw_data():
    """Prefer the local file uploaded in Data Registration. Falls back to
    Hugging Face Hub, which is how this script gets its input once it's
    running inside the GitHub Actions pipeline (no local file there)."""
    if os.path.exists(LOCAL_RAW_PATH):
        return pd.read_csv(LOCAL_RAW_PATH)

    downloaded_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename="tourism.csv",
        token=HF_TOKEN,
    )
    return pd.read_csv(downloaded_path)


def clean_data(df):
    df = df.copy()

    df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

    # Fix typos/duplicate labels found during EDA
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Defensive missing-value handling for future data drops
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop("ProdTaken")
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def split_and_save(df):
    target = "ProdTaken"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs("tourism_project/data", exist_ok=True)
    X_train.to_csv("tourism_project/data/X_train.csv", index=False)
    X_test.to_csv("tourism_project/data/X_test.csv", index=False)
    y_train.to_csv("tourism_project/data/y_train.csv", index=False)
    y_test.to_csv("tourism_project/data/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test


def upload_to_hub():
    api = HfApi(token=HF_TOKEN)
    for filename in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        api.upload_file(
            path_or_fileobj=f"tourism_project/data/{filename}",
            path_in_repo=filename,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    split_and_save(clean_df)

    if HF_TOKEN:
        upload_to_hub()
    else:
        print("HF_TOKEN not set, skipping upload. Splits are saved locally "
              "under tourism_project/data/.")
