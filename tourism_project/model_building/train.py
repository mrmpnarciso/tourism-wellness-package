
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from huggingface_hub import hf_hub_download, HfApi

# ---- Config: update before running ----
HF_USERNAME = "mrmpnarciso"
HF_DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-package-dataset"
HF_MODEL_REPO = f"{HF_USERNAME}/tourism-wellness-package-model"
LOCAL_DATA_DIR = "tourism_project/data"

HF_TOKEN = os.getenv("HF_TOKEN")


def load_split(filename):
    local_path = f"{LOCAL_DATA_DIR}/{filename}"
    if os.path.exists(local_path):
        return pd.read_csv(local_path)

    downloaded_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=filename,
        token=HF_TOKEN,
    )
    return pd.read_csv(downloaded_path)


def build_pipeline(X_train):
    categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    return pipeline


def train_and_track():
    X_train = load_split("X_train.csv")
    X_test = load_split("X_test.csv")
    y_train = load_split("y_train.csv").squeeze()
    y_test = load_split("y_test.csv").squeeze()

    pipeline = build_pipeline(X_train)

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10, None],
    }

    mlflow.set_experiment("tourism-wellness-package")

    with mlflow.start_run(run_name="random-forest-grid-search") as run:
        search = GridSearchCV(pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        preds = best_pipeline.predict(X_test)
        probs = best_pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_pipeline, name="model", serialization_format="pickle")
        print("Best params:", search.best_params_)
        print("Test metrics:", metrics)

        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="tourism-wellness-package-model")

    model_path = "tourism_project/model_building/best_model.joblib"
    joblib.dump(best_pipeline, model_path)

    if HF_TOKEN:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_model.joblib",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            token=HF_TOKEN,
        )
    else:
        print("HF_TOKEN not set, skipping model upload.")


if __name__ == "__main__":
    train_and_track()
