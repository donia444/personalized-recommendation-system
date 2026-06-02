from pathlib import Path
import json
import mlflow


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model_artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


# =========================
# MLflow setup
# =========================

FINAL_MODEL_NAME = "Hybrid ALS + Content-Based Recommender"
EXPERIMENT_NAME = "Final_Recommender_Clean_Tracking"


# =========================
# Final model parameters
# =========================

FINAL_PARAMS = {
    "model_type": "hybrid_recommender",
    "collaborative_model": "ALS",
    "content_model": "TF-IDF Content-Based",
    "fusion_method": "Weighted Rank Fusion",
    "als_weight": 0.7,
    "content_weight": 0.3,
    "candidate_k": 100,
    "cold_start_strategy": "Popularity Fallback",
    "evaluation_split": "test_all",
    "evaluation_data": "view + addtocart + transaction",
    "deployment_status": "local_demo_only",
}


# =========================
# Key final metrics only
# =========================
# These metrics are copied from the final notebook output:
# FINAL HYBRID - TEST ALL
#
# We only log @10 metrics to keep the MLflow dashboard clean.

FINAL_METRICS = {
    "test_all_precision_at_10": 0.027760,
    "test_all_recall_at_10": 0.184824,
    "test_all_map_at_10": 0.072545,
    "test_all_ndcg_at_10": 0.106818,
    "test_all_coverage_at_10": 0.823475,
}


# =========================
# Helper functions
# =========================

def log_artifact_if_exists(path):
    """
    Log an artifact to MLflow only if the file exists.
    Missing optional files are skipped without stopping the script.
    """

    if path.exists():
        mlflow.log_artifact(str(path))
        print(f"Logged artifact: {path.name}")
    else:
        print(f"Skipped missing artifact: {path.name}")


# =========================
# Main tracking script
# =========================

def main():
    """
    Track the final selected recommender model using MLflow.

    This script logs:
    - final model identity
    - final model parameters
    - key @10 evaluation metrics
    - important saved model artifacts
    - a small JSON summary report
    """

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="final_hybrid_als_content_clean"):

        # Tags describe the run context.
        mlflow.set_tag("project", "Retailrocket Personalized Recommendation System")
        mlflow.set_tag("milestone", "Milestone 4 - MLOps")
        mlflow.set_tag("model_name", FINAL_MODEL_NAME)
        mlflow.set_tag("tracking_type", "final_model_clean_summary")
        mlflow.set_tag("evaluation_focus", "test_all_at_10")
        mlflow.set_tag("cloud_deployment", "cancelled")
        mlflow.set_tag("demo_type", "local_streamlit_demo")

        # Log final model parameters.
        for key, value in FINAL_PARAMS.items():
            mlflow.log_param(key, value)

        # Log only key @10 metrics.
        for key, value in FINAL_METRICS.items():
            mlflow.log_metric(key, value)

        # Log important artifacts only.
        artifact_files = [
            "hybrid_model_bundle.pkl",
            "user_encoder.pkl",
            "item_encoder.pkl",
            "train_user_item_matrix.npz",
            "best_hybrid_params.json",
        ]

        for filename in artifact_files:
            log_artifact_if_exists(MODEL_DIR / filename)

        # Create a small summary report.
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        summary = {
            "model_name": FINAL_MODEL_NAME,
            "experiment_name": EXPERIMENT_NAME,
            "run_name": "final_hybrid_als_content_clean",
            "params": FINAL_PARAMS,
            "metrics": FINAL_METRICS,
            "notes": (
                "This is a simplified MLflow tracking run. "
                "Only key @10 test_all metrics are logged to keep the dashboard clean. "
                "Full evaluation metrics are available in the final notebook. "
                "Milestone 3 cloud deployment was cancelled, so the model is demonstrated locally using Streamlit."
            ),
        }

        summary_path = REPORTS_DIR / "mlflow_clean_summary.json"

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4)

        mlflow.log_artifact(str(summary_path))

        print("\nClean MLflow tracking completed successfully.")
        print(f"Experiment name: {EXPERIMENT_NAME}")
        print("Run name: final_hybrid_als_content_clean")
        print(f"Model tracked: {FINAL_MODEL_NAME}")


if __name__ == "__main__":
    main()