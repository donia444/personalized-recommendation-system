from pathlib import Path
import argparse
import json
import time

import joblib
import mlflow
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix, load_npz, save_npz
from sklearn.preprocessing import LabelEncoder, normalize
from implicit.als import AlternatingLeastSquares

from run_local_demo import load_artifacts, validate_bundle


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "model_artifacts"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEEDBACK_LOG_PATH = DATA_DIR / "feedback_logs" / "feedback_logs.csv"

OUTPUT_RETRAINING_DIR = PROJECT_ROOT / "outputs" / "retraining"
REPORTS_DIR = PROJECT_ROOT / "reports"

EXPERIMENT_NAME = "Final_Recommender_Clean_Tracking"


# =========================
# Load input data
# =========================

def find_train_matrix_path():
    """
    Find the original training matrix.

    Expected shape:
    users x items
    """

    possible_paths = [
        MODEL_DIR / "train_user_item_matrix.npz",
        PROCESSED_DIR / "train_user_item_matrix.npz",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "train_user_item_matrix.npz was not found. "
        "Copy it to model_artifacts/ or data/processed/."
    )


def load_feedback_logs(min_feedback_events):
    """
    Load feedback logs collected from Streamlit.

    The feedback logs contain real user interactions:
    visitorid, itemid, event, weight.
    """

    if not FEEDBACK_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Feedback log not found: {FEEDBACK_LOG_PATH}"
        )

    feedback_df = pd.read_csv(FEEDBACK_LOG_PATH)

    if feedback_df.empty:
        raise ValueError("Feedback log is empty.")

    required_columns = {"visitorid", "itemid", "event", "weight"}
    missing_columns = required_columns - set(feedback_df.columns)

    if missing_columns:
        raise ValueError(f"Missing feedback columns: {missing_columns}")

    if len(feedback_df) < min_feedback_events:
        raise ValueError(
            f"Not enough feedback events.\n"
            f"Current: {len(feedback_df)}\n"
            f"Required: {min_feedback_events}\n"
            "For testing, use --min_feedback_events 1"
        )

    feedback_df["visitorid"] = feedback_df["visitorid"].astype(str)
    feedback_df["itemid"] = feedback_df["itemid"].astype(str)
    feedback_df["weight"] = feedback_df["weight"].astype(float)

    return feedback_df


# =========================
# Build candidate training data
# =========================

def build_candidate_user_encoder(old_user_encoder, feedback_df):
    """
    Create a new user encoder that includes:
    - old training users
    - new users from feedback logs
    """

    old_users = [str(user) for user in old_user_encoder.classes_]
    feedback_users = feedback_df["visitorid"].astype(str).tolist()

    all_users = sorted(set(old_users) | set(feedback_users))

    candidate_user_encoder = LabelEncoder()
    candidate_user_encoder.fit(all_users)

    return candidate_user_encoder


def remap_old_matrix_to_candidate_users(
    old_matrix,
    old_user_encoder,
    candidate_user_encoder,
):
    """
    Rebuild the old training matrix using the candidate user encoder order.

    This is needed because the candidate encoder may contain new users.
    """

    old_users_as_str = np.array([str(user) for user in old_user_encoder.classes_])
    old_rows_in_candidate = candidate_user_encoder.transform(old_users_as_str)

    old_coo = old_matrix.tocoo()

    new_rows = old_rows_in_candidate[old_coo.row]
    new_cols = old_coo.col
    new_data = old_coo.data

    new_shape = (
        len(candidate_user_encoder.classes_),
        old_matrix.shape[1],
    )

    remapped_matrix = coo_matrix(
        (new_data, (new_rows, new_cols)),
        shape=new_shape,
    ).tocsr()

    remapped_matrix.sum_duplicates()

    return remapped_matrix


def build_feedback_matrix(feedback_df, candidate_user_encoder, item_encoder):
    """
    Convert feedback logs into a sparse user-item matrix.

    Unknown items are skipped.
    """

    item_to_idx = {
        str(itemid): idx
        for idx, itemid in enumerate(item_encoder.classes_)
    }

    rows = []
    cols = []
    values = []
    skipped_unknown_items = 0

    for _, row in feedback_df.iterrows():
        visitorid = str(row["visitorid"])
        itemid = str(row["itemid"])
        weight = float(row["weight"])

        if itemid not in item_to_idx:
            skipped_unknown_items += 1
            continue

        user_idx = int(candidate_user_encoder.transform([visitorid])[0])
        item_idx = int(item_to_idx[itemid])

        rows.append(user_idx)
        cols.append(item_idx)
        values.append(weight)

    feedback_matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(len(candidate_user_encoder.classes_), len(item_encoder.classes_)),
    ).tocsr()

    feedback_matrix.sum_duplicates()

    return feedback_matrix, skipped_unknown_items


def build_candidate_matrix(
    old_train_matrix,
    old_user_encoder,
    candidate_user_encoder,
    item_encoder,
    feedback_df,
):
    """
    Combine:
    original training interactions + new feedback interactions
    """

    remapped_old_matrix = remap_old_matrix_to_candidate_users(
        old_matrix=old_train_matrix,
        old_user_encoder=old_user_encoder,
        candidate_user_encoder=candidate_user_encoder,
    )

    feedback_matrix, skipped_unknown_items = build_feedback_matrix(
        feedback_df=feedback_df,
        candidate_user_encoder=candidate_user_encoder,
        item_encoder=item_encoder,
    )

    candidate_matrix = (remapped_old_matrix + feedback_matrix).tocsr()
    candidate_matrix.sum_duplicates()

    return candidate_matrix, skipped_unknown_items


# =========================
# Train candidate model
# =========================

def train_candidate_als(user_item_matrix, factors, regularization, iterations, alpha):
    """
    Train candidate ALS model.

    Important:
    Do not transpose the matrix.
    The expected shape is:
    users x items
    """

    confidence_matrix = (user_item_matrix * alpha).astype(np.float32)

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=42,
    )

    model.fit(confidence_matrix)

    return model


def build_user_content_profiles(user_item_matrix, item_content_matrix):
    """
    Build user content profiles from the updated user-item matrix.
    """

    user_content_profiles = user_item_matrix @ item_content_matrix
    user_content_profiles = normalize(user_content_profiles, norm="l2", axis=1)

    return user_content_profiles


def build_user_seen_items(user_item_matrix):
    """
    Store seen item indices for each user.
    This is used to avoid recommending already-seen items.
    """

    matrix_csr = user_item_matrix.tocsr()

    return {
        user_idx: matrix_csr.getrow(user_idx).indices.tolist()
        for user_idx in range(matrix_csr.shape[0])
    }


def build_popular_items(user_item_matrix):
    """
    Build popularity fallback list from the updated candidate matrix.
    """

    item_scores = np.asarray(user_item_matrix.sum(axis=0)).ravel()
    return np.argsort(-item_scores)


# =========================
# Validation and saving
# =========================

def validate_candidate_model(candidate_bundle, candidate_user_encoder, item_encoder):
    """
    Check that ALS, content profiles, and encoders have matching shapes.
    """

    als_model = candidate_bundle["als_model"]
    item_content_matrix = candidate_bundle["item_content_matrix"]
    user_content_profiles = candidate_bundle["user_content_profiles"]

    expected_users = len(candidate_user_encoder.classes_)
    expected_items = len(item_encoder.classes_)

    checks = {
        "ALS users": als_model.user_factors.shape[0] == expected_users,
        "ALS items": als_model.item_factors.shape[0] == expected_items,
        "Content users": user_content_profiles.shape[0] == expected_users,
        "Content items": item_content_matrix.shape[0] == expected_items,
    }

    failed_checks = [name for name, passed in checks.items() if not passed]

    if failed_checks:
        raise ValueError(f"Candidate shape validation failed: {failed_checks}")

    return True


def save_candidate_artifacts(
    candidate_dir,
    candidate_bundle,
    candidate_user_encoder,
    item_encoder,
    candidate_matrix,
    feedback_df,
    summary,
):
    """
    Save candidate model artifacts separately.

    This does not overwrite the original final model.
    """

    candidate_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(candidate_bundle, candidate_dir / "hybrid_model_bundle.pkl")
    joblib.dump(candidate_user_encoder, candidate_dir / "user_encoder.pkl")
    joblib.dump(item_encoder, candidate_dir / "item_encoder.pkl")

    save_npz(candidate_dir / "train_user_item_matrix.npz", candidate_matrix)

    feedback_df.to_csv(candidate_dir / "feedback_used.csv", index=False)

    with open(candidate_dir / "retraining_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    return candidate_dir


def save_retraining_report(summary):
    """
    Save a simple Markdown report for the retraining run.
    """

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "retraining_strategy.md"

    report = f"""# Retraining Pipeline Report

## Context

Milestone 3 deployment was cancelled, so the feedback loop was implemented locally using Streamlit.

The retraining pipeline uses:

- original training matrix
- new feedback logs from Streamlit

to create a candidate retrained model.

---

## Base Model

Base model version:

`{summary["base_model_version"]}`

The base model is not overwritten.

---

## Candidate Model

Candidate model version:

`{summary["candidate_model_version"]}`

Candidate model path:

`{summary["candidate_model_path"]}`

---

## Feedback Used

| Metric | Value |
|---|---:|
| Feedback events used | {summary["feedback_events_used"]} |
| Unique feedback users | {summary["unique_feedback_users"]} |
| Unique feedback items | {summary["unique_feedback_items"]} |
| New users added | {summary["new_users_added"]} |
| Skipped unknown feedback items | {summary["skipped_unknown_feedback_items"]} |

---

## Candidate Matrix

| Metric | Value |
|---|---:|
| Old user count | {summary["old_user_count"]} |
| Candidate user count | {summary["candidate_user_count"]} |
| Candidate item count | {summary["candidate_item_count"]} |
| Candidate nonzero interactions | {summary["candidate_nonzero_interactions"]} |

---

## Promotion Strategy

The candidate model is not automatically promoted.

It should only replace the base model after proper evaluation.

Important note:

{summary["important_note"]}
"""

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    return report_path


# =========================
# MLflow logging
# =========================

def log_retraining_to_mlflow(summary, candidate_dir, report_path):
    """
    Log retraining run to MLflow.
    """

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=summary["run_name"]):

        mlflow.set_tag("project", "Retailrocket Personalized Recommendation System")
        mlflow.set_tag("milestone", "Milestone 4 - Retraining Pipeline")
        mlflow.set_tag("run_type", "candidate_retraining")
        mlflow.set_tag("source", "local_streamlit_feedback_logs")
        mlflow.set_tag("promotion_status", summary["promotion_status"])

        # Parameters
        mlflow.log_param("base_model_version", summary["base_model_version"])
        mlflow.log_param("candidate_model_version", summary["candidate_model_version"])
        mlflow.log_param("als_factors", summary["als_factors"])
        mlflow.log_param("als_regularization", summary["als_regularization"])
        mlflow.log_param("als_iterations", summary["als_iterations"])
        mlflow.log_param("als_alpha", summary["als_alpha"])
        mlflow.log_param("promotion_status", summary["promotion_status"])

        # Metrics
        mlflow.log_metric("feedback_events_used", summary["feedback_events_used"])
        mlflow.log_metric("unique_feedback_users", summary["unique_feedback_users"])
        mlflow.log_metric("unique_feedback_items", summary["unique_feedback_items"])
        mlflow.log_metric("new_users_added", summary["new_users_added"])
        mlflow.log_metric(
            "skipped_unknown_feedback_items",
            summary["skipped_unknown_feedback_items"],
        )
        mlflow.log_metric("old_user_count", summary["old_user_count"])
        mlflow.log_metric("candidate_user_count", summary["candidate_user_count"])
        mlflow.log_metric("candidate_item_count", summary["candidate_item_count"])
        mlflow.log_metric(
            "candidate_nonzero_interactions",
            summary["candidate_nonzero_interactions"],
        )

        # Artifacts
        mlflow.log_artifacts(str(candidate_dir))
        mlflow.log_artifact(str(report_path))


# =========================
# Main retraining pipeline
# =========================

def run_retraining_pipeline(
    min_feedback_events,
    factors,
    regularization,
    iterations,
    alpha,
):
    """
    Full local retraining pipeline.

    It creates a new candidate model version from:
    old train matrix + current feedback logs.
    """

    print("\nStarting local retraining pipeline...")

    # 1. Load original model artifacts
    hybrid_bundle, old_user_encoder, item_encoder = load_artifacts()
    validate_bundle(hybrid_bundle)

    item_content_matrix = hybrid_bundle["item_content_matrix"]
    hybrid_params = hybrid_bundle["hybrid_params"]

    # 2. Load old training matrix
    train_matrix_path = find_train_matrix_path()
    old_train_matrix = load_npz(train_matrix_path).tocsr()

    print(f"Loaded old train matrix: {train_matrix_path}")
    print(f"Old train matrix shape: {old_train_matrix.shape}")

    # 3. Load feedback
    feedback_df = load_feedback_logs(min_feedback_events)

    print(f"Loaded feedback events: {len(feedback_df)}")

    # 4. Build candidate encoder and matrix
    candidate_user_encoder = build_candidate_user_encoder(
        old_user_encoder=old_user_encoder,
        feedback_df=feedback_df,
    )

    old_user_count = len(old_user_encoder.classes_)
    candidate_user_count = len(candidate_user_encoder.classes_)
    new_users_added = candidate_user_count - old_user_count

    candidate_matrix, skipped_unknown_items = build_candidate_matrix(
        old_train_matrix=old_train_matrix,
        old_user_encoder=old_user_encoder,
        candidate_user_encoder=candidate_user_encoder,
        item_encoder=item_encoder,
        feedback_df=feedback_df,
    )

    print("\nCandidate training data summary:")
    print(f"Old users: {old_user_count}")
    print(f"Candidate users: {candidate_user_count}")
    print(f"New users added: {new_users_added}")
    print(f"Feedback events used: {len(feedback_df)}")
    print(f"Candidate matrix shape: {candidate_matrix.shape}")
    print(f"Candidate interactions: {candidate_matrix.nnz}")

    # 5. Train candidate ALS
    print("\nTraining candidate ALS model...")

    candidate_als_model = train_candidate_als(
        user_item_matrix=candidate_matrix,
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
    )

    # 6. Build candidate hybrid components
    candidate_user_content_profiles = build_user_content_profiles(
        user_item_matrix=candidate_matrix,
        item_content_matrix=item_content_matrix,
    )

    candidate_user_seen_items = build_user_seen_items(candidate_matrix)
    candidate_popular_items = build_popular_items(candidate_matrix)

    # 7. Create candidate bundle
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate_model_version = f"candidate_hybrid_als_content_v2_{timestamp}"
    candidate_dir = OUTPUT_RETRAINING_DIR / candidate_model_version

    candidate_bundle = {
        "model_name": "Hybrid ALS + Content - Candidate Retrained Model",
        "model_type": "hybrid_recommender_candidate",
        "als_model": candidate_als_model,
        "item_content_matrix": item_content_matrix,
        "user_content_profiles": candidate_user_content_profiles,
        "user_seen_items_idx": candidate_user_seen_items,
        "popular_items": candidate_popular_items,
        "hybrid_params": hybrid_params,
        "base_model_version": "final_hybrid_als_content_v1",
        "candidate_model_version": candidate_model_version,
        "retraining_source": "streamlit_feedback_logs",
    }

    validate_candidate_model(
        candidate_bundle=candidate_bundle,
        candidate_user_encoder=candidate_user_encoder,
        item_encoder=item_encoder,
    )

    print("Candidate model shape validation passed.")

    # 8. Prepare summary
    summary = {
        "run_name": f"retraining_{candidate_model_version}",
        "base_model_version": "final_hybrid_als_content_v1",
        "candidate_model_version": candidate_model_version,
        "candidate_model_path": str(candidate_dir),
        "feedback_events_used": int(len(feedback_df)),
        "unique_feedback_users": int(feedback_df["visitorid"].nunique()),
        "unique_feedback_items": int(feedback_df["itemid"].nunique()),
        "new_users_added": int(new_users_added),
        "skipped_unknown_feedback_items": int(skipped_unknown_items),
        "old_user_count": int(old_user_count),
        "candidate_user_count": int(candidate_user_count),
        "candidate_item_count": int(candidate_matrix.shape[1]),
        "candidate_nonzero_interactions": int(candidate_matrix.nnz),
        "als_factors": int(factors),
        "als_regularization": float(regularization),
        "als_iterations": int(iterations),
        "als_alpha": float(alpha),
        "promotion_status": "not_promoted_pending_evaluation",
        "important_note": (
            "The original final model was not overwritten. "
            "This candidate model was saved separately and should only be promoted "
            "after proper evaluation."
        ),
    }

    # 9. Save candidate and log to MLflow
    save_candidate_artifacts(
        candidate_dir=candidate_dir,
        candidate_bundle=candidate_bundle,
        candidate_user_encoder=candidate_user_encoder,
        item_encoder=item_encoder,
        candidate_matrix=candidate_matrix,
        feedback_df=feedback_df,
        summary=summary,
    )

    report_path = save_retraining_report(summary)

    log_retraining_to_mlflow(
        summary=summary,
        candidate_dir=candidate_dir,
        report_path=report_path,
    )

    print("\nRetraining pipeline completed successfully.")
    print(f"Candidate model saved to: {candidate_dir}")
    print(f"Retraining report saved to: {report_path}")
    print("Original model was NOT overwritten.")

    return summary


# =========================
# Command-line interface
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Local retraining pipeline using Streamlit feedback logs."
    )

    parser.add_argument(
        "--min_feedback_events",
        type=int,
        default=20,
        help="Minimum feedback events required before retraining.",
    )

    parser.add_argument(
        "--factors",
        type=int,
        default=32,
        help="ALS latent factors for candidate retraining.",
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=0.05,
        help="ALS regularization.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="ALS iterations. Use small value for local demo.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=40.0,
        help="Confidence scaling factor for implicit feedback.",
    )

    args = parser.parse_args()

    run_retraining_pipeline(
        min_feedback_events=args.min_feedback_events,
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()