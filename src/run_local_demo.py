from pathlib import Path
import argparse
import json
import time

import joblib
import numpy as np


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model_artifacts"
DEMO_LOG_DIR = PROJECT_ROOT / "data" / "simulated_logs"


# =========================
# Load and validate model artifacts
# =========================

def load_artifacts():
    """
    Load the saved final model artifacts.

    This function does not train anything.
    It only loads the already saved model files.
    """

    required_files = {
        "hybrid_bundle": MODEL_DIR / "hybrid_model_bundle.pkl",
        "user_encoder": MODEL_DIR / "user_encoder.pkl",
        "item_encoder": MODEL_DIR / "item_encoder.pkl",
    }

    missing_files = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required artifact files:\n"
            + "\n".join(missing_files)
        )

    hybrid_bundle = joblib.load(required_files["hybrid_bundle"])
    user_encoder = joblib.load(required_files["user_encoder"])
    item_encoder = joblib.load(required_files["item_encoder"])

    return hybrid_bundle, user_encoder, item_encoder


def validate_bundle(hybrid_bundle):
    """
    Make sure the loaded hybrid bundle has the required components
    and that ALS and content-based matrices have matching shapes.
    """

    required_keys = [
        "model_name",
        "model_type",
        "als_model",
        "item_content_matrix",
        "user_content_profiles",
        "user_seen_items_idx",
        "popular_items",
        "hybrid_params",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in hybrid_bundle
    ]

    if missing_keys:
        raise KeyError(
            "Missing keys in hybrid bundle: "
            + ", ".join(missing_keys)
        )

    als_model = hybrid_bundle["als_model"]
    item_content_matrix = hybrid_bundle["item_content_matrix"]
    user_content_profiles = hybrid_bundle["user_content_profiles"]

    als_users = als_model.user_factors.shape[0]
    als_items = als_model.item_factors.shape[0]

    content_users = user_content_profiles.shape[0]
    content_items = item_content_matrix.shape[0]

    if als_users != content_users:
        raise ValueError("Shape mismatch: ALS users != content profile users.")

    if als_items != content_items:
        raise ValueError("Shape mismatch: ALS items != content items.")

    return True


# =========================
# Helper functions
# =========================

def convert_visitorid_type(visitorid, user_encoder):
    """
    Convert visitorid to the same type used by the saved user encoder.

    Example:
    If the encoder was trained on integer IDs, "64" becomes 64.
    """

    encoder_dtype = user_encoder.classes_.dtype

    try:
        if np.issubdtype(encoder_dtype, np.integer):
            return int(visitorid)

        if np.issubdtype(encoder_dtype, np.floating):
            return float(visitorid)

        return str(visitorid)

    except Exception:
        return visitorid


def get_seen_items(user_idx, user_seen_items_idx):
    """
    Return items already seen by the user.
    These items should not be recommended again.
    """

    return set(user_seen_items_idx.get(user_idx, []))


def get_top_items_from_scores(scores, seen_items=None, k=10):
    """
    Get the top-k item indices from a score vector.

    Seen items are removed by setting their score to negative infinity.
    """

    scores = np.asarray(scores).copy()

    if seen_items:
        valid_seen_items = [
            item_idx
            for item_idx in seen_items
            if 0 <= item_idx < len(scores)
        ]

        scores[valid_seen_items] = -np.inf

    k = min(k, len(scores))

    top_items = np.argsort(-scores)[:k]

    return top_items.tolist()


def popularity_fallback(visitorid, hybrid_bundle, item_encoder, k=10):
    """
    Recommend popular items for a new / unknown user.
    """

    popular_items = hybrid_bundle["popular_items"][:k]
    popular_item_ids = item_encoder.inverse_transform(popular_items).tolist()

    return {
        "visitorid": visitorid,
        "status": "cold_start_user",
        "strategy": "popularity_fallback",
        "recommendations_idx": popular_items.tolist()
        if hasattr(popular_items, "tolist")
        else list(popular_items),
        "recommendations_itemid": popular_item_ids,
        "seen_items_count": 0,
    }


# =========================
# Recommendation logic
# =========================

def generate_hybrid_recommendations_for_user(
    visitorid,
    hybrid_bundle,
    user_encoder,
    item_encoder,
    k=10,
):
    """
    Generate recommendations for one visitor.

    Known user:
    - Use ALS scores.
    - Use content-based scores.
    - Combine both using weighted rank fusion.

    Unknown user:
    - Use popularity fallback.
    """

    visitorid = convert_visitorid_type(visitorid, user_encoder)

    known_users = set(user_encoder.classes_)

    if visitorid not in known_users:
        return popularity_fallback(
            visitorid=visitorid,
            hybrid_bundle=hybrid_bundle,
            item_encoder=item_encoder,
            k=k,
        )

    # Get model components
    als_model = hybrid_bundle["als_model"]
    item_content_matrix = hybrid_bundle["item_content_matrix"]
    user_content_profiles = hybrid_bundle["user_content_profiles"]
    user_seen_items_idx = hybrid_bundle["user_seen_items_idx"]
    hybrid_params = hybrid_bundle["hybrid_params"]

    als_weight = hybrid_params["als_weight"]
    content_weight = hybrid_params["content_weight"]
    candidate_k = hybrid_params["candidate_k"]

    # Convert original visitorid to internal user index
    user_idx = int(user_encoder.transform([visitorid])[0])
    seen_items = get_seen_items(user_idx, user_seen_items_idx)

    # 1. ALS candidate items
    als_scores = als_model.user_factors[user_idx] @ als_model.item_factors.T

    als_top_items = get_top_items_from_scores(
        scores=als_scores,
        seen_items=seen_items,
        k=candidate_k,
    )

    # 2. Content-based candidate items
    content_scores = user_content_profiles[user_idx] @ item_content_matrix.T
    content_scores = np.asarray(content_scores.toarray()).ravel()

    content_top_items = get_top_items_from_scores(
        scores=content_scores,
        seen_items=seen_items,
        k=candidate_k,
    )

    # 3. Weighted rank fusion
    hybrid_scores = {}

    for rank, item_idx in enumerate(als_top_items, start=1):
        hybrid_scores[item_idx] = hybrid_scores.get(item_idx, 0) + als_weight / rank

    for rank, item_idx in enumerate(content_top_items, start=1):
        hybrid_scores[item_idx] = hybrid_scores.get(item_idx, 0) + content_weight / rank

    final_item_indices = [
        item_idx
        for item_idx, score in sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    ][:k]

    final_item_ids = item_encoder.inverse_transform(final_item_indices).tolist()

    return {
        "visitorid": visitorid,
        "user_idx": user_idx,
        "status": "success",
        "strategy": "hybrid_als_content_rank_fusion",
        "recommendations_idx": final_item_indices,
        "recommendations_itemid": final_item_ids,
        "als_candidates_idx": als_top_items[:k],
        "content_candidates_idx": content_top_items[:k],
        "seen_items_count": len(seen_items),
    }


# =========================
# Output helpers
# =========================

def print_result(result, latency_ms, top_k):
    """
    Print recommendations in a clear terminal format.
    """

    print("\n" + "=" * 70)
    print("LOCAL RECOMMENDER DEMO")
    print("=" * 70)

    print(f"Visitor ID: {result['visitorid']}")
    print(f"Status: {result['status']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Top K: {top_k}")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Seen items count: {result.get('seen_items_count', 0)}")

    print("\nRecommended item IDs:")

    for rank, itemid in enumerate(result["recommendations_itemid"], start=1):
        print(f"{rank}. {itemid}")

    print("=" * 70)


def save_demo_log(result, latency_ms, top_k):
    """
    Save the latest local demo result as a JSON file.

    This is only a demo log.
    Retraining uses feedback_logs.csv, not this file.
    """

    DEMO_LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "visitorid": str(result["visitorid"]),
        "status": result["status"],
        "strategy": result["strategy"],
        "top_k": top_k,
        "latency_ms": latency_ms,
        "recommended_items": result["recommendations_itemid"],
    }

    log_path = DEMO_LOG_DIR / "local_demo_last_result.json"

    with open(log_path, "w", encoding="utf-8") as file:
        json.dump(log_record, file, indent=4)

    print(f"\nDemo log saved to: {log_path}")


# =========================
# Command-line demo
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Run local inference for the final hybrid recommender model."
    )

    parser.add_argument(
        "--visitorid",
        type=str,
        default=None,
        help="Original visitorid. If empty, a sample known user is used.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of recommendations to return.",
    )

    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only check that the model artifacts load correctly.",
    )

    args = parser.parse_args()

    hybrid_bundle, user_encoder, item_encoder = load_artifacts()
    validate_bundle(hybrid_bundle)

    print("Model artifacts loaded successfully.")
    print(f"Model name: {hybrid_bundle['model_name']}")
    print(f"Model type: {hybrid_bundle['model_type']}")
    print(f"Hybrid params: {hybrid_bundle['hybrid_params']}")

    if args.check_only:
        print("\nCheck passed. The local demo is ready.")
        return

    if args.visitorid is None:
        visitorid = user_encoder.classes_[0]
        print(f"\nNo visitorid provided. Using sample known user: {visitorid}")
    else:
        visitorid = args.visitorid

    start_time = time.perf_counter()

    result = generate_hybrid_recommendations_for_user(
        visitorid=visitorid,
        hybrid_bundle=hybrid_bundle,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        k=args.top_k,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    print_result(result, latency_ms, args.top_k)
    save_demo_log(result, latency_ms, args.top_k)


if __name__ == "__main__":
    main()