from pathlib import Path
import time
import pandas as pd


# =========================
# Paths and constants
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback_logs"
FEEDBACK_LOG_PATH = FEEDBACK_DIR / "feedback_logs.csv"

FEEDBACK_COLUMNS = [
    "timestamp",
    "visitorid",
    "itemid",
    "event",
    "weight",
    "model_version",
    "strategy",
    "recommendation_rank",
    "source",
]

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 10.0,
}

RETRAINING_THRESHOLD = 20


# =========================
# File setup
# =========================

def ensure_feedback_log_exists():
    """
    Create feedback_logs.csv if it does not already exist.
    """

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    if not FEEDBACK_LOG_PATH.exists():
        pd.DataFrame(columns=FEEDBACK_COLUMNS).to_csv(
            FEEDBACK_LOG_PATH,
            index=False
        )


# =========================
# Save feedback
# =========================

def save_feedback_event(
    visitorid,
    itemid,
    event,
    model_version="final_hybrid_als_content_v1",
    strategy=None,
    recommendation_rank=None,
    source="streamlit_local_demo",
):
    """
    Save one user interaction event.

    This is real feedback, not just a recommendation.
    The retraining pipeline can later use this feedback as new interaction data.
    """

    ensure_feedback_log_exists()

    if event not in EVENT_WEIGHTS:
        raise ValueError(
            f"Invalid event '{event}'. Allowed events: {list(EVENT_WEIGHTS.keys())}"
        )

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "visitorid": str(visitorid),
        "itemid": str(itemid),
        "event": event,
        "weight": EVENT_WEIGHTS[event],
        "model_version": model_version,
        "strategy": strategy,
        "recommendation_rank": recommendation_rank,
        "source": source,
    }

    pd.DataFrame([record]).to_csv(
        FEEDBACK_LOG_PATH,
        mode="a",
        header=False,
        index=False
    )

    return record


# =========================
# Load and summarize feedback
# =========================

def load_feedback_logs():
    """
    Load all saved feedback events.
    """

    ensure_feedback_log_exists()
    return pd.read_csv(FEEDBACK_LOG_PATH)


def summarize_feedback_logs():
    """
    Return a simple summary of collected feedback.
    """

    logs_df = load_feedback_logs()

    if logs_df.empty:
        return {
            "total_feedback_events": 0,
            "unique_users": 0,
            "unique_items": 0,
            "event_distribution": {},
            "total_feedback_weight": 0.0,
            "ready_for_retraining": False,
        }

    return {
        "total_feedback_events": int(len(logs_df)),
        "unique_users": int(logs_df["visitorid"].nunique()),
        "unique_items": int(logs_df["itemid"].nunique()),
        "event_distribution": logs_df["event"].value_counts().to_dict(),
        "total_feedback_weight": float(logs_df["weight"].sum()),
        "ready_for_retraining": bool(len(logs_df) >= RETRAINING_THRESHOLD),
    }