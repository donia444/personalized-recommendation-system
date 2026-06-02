# Personalized Recommendation System with Feedback Loop

## Project Overview

This project is an end-to-end personalized recommendation system for e-commerce user behavior data.

The system recommends relevant items to users based on implicit feedback such as product views, add-to-cart actions, and transactions. The project covers data preprocessing, feature engineering, model training, model evaluation, hybrid recommendation logic, feedback logging, retraining workflow, MLflow tracking, and a local Streamlit demo.

## Problem Statement

E-commerce platforms need to recommend relevant products to users based on their previous behavior.

Unlike explicit rating systems, this project works with implicit feedback. Users do not directly rate products, so their actions must be converted into meaningful interaction signals.

The goal is to generate Top-N item recommendations for each user.

## Dataset

The project uses Retailrocket-style e-commerce data, including:

* User-item interaction events
* Event types: view, addtocart, transaction
* Item properties
* Timestamps

Raw and processed data files are not included in this repository because they are generated files and may be large.

## Main Pipeline

The project pipeline includes:

1. Loading interaction and item property data
2. Removing duplicate events
3. Converting timestamps
4. Creating implicit interaction strength
5. Applying recency weighting
6. Filtering noisy users and rare items
7. Creating chronological train, validation, and test splits
8. Building a sparse user-item interaction matrix
9. Creating item content features using TF-IDF
10. Training multiple recommendation models
11. Evaluating models using ranking metrics
12. Selecting the final hybrid recommender
13. Logging user feedback
14. Supporting retraining workflow
15. Running a local Streamlit demo

## Models Implemented

The following recommendation approaches were implemented and compared:

* Popularity Baseline
* ALS Collaborative Filtering
* BPR
* Content-Based Recommendation
* Hybrid ALS + Content-Based Recommendation

## Final Model

The final selected model is a Hybrid ALS + Content-Based Recommender.

The hybrid model combines:

* Collaborative filtering signals from ALS
* Item similarity signals from content-based features
* Weighted rank fusion for final ranking

Final hybrid configuration:

* ALS weight: 0.7
* Content-based weight: 0.3
* Candidate generation size: 100
* Cold-start strategy: Popularity fallback

## Evaluation Metrics

The models were evaluated using Top-K ranking metrics:

* Precision@K
* Recall@K
* MAP@K
* NDCG@K
* Coverage@K

NDCG@10 was used as the primary ranking metric because it considers both recommendation relevance and item position in the ranked list.

## System Behavior

The system handles two main user cases:

### Known Users

If the user exists in the training data, the system generates personalized recommendations using the hybrid recommendation model.

### Cold-Start Users

If the user is new and has no historical interactions, the system uses a popularity-based fallback strategy.

When the user interacts with recommended items, the feedback can be logged and later used in the retraining pipeline.

## Project Structure

```text
.
├── notebooks/
│   ├── Milestone_1.ipynb
│   └── milestone_2.ipynb
│
├── reports/
│   ├── final_data_quality_leakage_report.json
│   ├── milestone_1_summary.json
│   ├── mlflow_clean_summary.json
│   ├── mlflow_run_summary.json
│   └── retraining_strategy.md
│
├── src/
│   ├── feedback_logger.py
│   ├── mlflow_tracking.py
│   ├── retraining_pipeline.py
│   └── run_local_demo.py
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Main Scripts

### `src/feedback_logger.py`

Logs user feedback such as views, add-to-cart actions, and transactions. This feedback can later be used for retraining.

### `src/retraining_pipeline.py`

Simulates the retraining workflow by using newly logged feedback and updating the recommendation pipeline.

### `src/mlflow_tracking.py`

Tracks model parameters, evaluation results, artifacts, and retraining information using MLflow.

### `src/run_local_demo.py`

Runs a local command-line demo for generating recommendations for a selected user.

### `streamlit_app.py`

Provides a simple local Streamlit interface to test the recommendation system.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/donia444/personalized-recommendation-system.git
cd personalized-recommendation-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit demo

```bash
streamlit run streamlit_app.py
```

## Notes

The repository does not include raw data, processed data, MLflow runs, or trained model artifacts.

Ignored files include:

* `data/`
* `model_artifacts/`
* `mlruns/`
* `outputs/`
* `mlflow.db`

These files are excluded to keep the repository clean and lightweight.

## Limitations

* The current demo is local and not deployed to a cloud platform.
* Large model artifacts are not included in the repository.
* Cold-start users are handled using popularity fallback.
* The retraining workflow is implemented as a local pipeline.

## Future Work

* Deploy the recommender system using FastAPI
* Add Azure cloud deployment
* Improve cold-start recommendations using item metadata
* Add automated retraining triggers
* Add monitoring dashboard
* Add Docker support
* Improve frontend design

## Skills Demonstrated

* Data preprocessing
* Implicit feedback modeling
* Recommendation systems
* Collaborative filtering
* Content-based filtering
* Hybrid recommendation
* Ranking evaluation metrics
* Feedback logging
* MLflow tracking
* Local ML demo development
