# Retraining Pipeline Report

## Context

Milestone 3 deployment was cancelled, so the feedback loop was implemented locally using Streamlit.

The retraining pipeline uses:

- original training matrix
- new feedback logs from Streamlit

to create a candidate retrained model.

---

## Base Model

Base model version:

`final_hybrid_als_content_v1`

The base model is not overwritten.

---

## Candidate Model

Candidate model version:

`candidate_hybrid_als_content_v2_20260520_204248`

Candidate model path:

`C:\Users\pc\Music\Final_recommender_system\outputs\retraining\candidate_hybrid_als_content_v2_20260520_204248`

---

## Feedback Used

| Metric | Value |
|---|---:|
| Feedback events used | 42 |
| Unique feedback users | 13 |
| Unique feedback items | 22 |
| New users added | 11 |
| Skipped unknown feedback items | 0 |

---

## Candidate Matrix

| Metric | Value |
|---|---:|
| Old user count | 22890 |
| Candidate user count | 22901 |
| Candidate item count | 18224 |
| Candidate nonzero interactions | 155083 |

---

## Promotion Strategy

The candidate model is not automatically promoted.

It should only replace the base model after proper evaluation.

Important note:

The original final model was not overwritten. This candidate model was saved separately and should only be promoted after proper evaluation.
