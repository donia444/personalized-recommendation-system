\# Personalized Recommendation System



An end-to-end e-commerce recommendation system using implicit user feedback.



\## Project Overview



This project builds a personalized recommender system using user-item interactions such as views, add-to-cart actions, and transactions.



The system includes:

\- Data preprocessing

\- Implicit feedback weighting

\- Chronological train/validation/test split

\- Collaborative filtering models

\- Content-based recommendation

\- Hybrid recommendation model

\- Evaluation using ranking metrics

\- Feedback logging

\- Retraining pipeline

\- Local Streamlit demo



\## Models



The project compares:

\- Popularity Baseline

\- ALS Collaborative Filtering

\- BPR

\- Content-Based Recommendation

\- Hybrid ALS + Content-Based Recommendation



\## Evaluation Metrics



Models are evaluated using:

\- Precision@K

\- Recall@K

\- MAP@K

\- NDCG@K

\- Coverage@K



\## Final Model



The final model is a Hybrid ALS + Content-Based Recommender using weighted rank fusion.



\## Project Structure



```text

.

├── notebooks/

├── reports/

├── src/

├── streamlit\_app.py

├── requirements.txt

├── README.md

└── .gitignore

