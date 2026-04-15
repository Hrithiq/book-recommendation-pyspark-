# Book Recommendation System using PySpark

A scalable **Book Recommendation System** built using **PySpark** that leverages collaborative filtering techniques to generate personalized recommendations from large-scale datasets.

---

## Overview

With thousands of books available online, discovering relevant content can be overwhelming. This project builds a **distributed recommendation system** using the **Goodbooks-10K dataset**, implementing and comparing multiple algorithms:

- Alternating Least Squares (ALS)
- Singular Value Decomposition (SVD)
- k-Nearest Neighbors (kNN)

The system is designed to handle **millions of ratings efficiently** using PySpark’s distributed computing capabilities.

---

## Objectives

- Preprocess large-scale recommendation datasets
- Implement collaborative filtering algorithms
- Evaluate models using:
  - RMSE (prediction accuracy)
  - Precision@K
  - Recall@K
  - NDCG@K
- Generate personalized Top-K recommendations
- Demonstrate scalability using PySpark

---

## Dataset

- **Source:** Goodbooks-10K (Kaggle)
- ~6 million ratings
- 53,000+ users
- 10,000 books
- Highly sparse (~99.9%)

---

## Tech Stack

- **PySpark (MLlib)**
- Python
- NumPy / Pandas
- Hadoop HDFS (optional)

---

## Methodology

### 1. Data Preprocessing
- Duplicate removal
- Missing value imputation
- Feature engineering (user & book stats)
- Encoding categorical features
- Normalization and vector assembly

### 2. Models Implemented

#### ALS (Best Performing)
- Matrix factorization using Spark MLlib
- Hyperparameter tuning via Cross Validation

#### SVD
- Distributed matrix decomposition using RowMatrix

#### kNN
- Item-based similarity using latent factors

---

## Results

| Model | RMSE | Precision@10 | Recall@10 | NDCG@10 |
|------|------|-------------|----------|--------|
| ALS  | 0.879 | 0.342 | 0.187 | 0.412 |
| SVD  | 0.891 | 0.318 | 0.174 | 0.398 |
| kNN  | —     | 0.285 | 0.156 | 0.361 |

### Key Insight:
**ALS performs best** due to:
- Regularization
- Iterative optimization
- Better handling of sparse data

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/book-recommendation-pyspark.git
cd book-recommendation-pyspark
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the project
```bash
spark-submit --master local[*] scripts/recommender.py
```

## Project Structure
scripts/
    recommender.py    # Main pipeline

README.md
requirements.txt

## Features
- Distributed processing with PySpark
- Scalable for millions of users/items
- Multiple recommendation algorithms
- Comprehensive evaluation metrics
- Modular pipeline design

## Limitations
- Cold-start problem (new users/items)
- Popularity bias
- No temporal dynamics
- Assumes explicit ratings only

## Future Work
- Hybrid recommendation systems
- Deep learning-based recommenders
- Context-aware recommendations
- Explainable AI for recommendations
- Real-time recommendation systems

## Authors
- Hrithiq Gupta (230962300)
- Arnav Sahu (230962306)

Under the guidance of:
**Dr. Anup Bhat B**
Assistant Professor
Manipal Institute of Technology

## License

This project is for academic purposes.

---

## Final Steps (Git Commands)

```bash
git init
git add .
git commit -m "Initial commit - Book Recommendation System (PySpark)"
git branch -M main
git remote add origin https://github.com/your-username/book-recommendation-pyspark.git
git push -u origin main
```
