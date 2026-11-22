# Telco Customer Churn – Comparison of Five Classification Models

## 1. Problem Statement
Customer churn is costly for telecom companies, and keeping an existing customer is cheaper
than acquiring a new one. This project builds and compares five classification models that
predict whether a customer will churn, based on customer profile, service and billing data.

## 2. Dataset Description
- **Source:** Telco Customer Churn dataset (IBM sample data) from Kaggle
- **Size:** 7,043 customers, 21 columns (19 features, 1 customer ID, 1 target)
- **Target:** `Churn` (Yes/No), encoded as 1/0
- **Class distribution:** 5,174 No (73.46%) and 1,869 Yes (26.54%), so the target is imbalanced
- **Feature types:** 4 numeric-type (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`
  stored as 0/1) and 15 categorical (e.g. `Contract`, `InternetService`, `PaymentMethod`)
- **Why this dataset:** it has mixed feature types, real data-quality problems and a realistic
  class imbalance, which makes preprocessing and metric choice meaningful.

## 3. GitHub Repository
https://github.com/2026ct05021-cyber/ml-assignment-2-telco-churn

## 4. Preprocessing
- **Missing values:** `TotalCharges` was stored as text; 11 blank values were found, all with
  `tenure = 0` (new customers not yet billed), so they are imputed with 0.
- **Duplicates:** none found (0 duplicate rows).
- **Identifier:** `customerID` was dropped (it carries no predictive information).
- **Encoding:** categorical columns are one-hot encoded.
- **Scaling:** numeric columns are standardized (StandardScaler).
- **Split:** 80:20 train/test, stratified on `Churn`, `random_state=42`
  (5,634 training rows, 1,409 test rows; churn share 26.54% in both).
- **No data leakage:** all preprocessing sits inside a scikit-learn `Pipeline`, so it is fitted
  on the training data only.

## 5. Models Used
All five models use the same split and the same preprocessing.

| Model | Configuration |
|---|---|
| Logistic Regression | max_iter=1000, random_state=42 |
| Decision Tree | default parameters, random_state=42 |
| K-Nearest Neighbors | n_neighbors=5 |
| Naive Bayes | GaussianNB (default) |
| Random Forest | n_estimators=200, random_state=42 |

Metrics: Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC).

## 6. Results (test set, 1,409 customers)

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.8420 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7182 | 0.6415 | 0.4698 | 0.4786 | 0.4742 | 0.2818 |
| KNN | 0.7573 | 0.7882 | 0.5419 | 0.5535 | 0.5476 | 0.3818 |
| Naive Bayes | 0.6955 | 0.8075 | 0.4597 | 0.8396 | 0.5941 | 0.4268 |
| Random Forest | 0.7828 | 0.8196 | 0.6164 | 0.4813 | 0.5405 | 0.4064 |

![Metric comparison](comparison_chart.png)

## 7. Observations
- **Logistic Regression:** [your words – best on accuracy, AUC, precision, F1 and MCC; a
  simple linear model suits features like contract type and tenure.]
- **Decision Tree:** [weakest overall – AUC 0.6415, MCC 0.2818; likely overfits.]
- **KNN:** [middle of the pack – AUC 0.7882; one-hot features dilute distances.]
- **Naive Bayes:** [highest recall 0.8396 but lowest accuracy 0.6955 and precision 0.4597 –
  it flags many customers as churners; useful if missing a churner costs more.]
- **Random Forest:** [second-best AUC 0.8196 and good precision 0.6164, but recall only 0.4813.]
- **Overall:** [accuracy alone would be misleading with 26.5% churn; Logistic Regression is the
  best balanced model; Naive Bayes is preferable if recall is the priority.]

## 8. How to Run
1. Clone the repository and open the folder.
2. `pip install -r requirements.txt`
3. Run `jupyter notebook` and open `ML_Assignment2.ipynb`, then Kernel → Restart & Run All.
   The dataset is in `data/telco_churn.csv`; the held-out test set is `test_data.csv`.

## 9. Repository Structure