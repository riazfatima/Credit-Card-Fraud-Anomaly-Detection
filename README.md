# Credit-Card-Fraud-Anomaly-Detection Pipeline: Anomaly Modeling & Risk Mitigation

This repository contains an end-to-end machine learning pipeline developed to address extreme class imbalance in financial transactions. The project processes credit card transaction records, applies synthetic oversampling techniques, evaluates competing statistical models, and renders an interactive diagnostic dashboard for performance analysis.

## Project Architecture & Methodology

### 1. Data Profile & The Class Imbalance Challenge
The dataset comprises **284,807 transactions**, presenting an extreme class imbalance typical of real-world fraud detection scenarios:
* **Normal Transactions (Class 0):** 284,315 ($99.827\%$)
* **Fraudulent Transactions (Class 1):** 492 ($0.173\%$)

To prevent the models from favoring the majority class, the data was partitioned using a **80/20 Train-Test Split**:
* **Training Set:** 213,605 samples (369 Fraudulent)
* **Testing Set:** 71,202 samples (123 Fraudulent)
## Data Source & Accessibility

* **Source:** The model utilizes the standard, real-world benchmark **Credit Card Fraud Detection Dataset** originally collected during a research collaboration between Université Libre de Bruxelles (ULB) and Worldline.
* **Access:** The dataset can be downloaded directly from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
* **Data Dimensions:** The source file `creditcard.csv` contains transactions made by European cardholders over a two-day period, featuring 28 numerical features generated via Principal Component Analysis (PCA) transformations ($V1$ through $V28$), alongside `Time`, `Amount`, and the binary target variable `Class`.

### 2. Algorithmic Oversampling (SMOTE)
To ensure robust boundary delineation during model training, **Synthetic Minority Over-sampling Technique (SMOTE)** was applied exclusively to the training partition. This synthetically expanded the minority pool from **369** to **213,236** balanced samples, mapping perfectly to the majority class size and eliminating algorithmic bias toward normal transactions.

---

## Comparative Performance Evaluation

Two separate pipeline pipelines—**Logistic Regression** and **Random Forest**—were evaluated based on operational transaction volume impacts.

### Actual Transaction Classification (Test Partition = 71,202 Samples)

| Model Pipeline | True Negatives (Allowed) | False Positives (False Alarms) | False Negatives (Missed Fraud) | True Positives (Caught Fraud) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 69,363 | 1,716 | **14** | 109 |
| **Random Forest** | 71,064 | **15** | 22 | 101 |

### Statistical Breakdown

#### Logistic Regression
* **Accuracy:** $98\%$
* **Fraud Precision:** $0.06$ (High rate of false alarms)
* **Fraud Recall:** $0.89$ (Caught $89\%$ of actual fraud)
* **Macro Average F1-Score:** $0.55$

#### Random Forest
* **Accuracy:** $100\%$
* **Fraud Precision:** $0.87$ (Extremely clean alerts)
* **Fraud Recall:** $0.82$ (Caught $82\%$ of actual fraud)
* **Macro Average F1-Score:** $0.92$

---

## Core Analytical Insights & Business Implications

When deploying a fraud model into a live banking infrastructure, the choice between models represents a classic trade-off between **risk tolerance** and **operational overhead**:

### The Operational Overhead Trade-off (Logistic Regression)
Logistic Regression achieved a high fraud recall ($89\%$), missing only 14 fraudulent transactions. However, this came at the expense of generating **1,716 False Positives**. In a real-world banking operations center, this translates to 1,716 legitimate customer cards being frozen or flagged unnecessarily, creating heavy customer service friction and high manual review costs.

### The Precision Efficiency Model (Random Forest)
Random Forest proved to be exceptionally precise, generating only **15 False Positives** across more than 71,000 transactions. This drastically minimizes operational overhead and preserves customer trust. The trade-off is a slight increase in risk exposure: it missed 22 fraudulent transactions compared to Logistic Regression's 14 (a recall of $82\%$).

---

## Visualizations and Diagnostics
The pipeline generates and saves three critical statistical evaluation assets to the project root directory:
1.  `04_confusion_matrices.png`: Dual confusion matrices displaying raw transaction counts.
2.  `05_roc_curves.png`: ROC-AUC discrimination curves measuring sensitivity vs. specificity thresholds.
3.  `06_metrics_comparison.png`: A comparative chart mapping core operational metrics side by side.

An interactive diagnostic dashboard window automatically launches post-execution to allow deep-dive explorations of these performance boundaries.

## How to Run the Pipeline
1. Clone the repository.
2. Activate your virtual environment and install the required dependencies.
3. Execute the core module:
   ```bash
   python "Credit card fraud.py"
