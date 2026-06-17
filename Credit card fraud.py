"""
CREDIT CARD FRAUD DETECTION PROJECT
====================================
Dataset: 284,807 transactions with 492 frauds (0.17% imbalanced)
Models: Logistic Regression (LR) and Random Forest (RF)
Balancing: SMOTE (Synthetic Minority Over-sampling Technique)
Pipeline: Imblearn Leak-Free Pipeline Structure WITH COMPLETE NUMERICAL SUMMARIES
"""

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================
print("="*80)
print("STEP 1: IMPORTING REQUIRED LIBRARIES")
print("="*80)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, roc_auc_score, 
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
print("✓ Libraries imported successfully!\n")

# ============================================================================
# LOAD DATASET & BEFORE-MODELING ANALYSIS
# ============================================================================
print("="*80)
print("STEP 2: RAW NUMBERS BEFORE DATA MODELING (FULL DATASET)")
print("="*80)

df = pd.read_csv(r"c:\Users\NEW LAPTOP CITY\.vscode\Internship Projects\Project 2\creditcard.csv")

total_raw = len(df)
normal_raw = (df['Class'] == 0).sum()
fraud_raw = (df['Class'] == 1).sum()

print(f"Total Transactions in CSV:          {total_raw:,}")
print(f"  • Normal Transactions (Class 0):  {normal_raw:,} ({(normal_raw/total_raw)*100:.3f}%)")
print(f"  • Fraudulent Transactions (Class 1): {fraud_raw:,} ({(fraud_raw/total_raw)*100:.3f}%)")

# ============================================================================
# DATA PREPARATION AND SPLIT TRACKING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: TRAIN-TEST SPLIT SUBSET DISTRIBUTION NUMBERS")
print("="*80)

X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print(f"Training Set Partition (75%):       {len(y_train):,} samples")
print(f"  • Train Normal (Class 0):         {(y_train==0).sum():,}")
print(f"  • Train Fraud (Class 1):          {(y_train==1).sum():,}")
print(f"Testing Set Partition (25%):        {len(y_test):,} samples")
print(f"  • Test Normal (Class 0):          {(y_test==0).sum():,}")
print(f"  • Test Fraud (Class 1):           {(y_test==1).sum():,}")

# ============================================================================
# TRAIN PIPELINES AND EXECUTE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: MODEL TRAINING (SMOTE APPLIED LOGICALLY)")
print("="*80)

print(f"SMOTE target generation blueprint:")
print(f"  • Input Minority Sample Pool:     {(y_train==1).sum()}")
print(f"  • Balanced Output Minority Pool:  {(y_train==0).sum():,}")

pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        # Reduce SMOTE intensity: target a 10% minority ratio instead of a forced 50/50 balance
        ('smote', SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.1)),
        # Use class_weight='balanced' to naturally penalize remaining imbalances
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs', class_weight='balanced'))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        # Keep Random Forest as is, or give it a 0.3 sampling blueprint
        ('smote', SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.3)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20, n_jobs=-1))
    ])
}

predictions_store = {}
for name, pipeline in pipelines.items():
    print(f"Processing {name} Pipeline Asset...")
    pipeline.fit(X_train, y_train)
    predictions_store[name] = {
        'predictions': pipeline.predict(X_test),
        'probabilities': pipeline.predict_proba(X_test)[:, 1]
    }

# ============================================================================
# AFTER-MODELING COMPLETE NUMERICAL VALUE RESULTS TABLE
# ============================================================================
print("\n" + "="*80)
print("STEP 5: FINAL RESULTS TABLE VALUES (ACTUAL TRANSACTION NUMBERS)")
print("="*80)

summary_rows = []
for name, data in predictions_store.items():
    y_pred = data['predictions']
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    summary_rows.append({
        'Model Pipeline': name,
        'True Neg (Allowed)': f"{tn:,}",
        'False Pos (Alarms)': f"{fp:,}",
        'False Neg (Missed)': f"{fn:,}",
        'True Pos (Caught)': f"{tp:,}"
    })

summary_table_df = pd.DataFrame(summary_rows)
print(summary_table_df.to_string(index=False))
print("="*80)

# ============================================================================
# STEP 6 & 7: COMPILER VISUALIZATIONS GENERATOR (CONFUSION MATRICES)
# ============================================================================
print("\n" + "="*80)
print("STEP 7: COMPILING DUAL CONFUSION MATRICES")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for idx, (name, data) in enumerate(predictions_store.items()):
    y_pred = data['predictions']
    cm = confusion_matrix(y_test, y_pred)
    
    group_names = ['True Neg\n(Allowed)', 'False Pos\n(Alarm)', 'False Neg\n(Missed)', 'True Pos\n(Caught)']
    group_counts = [f"{value:,}" for value in cm.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=axes[idx], cbar=False, 
                annot_kws={'size': 11, 'fontweight': 'bold'}, square=True)
    axes[idx].set_title(f'{name} Counts', fontweight='bold')
    axes[idx].set_ylabel('Actual Classes')
    axes[idx].set_xlabel('Predicted Classes')
    axes[idx].set_xticklabels(['Normal', 'Fraud'])
    axes[idx].set_yticklabels(['Normal', 'Fraud'])

plt.tight_layout()
plt.savefig('04_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved: 04_confusion_matrices.png")

# ============================================================================
# VISUALIZATION 5: ROC-AUC PERFORMANCE CURVES
# ============================================================================
print("\n" + "="*80)
print("STEP 8: PLOTTING ROC-AUC DISCRIMINATION CURVES")
print("="*80)

plt.figure(figsize=(8, 6))
model_colors = ['#e67e22', '#9b59b6']

for idx, (name, data) in enumerate(predictions_store.items()):
    y_proba = data['probabilities']
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=model_colors[idx], lw=2.5, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Baseline (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Type I Error Index)', fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontweight='bold')
plt.title('Receiver Operating Characteristic (ROC) Comparison Curve', fontsize=13, fontweight='bold')
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig('05_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved: 05_roc_curves.png")

# ============================================================================
# VISUALIZATION 6: STRICT OPERATIONAL METRICS COMPARISON (NO ACCURACY)
# ============================================================================
print("\n" + "="*80)
print("STEP 9: COMPARING CORE OPERATIONAL METRICS")
print("="*80)

# CRITICAL FIX: Initializing the dictionary to prevent NameError crash
evaluation_results = {}

for name, data in predictions_store.items():
    y_pred = data['predictions']
    evaluation_results[name] = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }

results_df = pd.DataFrame(evaluation_results).T

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = ['Precision', 'Recall', 'F1-Score']
bar_colors = ['#e74c3c', '#3498db']

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    values = results_df[metric].values
    bars = ax.bar(range(len(values)), values, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_title(f'Operational {metric} Metrics', fontweight='bold', fontsize=11)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(results_df.index, fontweight='bold')
    ax.set_ylim([0, 1.15])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Strict Final Performance Evaluation (Excluding Misleading Global Accuracy)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('06_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved: 06_metrics_comparison.png")

# Print clean terminal summaries
print("\n" + "="*80)
for name, data in predictions_store.items():
    print(f"\n{name} Final Statistical Breakdown:")
    print(classification_report(y_test, data['predictions'], target_names=['Normal', 'Fraud']))

# Single UI invocation allows all generated windows to load cleanly together
print("\nRendering visualization dashboard windows...")
plt.show()

print("\n" + "="*80)
print("🏆 INTERACTIVE DIAGNOSTIC DASHBOARD LAUNCHED SUCCESSFULLY!")
print("="*80)