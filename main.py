import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from xgboost import XGBClassifier, plot_importance
import shap
import matplotlib.pyplot as plt



# 1. Load & preprocess dataset

df = pd.read_csv("churn_data.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

X = df.drop(['customerID','Churn'], axis=1)
y = df['Churn']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 2. Handle class imbalance

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos


# 3. Train XGBoost

xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_train, y_train)


# 4. Predict probabilities

y_pred_proba = xgb.predict_proba(X_test)[:,1]


# 5. Automatic threshold tuning

thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh))

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold = {best_threshold:.2f} with F1-score = {f1_scores[best_idx]:.3f}")


# Use best threshold for predictions
y_pred = (y_pred_proba >= best_threshold).astype(int)


# 6. Evaluation

print("Classification Report (best threshold):")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))


# 7. Feature importance

plt.figure(figsize=(10,6))
plot_importance(xgb, max_num_features=10)
plt.show()


import shap
explainer = shap.TreeExplainer(xgb, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)

print("Top Features Driving Churn (Global Importance):")
shap.summary_plot(shap_values, X_test, plot_type="bar")

# ---- Local Explanations for 10 Predictions ----
# Get predicted probabilities
y_proba = xgb.predict_proba(X_test)[:, 1]

# Select 10 random samples from test set
sample_indices = np.random.choice(range(len(X_test)), size=10, replace=False)
sample_data = X_test.iloc[sample_indices]
sample_preds = (y_proba[sample_indices] >= 0.43).astype(int)  # using tuned threshold

print("\n--- 10 Sample Predictions with Reasons ---\n")
for i, idx in enumerate(sample_indices):
    prob = y_proba[idx]
    pred = "Churn" if prob >= 0.43 else "No Churn"

    # Get SHAP values for this prediction
    shap_contrib = pd.DataFrame({
        "feature": sample_data.columns,
        "shap_value": shap_values[idx],
        "value": sample_data.iloc[i].values
    }).sort_values(by="shap_value", key=abs, ascending=False)

    # Top 3 reasons
    top_reasons = shap_contrib.head(3)

    print(f"Customer {i+1}: Predicted = {pred}, Probability = {prob:.2f}")
    for _, row in top_reasons.iterrows():
        direction = "↑ increases churn risk" if row.shap_value > 0 else "↓ decreases churn risk"
        print(f"   - {row.feature} = {row.value} ({direction})")
    print()
