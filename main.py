# ==============================
# Client Product Recommendation
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import ttest_ind

# ------------------------------
# 1. Load Data
# ------------------------------
df = pd.read_csv("data/client_data.csv")

print("Initial Data Shape:", df.shape)
print(df.head())

# ------------------------------
# 2. Data Cleaning
# ------------------------------
df.drop_duplicates(inplace=True)
df.fillna(df.median(), inplace=True)

# ------------------------------
# 3. Feature / Target Split
# ------------------------------
X = df.drop(columns=["client_id", "product_interested"])
y = df["product_interested"]

# ------------------------------
# 4. Hypothesis Testing
# Hypothesis: Clients with higher income are more likely
# to be interested in the product
# ------------------------------
interested = df[df["product_interested"] == 1]["income"]
not_interested = df[df["product_interested"] == 0]["income"]

t_stat, p_value = ttest_ind(interested, not_interested)

print("\nHypothesis Test (Income vs Interest)")
print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Result: Statistically significant difference ✅")
else:
    print("Result: No significant difference ❌")

# ------------------------------
# 5. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ------------------------------
# 6. Feature Scaling
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 7. Model Training
# ------------------------------

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        eval_metric="logloss"
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probas = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, probas)
    results[name] = auc

    print(f"\n{name}")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", auc)

# ------------------------------
# 8. Select Best Model
# ------------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

# Save model
joblib.dump(best_model, "model/trained_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# ------------------------------
# 9. SHAP Explainability
# ------------------------------
explainer = shap.Explainer(best_model, X_train_scaled)
shap_values = explainer(X_test_scaled)

shap.summary_plot(shap_values, X_test, show=False)
plt.title("Feature Importance (SHAP)")
plt.tight_layout()
plt.show()

# ------------------------------
# 10. Market Insight Generation
# ------------------------------
def generate_market_insights(model, data, feature_names):
    importances = model.feature_importances_
    insights = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    top_features = insights.head(3)

    insights_text = "Market Insights:\n"
    for _, row in top_features.iterrows():
        insights_text += f"- {row['Feature']} strongly influences client interest.\n"

    return insights_text


if hasattr(best_model, "feature_importances_"):
    insights = generate_market_insights(
        best_model,
        X_train_scaled,
        X.columns
    )
    print("\n" + insights)

print("\nPipeline executed successfully ✅")
