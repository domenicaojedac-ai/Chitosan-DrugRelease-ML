# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error

# ==========================
# LOAD DATA
# ==========================
df = pd.read_excel(r"/content/Data_base9.2_EE.xlsx")
print("Shape original:", df.shape)

# ==========================
# TARGET
# ==========================
target = "EE_Percent"

# Asegurar que target sea numérico
df[target] = pd.to_numeric(df[target], errors="coerce")

# ==========================
# DROP ID COLUMNS
# ==========================
id_cols = ["Paper_ID", "Form_ID"]
df = df.drop(columns=[col for col in id_cols if col in df.columns])

# ==========================
# COLUMNS TO EXCLUDE
# ==========================
exclude_cols = ['Drug_Found', 'Drug_content', 'Drug_loading']
df = df.drop(columns=[col for col in exclude_cols if col in df.columns])

# ==========================
# COLUMNS THAT SHOULD BE NUMERIC
# ==========================
maybe_numeric = [
    "Drug_Mass_Initial_mg",
    "Theorical_loading",
    "Crosslinker_Conc_percent",
    "CS_Crosslinker_Ratio",
    "CS_MW_kDa",
    "CS_DD_percent",
    "CS_Conc_wv_percent",
    "Drug_MW_g_mol",
    "Drug_LogP",
    "Drug_HBD",
    "Drug_HBA",
    "Drug_pKa",
    "Matrix_Mass"
]

for col in maybe_numeric:
    if col in df.columns:
        # Limpieza básica por si vienen como texto con espacios o comas
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace(["nan", "None", ""], np.nan)
        )
        # cambia coma decimal por punto si hiciera falta
        df[col] = df[col].str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================
# REMOVE ROWS WITHOUT TARGET
# ==========================
df = df.dropna(subset=[target])
print("Shape after dropping NaN target:", df.shape)

# ==========================
# FEATURES / TARGET
# ==========================
X = df.drop(columns=[target])
y = df[target]

# ==========================
# DETECT TYPES AGAIN
# ==========================
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Ensure all categorical columns are of string type for consistent imputation
for col in cat_cols:
    X[col] = X[col].astype(str)

print("\nNumerical columns:")
print(num_cols)

print("\nCategorical columns:")
print(cat_cols)

# ==========================
# PREPROCESSING
# ==========================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ==========================
# TRAIN / TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================
# MODELS
# ==========================
models = {
    "ElasticNet": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=10000
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
}

# ==========================
# TRAIN + EVALUATE
# ==========================
results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {"R2": r2, "RMSE": rmse}
    fitted_pipelines[name] = pipe

    print(f"\n{name}")
    print(f"R2   = {r2:.4f}")
    print(f"RMSE = {rmse:.4f}")

# ==========================
# BEST MODEL
# ==========================
best_model_name = max(results, key=lambda k: results[k]["R2"])
best_pipe = fitted_pipelines[best_model_name]

print("\nBest model:", best_model_name)
print("Metrics:", results[best_model_name])

# ==========================
# SAVE BEST MODEL
# ==========================
joblib.dump(best_pipe, "best_EE_model.pkl")
print("\nSaved as: best_EE_model.pkl")

"""## Model Evaluation Plots"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Recalculate predictions for the best model to ensure consistency
y_pred_best = best_pipe.predict(X_test)

"""### Parity Plot (True vs. Predicted Values)

This plot compares the actual values (`y_test`) against the model's predicted values (`y_pred`). A perfect model would show all points lying on the `y=x` line. The closer the points are to this line, the better the model's predictive accuracy.
"""

plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values (EE_Percent)')
plt.ylabel('Predicted Values (EE_Percent)')
plt.title('True vs. Predicted Values (XGBoost)')
plt.show()

"""### Residual Plot (Predicted Values vs. Residuals)

This plot shows the residuals (the difference between true and predicted values) against the predicted values. Ideally, residuals should be randomly scattered around zero, indicating that the model captures the underlying pattern well and there are no systematic errors. Any discernible pattern (e.g., a curve, a funnel shape) would suggest issues with the model assumptions or structure.
"""

residuals = y_test - y_pred_best

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_best, y=residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Values (EE_Percent)')
plt.ylabel('Residuals (True - Predicted)')
plt.title('Residual Plot (XGBoost)')
plt.show()

"""### Feature Importance Plot

This plot illustrates the importance of each feature in the XGBoost model. Features with higher importance contribute more significantly to the model's predictions. This can help in understanding which input variables are most influential in determining the `EE_Percent`.
"""

# Get feature importances from the best model (XGBoost)
model = best_pipe.named_steps['model']
feature_importances = model.feature_importances_

# Get feature names after preprocessing (OneHotEncoder for categorical, passthrough for numerical)
preprocessor = best_pipe.named_steps['preprocessor']
feature_names_out = preprocessor.get_feature_names_out()

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names_out,
    'Importance': feature_importances
})

# Sort by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importance (top 20 for better visualization if many features)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.tight_layout()
plt.show()