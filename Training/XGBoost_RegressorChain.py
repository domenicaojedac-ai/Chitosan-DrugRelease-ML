# train_method_1_xgb_chain.py
import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.multioutput import RegressorChain
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor


# -----------------------------
# Weibull + refined fitting
# -----------------------------
def weibull(t, A, k, b):
    t = np.clip(np.asarray(t, dtype=float), 0, None)
    arg = (t / np.clip(k, 1e-6, None))
    with np.errstate(over='ignore', invalid='ignore'):
        term = np.exp(-np.power(arg, b))
    return A * (1.0 - term)

def fit_weibull_refined(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y) & (t > 0)
    t, y = t[mask], y[mask]
    if len(t) < 5:
        return None, 0.0

    y_max = float(np.nanmax(y))
    t_max = float(np.nanmax(t))
    if not np.isfinite(y_max) or not np.isfinite(t_max) or t_max <= 0:
        return None, 0.0

    best_mse = np.inf
    best_p0 = None

    A_opts = [min(max(y_max, 1.0), 110.0), 100.0]
    k_opts = [max(t_max * 0.2, 0.05), max(t_max * 0.6, 0.05), max(t_max * 1.5, 0.05)]
    b_opts = [0.5, 0.8, 1.0, 1.5, 2.5]

    for A0 in A_opts:
        for k0 in k_opts:
            for b0 in b_opts:
                try:
                    yp = weibull(t, A0, k0, b0)
                    mse = np.mean((y - yp) ** 2)
                    if np.isfinite(mse) and mse < best_mse:
                        best_mse = mse
                        best_p0 = [A0, k0, b0]
                except Exception:
                    continue

    if best_p0 is None:
        return None, 0.0

    try:
        popt, _ = curve_fit(
            weibull, t, y,
            p0=best_p0,
            bounds=([0, 0.01, 0.1], [110, t_max * 5, 10]),
            maxfev=4000
        )
        y_pred_fit = weibull(t, *popt)
        ss_res = np.sum((y - y_pred_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        return popt, float(r2)
    except Exception:
        # fallback
        return np.asarray(best_p0, dtype=float), 0.5


# -----------------------------
# Smoothed Target Encoder
# -----------------------------
class SmoothedTargetEncoder:
    def __init__(self, col, target_cols, smoothing=10):
        self.col = col
        self.target_cols = target_cols
        self.smoothing = smoothing
        self.maps = {}
        self.global_means = {}

    def fit(self, df):
        for t_col in self.target_cols:
            self.global_means[t_col] = float(df[t_col].mean())
            stats = df.groupby(self.col)[t_col].agg(['count', 'mean'])
            numerator = (stats['count'] * stats['mean'] + self.smoothing * self.global_means[t_col])
            denominator = (stats['count'] + self.smoothing)
            self.maps[t_col] = (numerator / denominator).to_dict()
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for t_col in self.target_cols:
            new_col = f"{self.col}_enc_{t_col}"
            out[new_col] = df[self.col].map(self.maps[t_col]).fillna(self.global_means[t_col])
        return out


# -----------------------------
# Plot helpers (English outputs)
# -----------------------------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def plot_rmse_boxplot(curve_rmses, outdir, title):
    plt.figure()
    plt.boxplot(curve_rmses, vert=True)
    plt.ylabel("Curve RMSE (% release)")
    plt.title(title)
    savefig(os.path.join(outdir, "cv_curve_rmse_boxplot.png"))

def plot_parity(y_true, y_pred, name, outdir):
    plt.figure()
    plt.scatter(y_true, y_pred, s=18)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"Parity Plot: {name}")
    savefig(os.path.join(outdir, f"parity_{name.lower().replace(' ','_')}.png"))

def plot_residuals(y_true, y_pred, name, outdir):
    res = y_true - y_pred
    plt.figure()
    plt.hist(res, bins=30)
    plt.xlabel(f"Residuals (True - Pred) for {name}")
    plt.ylabel("Count")
    plt.title(f"Residual Histogram: {name}")
    savefig(os.path.join(outdir, f"residuals_{name.lower().replace(' ','_')}.png"))

def plot_example_curves(examples, outdir, filename="example_curves_true_vs_pred.png"):
    # examples: list of dicts {t, y_true, y_pred, cid}
    n = min(len(examples), 9)
    if n == 0:
        return
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 3.5 * rows))
    for i in range(n):
        ex = examples[i]
        plt.subplot(rows, cols, i + 1)
        plt.plot(ex["t"], ex["y_true"], label="True")
        plt.plot(ex["t"], ex["y_pred"], label="Predicted")
        plt.xlabel("Time (h)")
        plt.ylabel("Cumulative release (%)")
        plt.title(f"Curve {ex['cid']}")
        plt.legend()
    savefig(os.path.join(outdir, filename))

def plot_feature_importance_xgb(model_chain, feature_names, outdir, topk=25):
    # Use last estimator or first estimator importance; we’ll average available importances
    importances = []
    for est in model_chain.estimators_:
        if hasattr(est, "feature_importances_"):
            importances.append(est.feature_importances_)
    if not importances:
        return
    imp = np.mean(np.vstack(importances), axis=0)
    idx = np.argsort(imp)[::-1][:topk]
    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in idx][::-1], imp[idx][::-1])
    plt.xlabel("Mean feature importance")
    plt.title("Top Feature Importances (XGBoost, averaged across chain)")
    savefig(os.path.join(outdir, "feature_importance_top.png"))


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Data_base9.1_CURVE.xlsx")
    parser.add_argument("--out_model", type=str, default="xgboost_weibull_method1.joblib")
    parser.add_argument("--outdir", type=str, default="results_method1_xgb")
    parser.add_argument("--min_fit_r2", type=float, default=0.85)
    parser.add_argument("--kfold", type=int, default=5)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    print("=== Method 1: XGBoost + RegressorChain ===")
    df = pd.read_excel(args.data)

    req = ["Paper_ID", "Form_ID", "Time_h", "Cum_Release_percent"]
    df = df.dropna(subset=req).copy()

    cols_num = ["Time_h", "Cum_Release_percent", "CS_MW_kDa", "CS_DD_percent",
                "CS_Conc_wv_percent", "Drug_MW_g_mol", "Drug_LogP",
                "Medium_pH", "Crosslinker_Conc_percent"]

    for c in cols_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Feature engineering
    if "Medium_pH" in df.columns and "Drug_LogP" in df.columns:
        df["pH_LogP_Diff"] = (df["Medium_pH"].fillna(7) - df["Drug_LogP"].fillna(0)).abs()
    else:
        df["pH_LogP_Diff"] = np.nan

    if "Crosslinker_Conc_percent" in df.columns and "CS_Conc_wv_percent" in df.columns:
        df["Crosslinker_Density"] = df["Crosslinker_Conc_percent"] / (df["CS_Conc_wv_percent"] + 0.1)
    else:
        df["Crosslinker_Density"] = np.nan

    df["Curve_ID"] = df["Paper_ID"].astype(str) + "_" + df["Form_ID"].astype(str)

    # Fit Weibull to each curve
    rows = []
    real_curves = {}
    skipped = 0

    print(f"Fitting Weibull curves (filter: fit R2 >= {args.min_fit_r2}) ...")
    for cid, g in df.groupby("Curve_ID"):
        t = g["Time_h"].values
        y = g["Cum_Release_percent"].values
        params, r2 = fit_weibull_refined(t, y)
        if params is None or r2 < args.min_fit_r2:
            skipped += 1
            continue
        A, k, b = params
        # sanity filters
        if (not np.isfinite(A)) or (not np.isfinite(k)) or (not np.isfinite(b)):
            skipped += 1
            continue
        if k > 500 or b > 15 or b < 0.1:
            skipped += 1
            continue

        row = g.iloc[0].to_dict()
        row["Target_A"] = float(A)
        row["Target_log_k"] = float(np.log1p(k))
        row["Target_log_b"] = float(np.log1p(b))
        rows.append(row)
        real_curves[cid] = (np.asarray(t, float), np.asarray(y, float))

    df_model = pd.DataFrame(rows)
    print(f"Curves used: {len(df_model)} | skipped: {skipped}")

    # Features
    base_feats = ["CS_MW_kDa", "CS_DD_percent", "CS_Conc_wv_percent",
                  "Crosslinker_Density", "Drug_MW_g_mol", "Drug_LogP",
                  "Medium_pH", "pH_LogP_Diff",
                  "Prep_Method", "Crosslinker_Name"]
    features = [c for c in base_feats if c in df_model.columns]

    # Determine numeric/categorical
    num_cols = [c for c in features if df_model[c].dtype != 'object']
    cat_cols = [c for c in features if df_model[c].dtype == 'object']

    targets = ["Target_A", "Target_log_b", "Target_log_k"]
    Y_raw = df_model[targets].values

    # Base model
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        objective="reg:squarederror"
    )
    chain = RegressorChain(xgb, order=[0, 2, 1])

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)

    curve_rmses = []
    all_true_params = []
    all_pred_params = []
    example_curves = []

    # Per-parameter metrics
    per_param = {t: {"y_true": [], "y_pred": []} for t in targets}

    print("Running cross-validation + plots ...")
    for fold, (train_idx, test_idx) in enumerate(kf.split(df_model), start=1):
        train_df = df_model.iloc[train_idx].copy()
        test_df = df_model.iloc[test_idx].copy()

        Y_train = Y_raw[train_idx]
        Y_test = Y_raw[test_idx]

        # Target encoding for Drug_Name (if exists)
        if "Drug_Name" in train_df.columns:
            te = SmoothedTargetEncoder("Drug_Name", ["Target_log_k", "Target_log_b"], smoothing=5).fit(train_df)
            enc_cols = [f"Drug_Name_enc_{t}" for t in ["Target_log_k", "Target_log_b"]]
            X_train = pd.concat([train_df[features], te.transform(train_df)], axis=1)
            X_test = pd.concat([test_df[features], te.transform(test_df)], axis=1)
            current_num_cols = num_cols + enc_cols
        else:
            te = None
            enc_cols = []
            X_train = train_df[features].copy()
            X_test = test_df[features].copy()
            current_num_cols = num_cols

        # Preprocessor per fold
        pre = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), current_num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ], verbose_feature_names_out=False)

        X_train_p = pre.fit_transform(X_train)
        X_test_p = pre.transform(X_test)

        # Scale Y
        y_scaler = StandardScaler()
        Y_train_s = y_scaler.fit_transform(Y_train)

        chain.fit(X_train_p, Y_train_s)
        Y_pred_s = chain.predict(X_test_p)
        Y_pred = y_scaler.inverse_transform(Y_pred_s)

        # store per-param
        for j, tname in enumerate(targets):
            per_param[tname]["y_true"].extend(Y_test[:, j].tolist())
            per_param[tname]["y_pred"].extend(Y_pred[:, j].tolist())

        # curve-level RMSE + examples
        # sample up to 2 example curves per fold
        ex_added = 0
        for i, idx in enumerate(test_idx):
            cid = df_model.iloc[idx]["Curve_ID"]
            if cid not in real_curves:
                continue
            t_true, y_true = real_curves[cid]

            A_p = float(Y_pred[i, 0])
            b_p = float(np.expm1(Y_pred[i, 1]))
            k_p = float(np.expm1(Y_pred[i, 2]))
            y_sim = weibull(t_true, A_p, k_p, b_p)

            rmse = float(np.sqrt(mean_squared_error(y_true, y_sim)))
            curve_rmses.append(rmse)

            if ex_added < 2 and len(t_true) > 0:
                example_curves.append({
                    "t": t_true,
                    "y_true": y_true,
                    "y_pred": y_sim,
                    "cid": cid
                })
                ex_added += 1

        all_true_params.append(Y_test)
        all_pred_params.append(Y_pred)

        print(f"Fold {fold}/{args.kfold} done. Curves evaluated so far: {len(curve_rmses)}")

    all_true_params = np.vstack(all_true_params)
    all_pred_params = np.vstack(all_pred_params)

    # Convert log targets to physical for plots
    A_true = all_true_params[:, 0]
    A_pred = all_pred_params[:, 0]
    b_true = np.expm1(all_true_params[:, 1])
    b_pred = np.expm1(all_pred_params[:, 1])
    k_true = np.expm1(all_true_params[:, 2])
    k_pred = np.expm1(all_pred_params[:, 2])

    # Metrics summaries
    med_rmse = float(np.median(curve_rmses)) if curve_rmses else np.nan
    mean_rmse = float(np.mean(curve_rmses)) if curve_rmses else np.nan

    param_metrics = {
        "A": {
            "RMSE": float(np.sqrt(mean_squared_error(A_true, A_pred))),
            "R2": float(r2_score(A_true, A_pred))
        },
        "b": {
            "RMSE": float(np.sqrt(mean_squared_error(b_true, b_pred))),
            "R2": float(r2_score(b_true, b_pred))
        },
        "k": {
            "RMSE": float(np.sqrt(mean_squared_error(k_true, k_pred))),
            "R2": float(r2_score(k_true, k_pred))
        }
    }

    summary = {
        "method": "XGBoost + RegressorChain",
        "n_curves_used": int(len(df_model)),
        "n_curves_evaluated_in_cv": int(len(curve_rmses)),
        "cv_curve_rmse_median": med_rmse,
        "cv_curve_rmse_mean": mean_rmse,
        "param_metrics": param_metrics
    }

    with open(os.path.join(args.outdir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save per-curve RMSE
    pd.DataFrame({"curve_rmse": curve_rmses}).to_csv(os.path.join(args.outdir, "per_curve_metrics.csv"), index=False)

    # Save per-param metrics table
    per_param_df = pd.DataFrame([
        {"parameter": "A", "RMSE": param_metrics["A"]["RMSE"], "R2": param_metrics["A"]["R2"]},
        {"parameter": "b", "RMSE": param_metrics["b"]["RMSE"], "R2": param_metrics["b"]["R2"]},
        {"parameter": "k", "RMSE": param_metrics["k"]["RMSE"], "R2": param_metrics["k"]["R2"]},
    ])
    per_param_df.to_csv(os.path.join(args.outdir, "per_param_metrics.csv"), index=False)

    # Plots
    plot_rmse_boxplot(curve_rmses, args.outdir, "Cross-Validation Curve RMSE (Method 1: XGBoost Chain)")
    plot_parity(A_true, A_pred, "A", args.outdir)
    plot_parity(b_true, b_pred, "b", args.outdir)
    plot_parity(k_true, k_pred, "k", args.outdir)
    plot_residuals(A_true, A_pred, "A", args.outdir)
    plot_residuals(b_true, b_pred, "b", args.outdir)
    plot_residuals(k_true, k_pred, "k", args.outdir)
    plot_example_curves(example_curves, args.outdir)

    # Train final model on all data and save payload
    print("Training final model on full dataset ...")

    # Final target encoding
    if "Drug_Name" in df_model.columns:
        final_te = SmoothedTargetEncoder("Drug_Name", ["Target_log_k", "Target_log_b"], smoothing=5).fit(df_model)
        enc_cols = [f"Drug_Name_enc_{t}" for t in ["Target_log_k", "Target_log_b"]]
        X_full = pd.concat([df_model[features], final_te.transform(df_model)], axis=1)
        final_num_cols = num_cols + enc_cols
    else:
        final_te = None
        enc_cols = []
        X_full = df_model[features].copy()
        final_num_cols = num_cols

    final_pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), final_num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ], verbose_feature_names_out=False)

    X_full_p = final_pre.fit_transform(X_full)

    y_scaler = StandardScaler()
    Y_full_s = y_scaler.fit_transform(Y_raw)

    chain.fit(X_full_p, Y_full_s)

    # Feature names for importance plot
    try:
        feature_names = list(final_pre.get_feature_names_out())
        plot_feature_importance_xgb(chain, feature_names, args.outdir, topk=25)
    except Exception:
        pass

    payload = {
        "method": "XGBoost + RegressorChain",
        "model": chain,
        "preprocessor": final_pre,
        "y_scaler": y_scaler,
        "drug_encoder": final_te,
        "features_base": features
    }
    joblib.dump(payload, args.out_model)

    print("Done.")
    print(f"Results saved to: {args.outdir}")
    print(f"Model saved to: {args.out_model}")


if __name__ == "__main__":
    main()