'''
    Classification Modeling
    Author: Shreya Bangera 
    Goal: 
        ├── Evaluates how well behavioral and neural features predict bout success 
        ├── Uses session-wise cross-validation nested within time-based phase segments.
        ├── It applies XGBoost, Random Forest, and Logistic Regression models to distinguish Successful vs Unsuccessful bouts, and summarizes results with confidence intervals.

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
import numpy as np
from itertools import chain, combinations, product
from scipy.stats import zscore
from scipy.stats import ttest_ind
import os
import shap
from typing import List, Tuple
from scipy.stats import wasserstein_distance
import textwrap
from typing import List, Tuple, Optional
from scipy.stats import gaussian_kde
import umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import OneHotEncoder
import matplotlib.cm as cm
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             precision_recall_curve, average_precision_score, roc_curve)


#######################################################
# Assign Phase blocks/numbering per block size
#######################################################

def assign_phase_blocks(df, block_size=3000):
    """
    Assign phase blocks to the dataframe by segmenting each session into equal-sized frame blocks.
    Args:
        df (pd.DataFrame): Input dataframe with a 'Session' column and frame-based rows.
        block_size (int): Number of frames per phase block.
    Returns:
        pd.DataFrame: Updated dataframe with a 'Phase' column.
    """
    df['Phase'] = -1
    session_groups = df.groupby('Session')

    for session, session_df in session_groups:
        n_rows = len(session_df)
        n_blocks = int(np.ceil(n_rows / block_size))
        phase_values = np.repeat(np.arange(n_blocks), block_size)[:n_rows]
        df.loc[session_df.index, 'Phase'] = phase_values

    return df


############################################################################
# Session-wise cross-validation nested within time-based phase segments
############################################################################

def run_phase_session_models(df, feature_cols, feature_name_map):
    metrics, shap_vals_list, shap_feats_list = [], [], []
    y_true_all, prob_xgb_all, prob_rf_all, prob_lr_all = [], [], [], []
    X_all, y_all = [], []

    for phase in sorted(df['Phase'].unique()):
        phase_data = df[df['Phase'] == phase]
        for held_out_session in phase_data['Session'].unique():
            test_data = phase_data[phase_data['Session'] == held_out_session]
            train_data = phase_data[phase_data['Session'] != held_out_session]

            if train_data['Successful_bout'].nunique() < 2 or test_data['Successful_bout'].nunique() < 2:
                continue

            X = train_data[feature_cols]
            y = train_data['Successful_bout']
            X_test = test_data[feature_cols]
            y_test = test_data['Successful_bout']

            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)
            X_test_scaled = scaler.transform(X_test)

            model_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                          reg_lambda=1.0, use_label_encoder=False, eval_metric='logloss',
                                          scale_pos_weight=(y == 0).sum() / (y == 1).sum(), random_state=42)
            model_rf = RandomForestClassifier(random_state=42)
            model_lr = LogisticRegression()

            model_xgb.fit(X_scaled, y)
            model_rf.fit(X_scaled, y)
            model_lr.fit(X_scaled, y)

            prob_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
            prob_rf = model_rf.predict_proba(X_test_scaled)[:, 1]
            prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]

            y_true_all.append(y_test.values)
            prob_xgb_all.append(prob_xgb)
            prob_rf_all.append(prob_rf)
            prob_lr_all.append(prob_lr)

            metrics.append({
                'Phase': phase, 'Session': held_out_session,
                'AUC_XGB': roc_auc_score(y_test, prob_xgb),
                'AUC_RF': roc_auc_score(y_test, prob_rf),
                'AUC_LR': roc_auc_score(y_test, prob_lr)
            })

            explainer = shap.TreeExplainer(model_xgb)
            shap_vals = explainer.shap_values(X_test_scaled)
            shap_vals_list.append(shap_vals)
            shap_feats_list.append(pd.DataFrame(X_test_scaled, columns=[feature_name_map.get(c, c) for c in X_test.columns]))

            X_all.append(X)
            y_all.append(y)

    return metrics, shap_vals_list, shap_feats_list, y_true_all, prob_xgb_all, prob_rf_all, prob_lr_all, X_all, y_all


############################################################################
# Bootstrapped Confidence Intervals (Optional)
############################################################################

def bootstrap_ci(y_true, y_probs, curve_type='roc', n_bootstraps=1000):
    rng = np.random.RandomState(42)
    base_line = np.linspace(0, 1, 100)
    interpolated = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_probs), len(y_probs))
        y_boot = y_true[indices]
        p_boot = y_probs[indices]
        if len(np.unique(y_boot)) < 2:
            continue

        if curve_type == 'roc':
            fpr, tpr, _ = roc_curve(y_boot, p_boot)
            interpolated.append(np.interp(base_line, fpr, tpr))
        else:
            precision, recall, _ = precision_recall_curve(y_boot, p_boot)
            interpolated.append(np.interp(base_line, recall[::-1], precision[::-1]))

    interpolated = np.array(interpolated)
    return base_line, np.mean(interpolated, axis=0), np.percentile(interpolated, 2.5, axis=0), np.percentile(interpolated, 97.5, axis=0)


############################################################################
# Session-wise Confidence Intervals (Optional)
############################################################################

def plot_ci_curves_sessionwise(y_true_list, prob_list, curve_type='roc', model_name='', color='#1f77b4', use_bootstrap=False):
    if use_bootstrap:
        y_true = np.concatenate(y_true_list)
        y_prob = np.concatenate(prob_list)
        base, mean, lower, upper = bootstrap_ci(np.array(y_true), np.array(y_prob), curve_type=curve_type)
    else:
        base = np.linspace(0, 1, 100)
        interpolated = []
        for y_true, y_prob in zip(y_true_list, prob_list):
            if curve_type == 'roc':
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                interpolated.append(np.interp(base, fpr, tpr))
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                interpolated.append(np.interp(base, recall[::-1], precision[::-1]))
        interpolated = np.array(interpolated)
        mean = np.mean(interpolated, axis=0)
        lower = np.percentile(interpolated, 2.5, axis=0)
        upper = np.percentile(interpolated, 97.5, axis=0)

    auc_score = np.mean([roc_auc_score(y, p) if curve_type == 'roc' else average_precision_score(y, p)
                         for y, p in zip(y_true_list, prob_list)])

    label = f"{model_name} (AUC = {auc_score:.2f})"
    plt.plot(base, mean, label=label, color=color, linewidth=2.5)
    plt.fill_between(base, lower, upper, color=color, alpha=0.2, label=f"{model_name} 95% CI")


######################################################################################################################
# Visualizations for xgboost, random forest, logistic regression (Feature Importance, PR curves, ROC-AUC curves)
######################################################################################################################

def summarize_and_plot_all(metrics, shap_vals, shap_feats, X_all, y_all, feature_cols, feature_name_map,
                            y_true_all, prob_xgb, prob_rf, prob_lr, use_bootstrap=False):
    df_metrics = pd.DataFrame(metrics)
    print("\nMean AUCs:", df_metrics[['AUC_XGB', 'AUC_RF', 'AUC_LR']].mean())

    if shap_vals:
        shap_values = np.vstack(shap_vals)
        shap_features = pd.concat(shap_feats)
        shap.summary_plot(shap_values, shap_features, plot_type="dot", show=True)

    X_final = StandardScaler().fit_transform(np.vstack(X_all))
    y_final = np.hstack(y_all)
    mapped_names = [feature_name_map[c] for c in feature_cols]

    xgb_final = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_final.fit(X_final, y_final)
    shap.summary_plot(shap.Explainer(xgb_final)(X_final), features=X_final, feature_names=mapped_names, plot_type="bar")

    rf_final = RandomForestClassifier(random_state=42)
    rf_final.fit(X_final, y_final)
    pd.Series(rf_final.feature_importances_, index=mapped_names).sort_values().plot(kind='barh', title='Random Forest Feature Importance')
    plt.tight_layout()
    plt.show()

    pd.Series(xgb_final.feature_importances_, index=mapped_names).sort_values().plot(kind='barh', title='XGBoost Feature Importance')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plot_ci_curves_sessionwise(y_true_all, prob_xgb, curve_type='roc', model_name='XGBoost', color='#1f77b4', use_bootstrap=use_bootstrap)
    plot_ci_curves_sessionwise(y_true_all, prob_rf, curve_type='roc', model_name='Random Forest', color='#2ca02c', use_bootstrap=use_bootstrap)
    plot_ci_curves_sessionwise(y_true_all, prob_lr, curve_type='roc', model_name='Logistic Regression', color='#d62728', use_bootstrap=use_bootstrap)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with 95% CI')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plot_ci_curves_sessionwise(y_true_all, prob_xgb, curve_type='pr', model_name='XGBoost', color='#1f77b4', use_bootstrap=use_bootstrap)
    plot_ci_curves_sessionwise(y_true_all, prob_rf, curve_type='pr', model_name='Random Forest', color='#2ca02c', use_bootstrap=use_bootstrap)
    plot_ci_curves_sessionwise(y_true_all, prob_lr, curve_type='pr', model_name='Logistic Regression', color='#d62728', use_bootstrap=use_bootstrap)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve with 95% CI')
    plt.legend()
    plt.tight_layout()
    plt.show()
