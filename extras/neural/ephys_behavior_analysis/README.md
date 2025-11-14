# Neuro-Behavior Analysis

A modular toolkit for analyzing how neural signals such as gamma, theta, and velocity interact with behavioral HHMM states during maze navigation, with a focus on power dynamics, decision nodes, and temporal encoding.

It supports structured analysis of:
- Power feature comparisons (gamma, theta, velocity) between successful and unsuccessful bouts
- Neural activity aligned to behavioral states and state transitions at decision nodes
- Pairwise correlation analysis between neural features (gamma, theta) and behavior (velocity)
- Temporal UMAP embedding to reveal low-dimensional structure of neural state transitions
- Training and evaluating classification models using neural, behavioral features that predict long-term goal success (successful vs. unsuccessful bouts)

---

##  Features

### -  Power × State Mapping & Comparison (`power_state_analysis.py`)
- Compare Neural Dynamics across Behavioral States
- Align and visualize temporal dynamics of power signals centered on HHMM state transitions


### - Bout-wise Neural Feature Dynamics (`boutwise_neural_dynamics.py`)
- Compute median gamma, theta, and velocity values for each bout
- Label each bout as "Valid" and "Successful" based on reward reach and engagement
- Visualize feature distributions between Successful vs. Unsuccessful bouts


### - Feature Correlation & Node-Level Metrics (`correlation_analysis.py`)
- Quantify pairwise relationships among gamma, theta, and movement metrics within and across decision nodes.
- Visualize heatmaps of correlation values across bout bins or session blocks


### - Temporal UMAP Embedding of State Dynamics  (`umaps_state_temporal_embedding.py`)
- Applies UMAP dimensionality reduction to neural signal windows surrounding HHMM state transitions.
- Extract sliding windows of gamma/theta around Decision (Reward) node events
- Reduce high-dimensional temporal structure to 2D embedding
- Reveal manifold structure of temporal power patterns within and across HHMM states


### - Classification Modeling  (`classification_modeling.py`)
- Train classifiers (e.g., XGBoost, Random Forest, Logistic Regression)
- Evaluate model performance using:
  - ROC-AUC and PR curves
  - Bootstrapped and session-wise confidence intervals
- Visualize SHAP values and feature importance
