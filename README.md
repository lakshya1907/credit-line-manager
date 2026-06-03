Credit Line Manager

An intelligent end-to-end pipeline for optimizing credit limits, enforcing portfolio budgets, and generating regulatory-grade explainability.


Overview
Standard credit scoring models answer a binary question — will this customer default? Credit Line Manager goes further: it determines how much credit each customer should actually receive, subject to portfolio-level risk constraints and human-readable reason codes.
The pipeline integrates three components that are typically treated in isolation:

Risk Modeling — calibrated probability of default (PD) and exposure-at-default (EAD) estimates
Limit Optimization — expected-profit maximization across a candidate limit grid
Portfolio Governance — budget enforcement, stress testing, and SHAP-based explainability


The Problem
Standard ModelsThis PipelineBinary approve / rejectContinuous limit recommendationNo link between PD and balance utilizationPD × EAD → Expected Loss budgetNo portfolio-level constraintsHard EL & EAD budget capsBlack-box decisionsSHAP feature importance + reason codes

Pipeline Architecture
UCI Credit Dataset
       │
       ▼
Data Cleaning & Feature Engineering (23 → 56 features)
  • Utilization ratios & trends
  • Payment ratios (full-payers vs minimum-payers)
  • Delinquency metrics (max delay, streaks)
  • Utilization × delinquency interaction terms
       │
       ▼
PD Model  ──────────────────────────────────────────────────────────────────────┐
(XGBoost Classifier)                                                            │
       │                                                                        │
       ▼                                                                        │
PD Calibration (Isotonic Regression)                                            │
       │                                                                        ▼
       └────────────────────────► Decision Engine ◄──── EAD Model (XGBoost Regressor)
                                        │
                              Evaluate candidate limits
                              → Increase / Decrease / Hold
                                        │
                                        ▼
                              Portfolio Constraint Check
                              (EL Budget + EAD Budget)
                                        │
                              ┌─────────┴─────────┐
                           Approve             Reject/Skip
                                        │
                                        ▼
                              SHAP Explainability
                              → Feature importance
                              → Per-customer reason codes
                                        │
                                        ▼
                              Stress Testing (+10/+20/+30% PD shocks)
                                        │
                                        ▼
                              Final Recommendations + Reports

Results
Model Performance
MetricValueNotesROC-AUC0.7788Discriminates defaulters vs non-defaultersPR-AUC0.5611Performance on imbalanced default classBrier Score0.1738 → 0.133323.3% improvement post-calibrationEAD Forecast MAENT$735.24~1.5% of mean balance (~NT$50K)
Portfolio Optimisation (No Stress)
MetricValueEL used / budget52.75 / 20,000,000EAD used / budget33,262,198 / 50,000,000Limits increased3,060 customersLimits decreased20,969 customersTotal EP upliftNT$39,035,211
Stress Testing
Portfolio remained within EL budget under PD shocks of +10%, +20%, and +30%.
Top SHAP Features
FeatureImportanceMeaningdelinq_max0.3216Max historical delinquencyPAY_00.1955Most recent repayment statusutil_20.1289Medium-term spending intensitypay_ratio_min0.1067Minimum payment discipline

Dataset
UCI Credit Card Default Dataset — 30,000 Taiwanese credit card holders, April–September 2005, 22.1% default rate.

Project Structure
credit-line-manager/
├── data/
│   ├── raw/                  # Raw UCI dataset
│   └── processed/            # Pipeline outputs (output.csv)
├── models/
│   ├── pd.pkl                # Trained PD model
│   └── calibrator.pkl        # Isotonic regression calibrator
├── src/
│   ├── data_prep.py          # Data loading and cleaning
│   ├── features.py           # Feature engineering (23 → 56 features)
│   ├── pd_model.py           # XGBoost PD classifier training
│   ├── calibrate.py          # Isotonic regression calibration
│   └── decision_engine.py    # Limit optimization + portfolio constraints
├── main.py                   # Pipeline entry point
└── requirements.txt

Installation
bashgit clone https://github.com/lakshya1907/credit-line-manager.git
cd credit-line-manager
pip install -r requirements.txt
Dependencies: pandas, numpy, scikit-learn, xgboost, lightgbm, shap, streamlit, plotly, joblib, pyarrow, scipy

Usage

Place the UCI credit dataset at data/raw/uci_credit.csv.
Run the full pipeline:

bashpython main.py
Outputs are saved to data/processed/output.csv. Trained models are persisted to models/.
To launch the interactive dashboard:
bashstreamlit run src/dashboard.py

Dashboard
The Streamlit dashboard includes four views:

Portfolio Overview — action distribution, EP uplift by action, PD distribution, EL budget utilization
Action Queue — per-customer recommendations sortable by EP uplift, downloadable as CSV
Customer Drilldown — individual limit recommendation with SHAP waterfall explanation
Policy Simulator — adjust EL/EAD budget constraints and PD shock parameters in real time, with live re-computation of portfolio outcomes


Future Work

Model Framework — benchmark LightGBM & CatBoost; Optuna hyperparameter tuning; test on LendingClub & Home Credit datasets
Explainability — add LIME alongside SHAP; counterfactual explanations; SHAP waterfall plots in dashboard
Portfolio Optimization — replace greedy heuristics with Integer Linear Programming; model LGD uncertainty; sector-specific stress tests
Production — deploy via FastAPI; PSI/CSI drift monitoring; automated retraining scheduler


Authors

<<<<<<< HEAD
Manya Chawla — 24/IT/113, Delhi Technological University
Lakshya Jindal — 24/IT/100, Delhi Technological University
=======
Manya Chawla — Delhi Technological University
Lakshya Jindal — Delhi Technological University
>>>>>>> d91c2b1819b5204237dcf2b4b826b7c5d4404f4b


References

Hand, D.J. & Henley, W.E. (1997). Statistical classification methods in consumer credit scoring.
Yeh, I.C. & Lien, C.H. (2009). The comparisons of data mining techniques.
Lessmann, S. et al. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring.
<<<<<<< HEAD
Sohn, S.Y. et al. (2014). Optimization-based credit limit management.
=======
Sohn, S.Y. et al. (2014). Optimization-based credit limit management.
>>>>>>> d91c2b1819b5204237dcf2b4b826b7c5d4404f4b
