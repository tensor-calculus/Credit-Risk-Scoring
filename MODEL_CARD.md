# Model Card: Home Credit Default Risk Scoring Model

## Model Details
- **Model type**: LightGBM (Gradient Boosted Decision Trees)
- **Version**: 1.0
- **Framework**: scikit-learn + LightGBM + Optuna
- **Training data**: Home Credit Default Risk dataset (Kaggle)
- **Features**: SQL-engineered aggregations from 7 relational tables (application, bureau, previous_application, installments_payments, pos_cash_balance, credit_card_balance, bureau_balance)
- **Date trained**: July 2026

## Training Data
- **Source**: Home Credit Group via Kaggle
- **Size**: ~307,000 loan applications with ~8% default rate
- **Feature engineering**: 120+ raw features across 7 relational tables, aggregated into ~180 engineered features via SQL queries (bureau trade-line stats, previous application approval rates, installment payment patterns, POS/Cash loan delinquency, credit card utilization)
- **Target**: Binary (1 = payment difficulties / default, 0 = repaid on time)
- **Class imbalance**: ~8% positive (default) class, addressed via:
  - `scale_pos_weight` in LightGBM
  - Stratified K-fold cross-validation
  - Evaluation with PR-AUC and KS-statistic (not misleading accuracy)

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Area under the Receiver Operating Characteristic curve |
| **PR-AUC** | Area under the Precision-Recall curve (more informative under class imbalance) |
| **KS-Statistic** | Maximum separation between cumulative distributions of positives and negatives |
| **Gini Coefficient** | 2 × AUC − 1; standard credit risk metric |
| **Brier Score** | Calibration quality measure |
| **PDO Scorecard** | Probability mapped to 300–850 range using Points to Double the Odds methodology with Bayes calibration |

### Explainability
- SHAP TreeExplainer provides both global feature importance and per-applicant explanations
- Global beeswarm and bar plots show which features drive the model most
- Per-applicant waterfall charts show exactly how each factor pushes the prediction
