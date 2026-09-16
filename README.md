# Freight Rate Prediction

This repository contains my solution for the freight rate prediction assessment.

The goal was to train a regression model on historical freight data, validate it using a realistic time-based split, generate predictions for the provided validation set, and produce daily December predictions for the fixed Lexington → Fort Wayne route.

## Final approach

The final model uses:

- **XGBoost**
- **Rate-per-mile prediction**
- **Expanding-window time validation**
- **11 input features**
- **Median imputation** for missing numeric values
- **Most-frequent imputation + one-hot encoding** for categorical values
- Basic data cleaning, including correcting invalid negative shipment weights

Instead of predicting the total freight rate directly, the model predicts:

```text
rate_per_mile = posted_rate / distance
```

The final freight rate is then reconstructed as:

```text
predicted_rate = predicted_rate_per_mile × distance
```

This gave better temporal validation results than direct total-rate prediction.

---

## Project structure

```text
freight-rate-prediction/
│
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
│
├── report/
│   └── Freight_Rate_Prediction_Final_Report.pdf
│
├── results/
│   └── rate_per_mile/
│       ├── backward_ablation.csv
│       └── feature_ablation.csv
│
├── scorer_results/
│   └── candidate_december.png
│
├── src/
│   ├── 01_explore_data.py
│   ├── 02_feature_analysis.py
│   ├── 03_data_quality.py
│   ├── 04_outlier_analysis.py
│   ├── 05_time_validation.py
│   ├── 06_compare_models.py
│   ├── 07_xgboost_tuning.py
│   ├── 08_generate_validation_predictions.py
│   ├── 09_december_model.py
│   │
│   └── experiments/
│       ├── backward_ablation.py
│       ├── december_target_test.py
│       ├── feature_ablation.py
│       ├── seed_robustness.py
│       └── target_experiment.py
│
├── validation_predictions.csv
├── score.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Validation strategy

Because the task is to predict future freight rates, I used a **time-based expanding-window split** instead of a random train/test split.

The validation folds were:

| Fold | Training data | Validation data |
|---|---|---|
| August | Before Aug 1, 2025 | August 2025 |
| September | Before Sep 1, 2025 | September 2025 |
| October | Before Oct 1, 2025 | October 2025 |

This setup better reflects how the model would behave in production, where future loads must be predicted using only historical information.

---

## Model comparison

I compared several regression approaches, including:

- XGBoost
- CatBoost
- LightGBM
- ExtraTrees

I also compared two target strategies:

- direct total-rate prediction
- rate-per-mile prediction

Among the tested configurations, **XGBoost with a rate-per-mile target** produced the strongest validation performance and was selected for the final model.

---

## Final features

The final validation model uses the following 11 features.

### Categorical

- `pickup`
- `delivery`
- `route`
- `equipment`

### Numeric

- `distance`
- `weight`
- `pickup_lat`
- `pickup_lon`
- `delivery_lat`
- `delivery_lon`
- `day_of_week`

The `route` feature is created as:

```text
pickup + " -> " + delivery
```

Feature selection was supported by feature ablation, backward elimination, and seed-robustness checks.

---

## Data preparation

The preprocessing pipeline includes:

- date parsing
- route creation
- day-of-week extraction
- correction of invalid negative weight values
- median imputation for numeric missing values
- most-frequent imputation for categorical missing values
- one-hot encoding for categorical features

Unknown categories are handled using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This allows the model to process locations that were not present in the training data.

---

## Final XGBoost configuration

```text
n_estimators       = 1000
max_depth          = 5
learning_rate      = 0.05
subsample          = 0.90
colsample_bytree   = 0.90
min_child_weight   = 1
reg_alpha          = 0.0
reg_lambda         = 1.0
objective          = reg:absoluteerror
eval_metric        = mae
tree_method        = hist
random_state       = 42
```

---

## Validation results

Final walk-forward MAE:

| Validation month | MAE |
|---|---:|
| August | $91.72 |
| September | $95.45 |
| October | $99.32 |
| **Average** | **$95.50** |

MAE represents the average absolute difference between the predicted freight rate and the actual freight rate.

---

## Generate validation predictions

Run:

```bash
python src/08_generate_validation_predictions.py
```

This trains the final model on all labeled historical rows and creates:

```text
validation_predictions.csv
```

The file contains exactly:

```text
load_id,predicted_rate
```

for all 12,000 validation loads.

---

## December prediction model

The December file does not contain all of the features available in the main validation set, so a separate compatible model is trained using only fields available in both historical and December data.

The December-compatible feature set includes:

### Categorical

- `pickup`
- `delivery`
- `route`
- `equipment`

### Numeric

- `distance`
- `weight`
- `month`
- `day_of_week`
- `day_of_year`

The December model also uses the rate-per-mile target.

Its walk-forward validation result was:

| Month | MAE |
|---|---:|
| August | $86.32 |
| September | $102.68 |
| October | $114.08 |
| **Average** | **$101.02** |

Run:

```bash
python src/09_december_model.py
```

This fills the `predicted_rate` column in:

```text
data/december_chart_inputs.csv
```

---

## Official scorer

After generating both prediction files, run:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates:

- all 12,000 validation predictions
- all 31 December predictions
- required file structure and columns
- positive prediction values

It also creates:

```text
scorer_results/candidate_december.png
```

Final hidden validation metrics are calculated separately after submission.

---

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Reproduce the final submission

Run the following from the project root:

```bash
python src/08_generate_validation_predictions.py
python src/09_december_model.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

---

## Supporting experiments

The `src/experiments/` folder contains the experiments used during model development, including:

- direct rate vs rate-per-mile prediction
- feature ablation
- backward feature elimination
- seed robustness
- December target comparison

These scripts were used to support model-selection decisions but are not required to generate the final submission files.

---

## Report

The full assessment report is available here:

```text
report/Freight_Rate_Prediction_Final_Report.pdf
```