# Freight Rate Prediction

Machine learning solution for predicting freight load rates from historical shipment data.

The project uses time-based validation to compare multiple regression models, target formulations, feature sets, and XGBoost configurations before generating the final validation and December predictions.

## Project Structure

```text
freight-rate-prediction/
│
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
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
│       ├── target_experiment.py
│       ├── feature_ablation.py
│       ├── backward_ablation.py
│       ├── seed_robustness.py
│       └── december_target_test.py
│
├── validation_predictions.csv
├── score.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Validation Strategy

Because the goal is to predict future freight rates, random train/test splitting was avoided.

An expanding-window time-based validation strategy was used:

- Train on data before August → validate on August
- Train on data before September → validate on September
- Train on data before October → validate on October

This better represents how the model would perform on future shipments.

## Model Selection

The following models were compared:

- XGBoost
- CatBoost
- LightGBM
- ExtraTrees

Both direct total-rate prediction and rate-per-mile prediction were tested.

The strongest approach was:

- Model: XGBoost
- Target: Rate per mile
- Final prediction: predicted rate per mile × shipment distance

Using rate per mile reduced the effect of distance scaling and produced better walk-forward validation results than directly predicting the total posted rate.

## Final Features

The final validation model uses 11 features:

### Categorical

- pickup
- delivery
- route
- equipment

### Numeric

- distance
- weight
- pickup_lat
- pickup_lon
- delivery_lat
- delivery_lon
- day_of_week

The `route` feature is created by combining the pickup and delivery locations.

The feature set was selected using feature ablation, backward ablation, and random-seed robustness testing.

## Final XGBoost Configuration

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
```

## Validation Performance

Final walk-forward MAE:

| Validation Month | MAE |
|---|---:|
| August | $91.82 |
| September | $95.58 |
| October | $99.51 |
| **Average** | **$95.63** |

MAE represents the average absolute difference between the predicted freight rate and the actual freight rate.

## Preprocessing

Numeric missing values are filled using median imputation.

Categorical missing values are filled using the most frequent category and then transformed using one-hot encoding.

Unknown categories in future data are handled using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This allows the pipeline to process locations that were not present in the historical training data.

## Generate Validation Predictions

Run:

```bash
python src/08_generate_validation_predictions.py
```

This trains the final model on all labeled historical data and creates:

```text
validation_predictions.csv
```

The file contains:

```text
load_id,predicted_rate
```

for all 12,000 validation loads.

## Generate December Predictions

Run:

```bash
python src/09_december_model.py
```

The December input contains fewer available features, so a separate December-compatible model is trained.

The December model also uses a rate-per-mile target.

Its walk-forward average MAE was approximately:

```text
$101.35
```

The script fills the `predicted_rate` column in:

```text
data/december_chart_inputs.csv
```

## Run the Official Scorer

After generating both prediction files, run:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

A successful run should report:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

The scorer also generates:

```text
scorer_results/candidate_december.png
```

Final validation metrics for the hidden validation set are calculated after submission.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Reproduce Final Outputs

The main final workflow is:

```bash
python src/08_generate_validation_predictions.py
python src/09_december_model.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Additional Experiments

The `src/experiments/` directory contains supporting experiments used during model development, including:

- direct rate vs rate-per-mile prediction
- feature-group ablation
- backward feature elimination
- random-seed robustness testing
- December target comparison

These experiments were used for model selection but are not required to generate the final submission files.