import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


# Step 1: Load and clean the historical freight data
data = pd.read_csv("data/train_test.csv")
data["date"] = pd.to_datetime(data["date"])


if (data["distance"] <= 0).any():
    raise ValueError(
        "Rate-per-mile modeling requires all distances to be positive."
    )


# Shipment weight cannot be negative, so correct negative values
negative_weight_count = (
    data["weight"] < 0
).sum()

print(
    "Negative historical weights corrected:",
    negative_weight_count
)

data["weight"] = (
    data["weight"].abs()
)


# Step 2: Create features that are also available for December
def prepare_features(df):
    df = df.copy()

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    df["route"] = (
        df["pickup"]
        + " -> "
        + df["delivery"]
    )

    return df


data = prepare_features(data)


# Step 3: Define the December-compatible feature set
categorical_features = [
    "pickup",
    "delivery",
    "route",
    "equipment",
]

numeric_features = [
    "distance",
    "weight",
    "month",
    "day_of_week",
    "day_of_year",
]

features = (
    categorical_features
    + numeric_features
)


# Step 4: Use the same walk-forward validation periods
folds = [
    {
        "name": "AUGUST",
        "train_end": "2025-08-01",
        "test_start": "2025-08-01",
        "test_end": "2025-09-01",
    },
    {
        "name": "SEPTEMBER",
        "train_end": "2025-09-01",
        "test_start": "2025-09-01",
        "test_end": "2025-10-01",
    },
    {
        "name": "OCTOBER",
        "train_end": "2025-10-01",
        "test_start": "2025-10-01",
        "test_end": "2025-11-01",
    },
]


# Step 5: Build the preprocessing and XGBoost pipeline
def create_pipeline():

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


# Step 6: Validate the December-compatible RPM model
results = []

for fold in folds:

    train_data = data[
        data["date"] < fold["train_end"]
    ].copy()

    test_data = data[
        (
            data["date"] >= fold["test_start"]
        )
        &
        (
            data["date"] < fold["test_end"]
        )
    ].copy()

    X_train = train_data[features]
    X_test = test_data[features]

    # Train the model on dollars per mile
    y_train_rpm = (
        train_data["posted_rate"]
        / train_data["distance"]
    )

    pipeline = create_pipeline()

    pipeline.fit(
        X_train,
        y_train_rpm
    )

    predicted_rpm = pipeline.predict(
        X_test
    )

    # Convert the RPM prediction back to the total freight rate
    predicted_rate = (
        predicted_rpm
        * test_data["distance"].to_numpy()
    )

    mae = mean_absolute_error(
        test_data["posted_rate"],
        predicted_rate
    )

    print(
        fold["name"],
        "MAE:",
        mae
    )

    results.append({
        "Month": fold["name"],
        "MAE": mae
    })


results_df = pd.DataFrame(
    results
)

print(
    "\nDECEMBER-COMPATIBLE RPM MODEL VALIDATION"
)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\nAverage MAE:",
    results_df["MAE"].mean()
)


# Step 7: Retrain the final model on all historical rows
X_all = data[features]

y_all_rpm = (
    data["posted_rate"]
    / data["distance"]
)

final_model = create_pipeline()

print(
    "\nTraining final December RPM model..."
)

final_model.fit(
    X_all,
    y_all_rpm
)


# Step 8: Load and prepare the 31 December inputs
december = pd.read_csv(
    "data/december_chart_inputs.csv"
)

december["date"] = pd.to_datetime(
    december["date"]
)


if (december["distance"] <= 0).any():
    raise ValueError(
        "December data contains non-positive distances."
    )


# Apply the same weight cleaning used during training
december["weight"] = (
    december["weight"].abs()
)


december_prepared = prepare_features(
    december
)


# Step 9: Predict the December rate per mile
predicted_rpm = final_model.predict(
    december_prepared[features]
)


# Convert the RPM predictions into total freight rates
december_predictions = (
    predicted_rpm
    * december["distance"].to_numpy()
)


# The scorer requires every prediction to be positive
december_predictions = np.maximum(
    december_predictions,
    1.0
)


# Step 10: Fill the predicted_rate column
december["predicted_rate"] = (
    december_predictions
)


# Step 11: Preserve the exact format expected by score.py
december = december[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]
]


december["date"] = (
    december["date"]
    .dt.strftime("%Y-%m-%d")
)


# Step 12: Save the completed December file
december.to_csv(
    "data/december_chart_inputs.csv",
    index=False
)


print(
    "\nDECEMBER PREDICTIONS:"
)

print(
    december.to_string(
        index=False
    )
)


print(
    "\nDECEMBER SUMMARY:"
)

print(
    december["predicted_rate"].describe()
)


print(
    "\nSaved completed file:"
)

print(
    "data/december_chart_inputs.csv"
)