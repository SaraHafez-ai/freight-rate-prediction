import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


# Step 1: Load the historical freight data
data = pd.read_csv("data/train_test.csv")
data["date"] = pd.to_datetime(data["date"])


# Rate-per-mile modeling requires a valid positive distance
if (data["distance"] <= 0).any():
    raise ValueError(
        "Rate-per-mile modeling requires all distances to be positive."
    )


# Step 2: Create the engineered features
def prepare_features(df):
    df = df.copy()

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    # Combine pickup and delivery into a lane feature
    df["route"] = (
        df["pickup"]
        + " -> "
        + df["delivery"]
    )

    return df


data = prepare_features(data)


# Step 3: Define the categorical features shared by every test
categorical_features = [
    "pickup",
    "delivery",
    "route",
    "equipment",
]


# Step 4: Define the feature groups we want to compare
base_numeric = [
    "distance",
    "weight",
    "month",
    "day_of_week",
    "day_of_year",
]


feature_sets = {

    "BASE_9_FEATURES": (
        base_numeric
    ),

    "PLUS_DELIVERY_COORDS": (
        base_numeric
        + [
            "delivery_lat",
            "delivery_lon",
        ]
    ),

    "PLUS_PICKUP_COORDS": (
        base_numeric
        + [
            "pickup_lat",
            "pickup_lon",
        ]
    ),

    "PLUS_ALL_COORDS": (
        base_numeric
        + [
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
        ]
    ),

    "PLUS_MARKET_SIGNALS": (
        base_numeric
        + [
            "market_index",
            "quote_signal",
        ]
    ),

    "ALL_EXTRA_FEATURES": (
        base_numeric
        + [
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "market_index",
            "quote_signal",
        ]
    ),
}


# Step 5: Use the same walk-forward validation periods
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


# Step 6: Build the preprocessing and tuned XGBoost pipeline
def create_pipeline(numeric_features):

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
        n_estimators=800,
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


# Step 7: Test every feature set using rate per mile as the target
results = []

for feature_set_name, numeric_features in feature_sets.items():

    print("\n" + "=" * 80)
    print("FEATURE SET:", feature_set_name)
    print("=" * 80)

    features = (
        categorical_features
        + numeric_features
    )

    fold_maes = []

    for fold in folds:

        # Train only on data available before the validation month
        train_data = data[
            data["date"] < fold["train_end"]
        ].copy()

        # Treat the following month as unseen future data
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

        # Train XGBoost to predict dollars per mile
        y_train_rpm = (
            train_data["posted_rate"]
            / train_data["distance"]
        )

        y_test_rate = test_data[
            "posted_rate"
        ]

        pipeline = create_pipeline(
            numeric_features
        )

        pipeline.fit(
            X_train,
            y_train_rpm
        )

        predicted_rpm = pipeline.predict(
            X_test
        )

        # Convert the predicted rate per mile back to total dollars
        predicted_rate = (
            predicted_rpm
            * test_data["distance"].to_numpy()
        )

        # Evaluate using total freight-rate MAE
        mae = mean_absolute_error(
            y_test_rate,
            predicted_rate
        )

        fold_maes.append(
            mae
        )

        print(
            fold["name"],
            "MAE:",
            mae
        )


    average_mae = np.mean(
        fold_maes
    )

    print(
        "AVERAGE MAE:",
        average_mae
    )

    results.append({
        "Feature Set": feature_set_name,
        "August MAE": fold_maes[0],
        "September MAE": fold_maes[1],
        "October MAE": fold_maes[2],
        "Average MAE": average_mae,
        "Numeric Feature Count": len(
            numeric_features
        ),
        "Total Feature Count": (
            len(numeric_features)
            + len(categorical_features)
        )
    })


# Step 8: Compare the feature sets
results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Average MAE"
)

print("\n" + "=" * 100)
print("RATE-PER-MILE FEATURE ABLATION RESULTS")
print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)


# Step 9: Show the best feature set
best = results_df.iloc[0]

print("\n" + "=" * 100)
print("BEST FEATURE SET")
print("=" * 100)

print(
    "Feature Set:",
    best["Feature Set"]
)

print(
    "August MAE:",
    best["August MAE"]
)

print(
    "September MAE:",
    best["September MAE"]
)

print(
    "October MAE:",
    best["October MAE"]
)

print(
    "Average MAE:",
    best["Average MAE"]
)