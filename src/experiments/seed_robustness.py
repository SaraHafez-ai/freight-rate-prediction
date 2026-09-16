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


# Rate-per-mile modeling requires positive distances
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

    df["route"] = (
        df["pickup"]
        + " -> "
        + df["delivery"]
    )

    return df


data = prepare_features(data)


# Step 3: Define the categorical features
categorical_features = [
    "pickup",
    "delivery",
    "route",
    "equipment",
]


# Step 4: Compare the three strongest feature configurations
feature_sets = {

    # Full geographic + date feature set
    "BASE_13": [
        "distance",
        "weight",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "month",
        "day_of_week",
        "day_of_year",
    ],

    # Current best feature set
    "NO_MONTH_NO_DAY_OF_YEAR": [
        "distance",
        "weight",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "day_of_week",
    ],

    # Remove all engineered date information
    "NO_DATE_FEATURES": [
        "distance",
        "weight",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ],
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


# Step 6: Test several random seeds
seeds = [
    42,
    123,
    2026,
]


# Step 7: Build the XGBoost pipeline
def create_pipeline(
    numeric_features,
    random_seed
):

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
        random_state=random_seed,
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


# Step 8: Evaluate every feature set across all seeds
results = []

for feature_set_name, numeric_features in feature_sets.items():

    features = (
        categorical_features
        + numeric_features
    )

    print("\n" + "=" * 90)
    print("FEATURE SET:", feature_set_name)
    print("=" * 90)

    for seed in seeds:

        print(
            "\nRANDOM SEED:",
            seed
        )

        fold_maes = []

        for fold in folds:

            # Train only on data before the validation month
            train_data = data[
                data["date"] < fold["train_end"]
            ].copy()

            # Use the following month as unseen future data
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
                numeric_features,
                seed
            )

            pipeline.fit(
                X_train,
                y_train_rpm
            )

            predicted_rpm = pipeline.predict(
                X_test
            )

            # Convert back to the total freight rate
            predicted_rate = (
                predicted_rpm
                * test_data["distance"].to_numpy()
            )

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
            "SEED AVERAGE MAE:",
            average_mae
        )

        results.append({
            "Feature Set": feature_set_name,
            "Seed": seed,
            "August MAE": fold_maes[0],
            "September MAE": fold_maes[1],
            "October MAE": fold_maes[2],
            "Average MAE": average_mae,
        })


# Step 9: Show the result for every seed
results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 110)
print("RATE-PER-MILE RESULTS BY RANDOM SEED")
print("=" * 110)

print(
    results_df.sort_values(
        [
            "Seed",
            "Average MAE",
        ]
    ).to_string(
        index=False
    )
)


# Step 10: Summarize stability across seeds
summary = (
    results_df
    .groupby("Feature Set")["Average MAE"]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max",
        ]
    )
    .reset_index()
)

summary = summary.rename(
    columns={
        "mean": "Mean MAE",
        "std": "Seed Std",
        "min": "Best Seed MAE",
        "max": "Worst Seed MAE",
    }
)

summary = summary.sort_values(
    "Mean MAE"
)


print("\n" + "=" * 110)
print("FINAL RATE-PER-MILE ROBUSTNESS SUMMARY")
print("=" * 110)

print(
    summary.to_string(
        index=False
    )
)


# Step 11: Show the strongest overall feature set
best = summary.iloc[0]

print("\n" + "=" * 110)
print("BEST ROBUST FEATURE SET")
print("=" * 110)

print(
    "Feature Set:",
    best["Feature Set"]
)

print(
    "Mean MAE across seeds:",
    best["Mean MAE"]
)

print(
    "Seed standard deviation:",
    best["Seed Std"]
)

print(
    "Best seed MAE:",
    best["Best Seed MAE"]
)

print(
    "Worst seed MAE:",
    best["Worst Seed MAE"]
)