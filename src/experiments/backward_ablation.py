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


# Step 3: Start from the best feature set from feature_ablation.py
base_categorical = [
    "pickup",
    "delivery",
    "route",
    "equipment",
]

base_numeric = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month",
    "day_of_week",
    "day_of_year",
]


# Step 4: Remove features one at a time and test useful combinations
feature_sets = {

    "BASE_13": {
        "categorical": base_categorical,
        "numeric": base_numeric,
    },

    "MINUS_ROUTE": {
        "categorical": [
            "pickup",
            "delivery",
            "equipment",
        ],
        "numeric": base_numeric,
    },

    "MINUS_PICKUP": {
        "categorical": [
            "delivery",
            "route",
            "equipment",
        ],
        "numeric": base_numeric,
    },

    "MINUS_DELIVERY": {
        "categorical": [
            "pickup",
            "route",
            "equipment",
        ],
        "numeric": base_numeric,
    },

    "MINUS_EQUIPMENT": {
        "categorical": [
            "pickup",
            "delivery",
            "route",
        ],
        "numeric": base_numeric,
    },

    "MINUS_DISTANCE": {
        "categorical": base_categorical,
        "numeric": [
            feature
            for feature in base_numeric
            if feature != "distance"
        ],
    },

    "MINUS_WEIGHT": {
        "categorical": base_categorical,
        "numeric": [
            feature
            for feature in base_numeric
            if feature != "weight"
        ],
    },

    "MINUS_MONTH": {
        "categorical": base_categorical,
        "numeric": [
            feature
            for feature in base_numeric
            if feature != "month"
        ],
    },

    "MINUS_DAY_OF_WEEK": {
        "categorical": base_categorical,
        "numeric": [
            feature
            for feature in base_numeric
            if feature != "day_of_week"
        ],
    },

    "MINUS_DAY_OF_YEAR": {
        "categorical": base_categorical,
        "numeric": [
            feature
            for feature in base_numeric
            if feature != "day_of_year"
        ],
    },

    "NO_PICKUP_DELIVERY_KEEP_ROUTE": {
        "categorical": [
            "route",
            "equipment",
        ],
        "numeric": base_numeric,
    },

    "NO_ROUTE_KEEP_CITIES": {
        "categorical": [
            "pickup",
            "delivery",
            "equipment",
        ],
        "numeric": base_numeric,
    },

    # This is the 11-feature set that performed well earlier
    "NO_MONTH_NO_DAY_OF_YEAR": {
        "categorical": base_categorical,
        "numeric": [
            feature
            for feature in base_numeric
            if feature not in [
                "month",
                "day_of_year",
            ]
        ],
    },

    "NO_DATE_FEATURES": {
        "categorical": base_categorical,
        "numeric": [
            feature
            for feature in base_numeric
            if feature not in [
                "month",
                "day_of_week",
                "day_of_year",
            ]
        ],
    },
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


# Step 6: Build the preprocessing and XGBoost pipeline
def create_pipeline(
    categorical_features,
    numeric_features
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


# Step 7: Test each feature set using rate per mile as the target
results = []

for feature_set_name, feature_config in feature_sets.items():

    categorical_features = (
        feature_config["categorical"]
    )

    numeric_features = (
        feature_config["numeric"]
    )

    features = (
        categorical_features
        + numeric_features
    )

    print("\n" + "=" * 80)
    print("FEATURE SET:", feature_set_name)
    print("=" * 80)

    fold_maes = []

    for fold in folds:

        # Train only on data before the validation month
        train_data = data[
            data["date"] < fold["train_end"]
        ].copy()

        # Use the next month as unseen future data
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

        # Train on dollars per mile
        y_train_rpm = (
            train_data["posted_rate"]
            / train_data["distance"]
        )

        y_test_rate = test_data[
            "posted_rate"
        ]

        pipeline = create_pipeline(
            categorical_features,
            numeric_features
        )

        pipeline.fit(
            X_train,
            y_train_rpm
        )

        predicted_rpm = pipeline.predict(
            X_test
        )

        # Convert back to total freight rate
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
        "AVERAGE MAE:",
        average_mae
    )

    results.append({
        "Feature Set": feature_set_name,
        "August MAE": fold_maes[0],
        "September MAE": fold_maes[1],
        "October MAE": fold_maes[2],
        "Average MAE": average_mae,
        "Feature Count": len(features),
    })


# Step 8: Compare all backward-ablation results
results_df = pd.DataFrame(
    results
)

base_mae = results_df.loc[
    results_df["Feature Set"] == "BASE_13",
    "Average MAE"
].iloc[0]

results_df[
    "Improvement vs Base"
] = (
    base_mae
    - results_df["Average MAE"]
)

results_df = results_df.sort_values(
    "Average MAE"
)


print("\n" + "=" * 110)
print("RATE-PER-MILE BACKWARD ABLATION RESULTS")
print("=" * 110)

print(
    results_df.to_string(
        index=False
    )
)


# Step 9: Show the best feature set
best = results_df.iloc[0]

print("\n" + "=" * 110)
print("BEST FEATURE SET")
print("=" * 110)

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

print(
    "Improvement over base:",
    best["Improvement vs Base"]
)