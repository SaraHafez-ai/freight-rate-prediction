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


if (data["distance"] <= 0).any():
    raise ValueError(
        "Rate-per-mile modeling requires all distances to be positive."
    )


# Step 2: Create the engineered features
def prepare_features(df):
    df = df.copy()

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["route"] = (
        df["pickup"]
        + " -> "
        + df["delivery"]
    )

    return df


data = prepare_features(data)


# Step 3: Use the final 11-feature set
categorical_features = [
    "pickup",
    "delivery",
    "route",
    "equipment",
]

numeric_features = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "day_of_week",
]

features = (
    categorical_features
    + numeric_features
)


# Step 4: Set up the walk-forward validation periods
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


# Step 5: Define the XGBoost configurations to compare
configs = {

    "current_best": {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },

    "shallower": {
        "n_estimators": 800,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },

    "slightly_deeper": {
        "n_estimators": 800,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },

    "slower_learning": {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },

    "more_trees": {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },

    "regularized": {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 3,
        "reg_alpha": 0.05,
        "reg_lambda": 3.0,
    },
}


# Step 6: Build the preprocessing and XGBoost pipeline
def create_pipeline(config):

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
        **config,
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


# Step 7: Test every configuration using rate per mile
results = []

for config_name, config in configs.items():

    print("\n" + "=" * 80)
    print("CONFIGURATION:", config_name)
    print("=" * 80)

    fold_maes = []

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

        y_train_rpm = (
            train_data["posted_rate"]
            / train_data["distance"]
        )

        y_test_rate = (
            test_data["posted_rate"]
        )

        pipeline = create_pipeline(
            config
        )

        pipeline.fit(
            X_train,
            y_train_rpm
        )

        predicted_rpm = pipeline.predict(
            X_test
        )

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
        "Configuration": config_name,
        "August MAE": fold_maes[0],
        "September MAE": fold_maes[1],
        "October MAE": fold_maes[2],
        "Average MAE": average_mae,
    })


# Step 8: Compare the tuning results
results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Average MAE"
)


print("\n" + "=" * 100)
print("RATE-PER-MILE XGBOOST TUNING RESULTS")
print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)


# Step 9: Show the best configuration
best = results_df.iloc[0]

print("\n" + "=" * 100)
print("BEST CONFIGURATION")
print("=" * 100)

print(
    "Configuration:",
    best["Configuration"]
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

print("\nPARAMETERS:")

print(
    configs[
        best["Configuration"]
    ]
)