import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


# Step 1: Load the historical freight data
data = pd.read_csv("data/train_test.csv")
data["date"] = pd.to_datetime(data["date"])


# Rate per mile requires a positive distance
if (data["distance"] <= 0).any():
    raise ValueError(
        "Rate-per-mile modeling requires all distances to be positive."
    )


# Step 2: Create the engineered features used in the final feature set
def prepare_features(df):
    df = df.copy()

    # Capture weekly pricing behavior
    df["day_of_week"] = df["date"].dt.dayofweek

    # Combine origin and destination into a lane feature
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


# Step 5: Build preprocessing for models that need numeric inputs
def create_preprocessor():

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

    return ColumnTransformer(
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


# Step 6: Define the model configurations
def create_model(model_name):

    if model_name == "CatBoost":

        return CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            loss_function="MAE",
            random_seed=42,
            verbose=False
        )

    if model_name == "XGBoost":

        # Best XGBoost settings found during tuning
        return XGBRegressor(
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

    if model_name == "LightGBM":

        return LGBMRegressor(
            n_estimators=600,
            max_depth=-1,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )

    if model_name == "ExtraTrees":

        return ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


model_names = [
    "XGBoost",
    "CatBoost",
    "LightGBM",
    "ExtraTrees",
]

target_types = [
    "Direct",
    "Rate Per Mile",
]


# Step 7: Test every model with both target strategies
results = []

for model_name in model_names:

    for target_type in target_types:

        print("\n" + "=" * 80)
        print(
            "MODEL:",
            model_name,
            "| TARGET:",
            target_type
        )
        print("=" * 80)

        fold_maes = []

        for fold in folds:

            # Train only on information available before the test month
            train_data = data[
                data["date"] < fold["train_end"]
            ].copy()

            # Use the following month as unseen future data
            test_data = data[
                (
                    data["date"]
                    >= fold["test_start"]
                )
                &
                (
                    data["date"]
                    < fold["test_end"]
                )
            ].copy()

            X_train = train_data[features]
            X_test = test_data[features]

            # Final evaluation is always against the true total posted rate
            y_test_rate = test_data[
                "posted_rate"
            ]


            # Choose which target the model learns
            if target_type == "Direct":

                y_train = train_data[
                    "posted_rate"
                ]

            else:

                y_train = (
                    train_data["posted_rate"]
                    / train_data["distance"]
                )


            model = create_model(
                model_name
            )


            # CatBoost handles the categorical columns directly
            if model_name == "CatBoost":

                model.fit(
                    X_train,
                    y_train,
                    cat_features=categorical_features
                )

                raw_predictions = model.predict(
                    X_test
                )


            # Other models use imputation and one-hot encoding first
            else:

                pipeline = Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            create_preprocessor()
                        ),
                        (
                            "model",
                            model
                        )
                    ]
                )

                pipeline.fit(
                    X_train,
                    y_train
                )

                raw_predictions = pipeline.predict(
                    X_test
                )


            # Convert rate-per-mile predictions back to total freight rates
            if target_type == "Rate Per Mile":

                predictions = (
                    raw_predictions
                    * test_data["distance"].to_numpy()
                )

            else:

                predictions = raw_predictions


            # Compare every model using the same final metric
            mae = mean_absolute_error(
                y_test_rate,
                predictions
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
            "Model": model_name,
            "Target": target_type,
            "August MAE": fold_maes[0],
            "September MAE": fold_maes[1],
            "October MAE": fold_maes[2],
            "Average MAE": average_mae
        })


# Step 8: Compare all model-target combinations
results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Average MAE"
)


print("\n" + "=" * 100)
print("MODEL AND TARGET COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)


# Step 9: Show the overall winner
best = results_df.iloc[0]

print("\n" + "=" * 100)
print("BEST COMBINATION")
print("=" * 100)

print(
    "Model:",
    best["Model"]
)

print(
    "Target:",
    best["Target"]
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