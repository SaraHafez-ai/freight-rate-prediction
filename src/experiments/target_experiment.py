import pandas as pd

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


# Step 1: Load the historical freight data
data = pd.read_csv("data/train_test.csv")
data["date"] = pd.to_datetime(data["date"])


# Step 2: Create the engineered features used by the final model
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


# Step 4: Set up the same walk-forward validation periods
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


# Step 5: Build the preprocessing and tuned XGBoost pipeline
def create_pipeline():

    # Fill missing numeric values using the median
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

    # Fill missing categories and convert them with one-hot encoding
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

    # Use the best settings found during XGBoost tuning
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


# Step 6: Compare direct-rate prediction with rate-per-mile prediction
results = []

for fold in folds:

    print("\n" + "=" * 70)
    print("VALIDATING:", fold["name"])
    print("=" * 70)

    # Train only on historical data before the validation month
    train_data = data[
        data["date"] < fold["train_end"]
    ].copy()

    # Treat the next month as unseen future data
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

    y_test_rate = test_data["posted_rate"]


    # Model A: predict the total posted rate directly
    print("\nMODEL A: DIRECT RATE")

    direct_model = create_pipeline()

    direct_model.fit(
        X_train,
        train_data["posted_rate"]
    )

    direct_predictions = direct_model.predict(
        X_test
    )

    direct_mae = mean_absolute_error(
        y_test_rate,
        direct_predictions
    )

    print(
        "Direct-rate MAE:",
        direct_mae
    )


    # Model B: predict rate per mile first
    print("\nMODEL B: RATE PER MILE")

    y_train_rpm = (
        train_data["posted_rate"]
        / train_data["distance"]
    )

    rpm_model = create_pipeline()

    rpm_model.fit(
        X_train,
        y_train_rpm
    )

    predicted_rpm = rpm_model.predict(
        X_test
    )

    # Convert dollars per mile back into the total freight rate
    rpm_rate_predictions = (
        predicted_rpm
        * test_data["distance"].to_numpy()
    )

    # Evaluate both approaches using the same total-rate MAE
    rpm_mae = mean_absolute_error(
        y_test_rate,
        rpm_rate_predictions
    )

    print(
        "Rate-per-mile model MAE:",
        rpm_mae
    )

    results.append({
        "Month": fold["name"],
        "Direct Rate MAE": direct_mae,
        "Rate Per Mile MAE": rpm_mae
    })


# Step 7: Show the monthly results
results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("TARGET EXPERIMENT RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# Step 8: Compare average performance across all three months
direct_average = (
    results_df["Direct Rate MAE"].mean()
)

rpm_average = (
    results_df["Rate Per Mile MAE"].mean()
)

print("\nAVERAGE PERFORMANCE")

print(
    "Direct Rate Average MAE:",
    direct_average
)

print(
    "Rate Per Mile Average MAE:",
    rpm_average
)


# Step 9: Report the better target strategy
print("\n" + "=" * 70)
print("WINNER")
print("=" * 70)

if direct_average < rpm_average:

    print("DIRECT RATE MODEL")

    print(
        "Improvement:",
        rpm_average - direct_average
    )

else:

    print("RATE PER MILE MODEL")

    print(
        "Improvement:",
        direct_average - rpm_average
    )