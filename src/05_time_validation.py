import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# 1. Load data

data = pd.read_csv("data/train_test.csv")

data["date"] = pd.to_datetime(data["date"])



# 2. Feature engineering

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



# 3. Features

features = [
    "pickup",
    "delivery",
    "route",

    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",

    "distance",
    "equipment",
    "weight",

    "market_index",
    "quote_signal",

    "month",
    "day_of_week",
    "day_of_year",
]


categorical_features = [
    "pickup",
    "delivery",
    "route",
    "equipment",
]


# 4. Define time-based folds

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



# 5. Run each fold

results = []


for fold in folds:

    print("\n" + "=" * 60)
    print("VALIDATING:", fold["name"])
    print("=" * 60)

    train_data = data[
        data["date"] < fold["train_end"]
    ].copy()

    test_data = data[
        (data["date"] >= fold["test_start"])
        & (data["date"] < fold["test_end"])
    ].copy()

    train_prepared = prepare_features(train_data)
    test_prepared = prepare_features(test_data)

    X_train = train_prepared[features]
    y_train = train_prepared["posted_rate"]

    X_test = test_prepared[features]
    y_test = test_prepared["posted_rate"]

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

  
    # 6. Model

    model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        random_seed=42,
        verbose=False
    )


    # 7. Train

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features
    )


    # 8. Predict

    predictions = model.predict(X_test)


    # 9. Evaluate

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    results.append({
        "Month": fold["name"],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# 10. Results table

results_df = pd.DataFrame(results)


print("\n" + "=" * 60)
print("TIME VALIDATION RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)



# 11. Average performance

print("\nAVERAGE PERFORMANCE")

print(
    "Average MAE:",
    results_df["MAE"].mean()
)

print(
    "Average RMSE:",
    results_df["RMSE"].mean()
)

print(
    "Average R2:",
    results_df["R2"].mean()
)