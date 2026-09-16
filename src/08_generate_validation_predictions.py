import pandas as pd

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Step 1: Load the training, validation, and template files
train = pd.read_csv(
    "data/train_test.csv"
)

validation = pd.read_csv(
    "data/validation.csv"
)

template = pd.read_csv(
    "data/validation_predictions_template.csv"
)
# Correct negative weight values
train["weight"] = train["weight"].abs()
validation["weight"] = validation["weight"].abs()

# Rate-per-mile modeling requires positive distances
if (train["distance"] <= 0).any():
    raise ValueError(
        "Training data contains non-positive distance values."
    )

if (validation["distance"] <= 0).any():
    raise ValueError(
        "Validation data contains non-positive distance values."
    )


# Step 2: Convert the date columns
train["date"] = pd.to_datetime(
    train["date"]
)

validation["date"] = pd.to_datetime(
    validation["date"]
)


# Step 3: Create the engineered features used by the final model
def prepare_features(df):
    df = df.copy()

    # Capture weekly pricing behavior
    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    # Combine origin and destination into a lane feature
    df["route"] = (
        df["pickup"]
        + " -> "
        + df["delivery"]
    )

    return df


train = prepare_features(
    train
)

validation = prepare_features(
    validation
)


# Step 4: Define the final 11-feature set
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


# Step 5: Prepare the model inputs
X_train = train[
    features
]

X_validation = validation[
    features
]


# Train on rate per mile instead of total posted rate
y_train = (
    train["posted_rate"]
    / train["distance"]
)


print(
    "Training rows:",
    len(X_train)
)

print(
    "Validation rows:",
    len(X_validation)
)


# Step 6: Fill missing numeric values using the median
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


# Step 7: Prepare the categorical columns
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


# Step 8: Combine numeric and categorical preprocessing
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


# Step 9: Use the final tuned XGBoost configuration
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


# Step 10: Build the full preprocessing and model pipeline
pipeline = Pipeline(
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


# Step 11: Train on all 48,000 labeled rows
print(
    "\nTraining final rate-per-mile model..."
)

pipeline.fit(
    X_train,
    y_train
)


# Step 12: Predict rate per mile for the 12,000 validation loads
print(
    "Generating validation predictions..."
)

predicted_rpm = pipeline.predict(
    X_validation
)


# Convert rate per mile back into the total freight rate
predictions = (
    predicted_rpm
    * validation["distance"].to_numpy()
)


# The scorer requires every predicted rate to be positive
predictions = predictions.clip(
    min=1.0
)


# Step 13: Match every prediction to its validation load ID
prediction_lookup = pd.DataFrame({
    "load_id":
        validation["load_id"],

    "predicted_rate":
        predictions
})


# Step 14: Fill the supplied prediction template by load_id
output = (
    template[["load_id"]]
    .merge(
        prediction_lookup,
        on="load_id",
        how="left"
    )
)


# Step 15: Check the final submission before saving
if len(output) != 12000:
    raise ValueError(
        "Output does not contain 12,000 rows."
    )


if output["load_id"].duplicated().any():
    raise ValueError(
        "Duplicate load IDs found."
    )


if output["predicted_rate"].isna().any():
    raise ValueError(
        "Some validation loads were not predicted."
    )


if (output["predicted_rate"] <= 0).any():
    raise ValueError(
        "Non-positive predictions found."
    )


# Step 16: Save the final validation predictions
output.to_csv(
    "validation_predictions.csv",
    index=False
)


# Step 17: Show a quick summary
print(
    "\nSUCCESS!"
)

print(
    "Created validation_predictions.csv"
)

print(
    "Rows:",
    len(output)
)


print(
    "\nPREDICTION SUMMARY:"
)

print(
    output["predicted_rate"].describe()
)


print(
    "\nFIRST 10 PREDICTIONS:"
)

print(
    output.head(10).to_string(
        index=False
    )
)