import pandas as pd


data = pd.read_csv("data/train_test.csv")

data["date"] = pd.to_datetime(data["date"])


# -------------------------------------------
# Rate per mile
# -------------------------------------------

data["rate_per_mile"] = (
    data["posted_rate"] / data["distance"]
)


print("RATE PER MILE SUMMARY:")
print(
    data["rate_per_mile"].describe(
        percentiles=[
            0.90,
            0.95,
            0.99,
            0.995,
            0.999
        ]
    )
)


# -------------------------------------------
# Detect unusually high rate-per-mile loads
# -------------------------------------------

extreme = data[
    data["rate_per_mile"] > 5
].copy()

normal = data[
    data["rate_per_mile"] <= 5
].copy()


print("\nEXTREME ROW COUNT:")
print(len(extreme))

print("\nEXTREME PERCENTAGE:")
print(
    len(extreme) / len(data) * 100
)


# -------------------------------------------
# Compare normal vs extreme loads
# -------------------------------------------

print("\nNORMAL LOAD AVERAGES:")

print(
    normal[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
            "rate_per_mile"
        ]
    ].mean()
)


print("\nEXTREME LOAD AVERAGES:")

print(
    extreme[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
            "rate_per_mile"
        ]
    ].mean()
)


# -------------------------------------------
# Check whether spikes happen in one month
# -------------------------------------------

data["month"] = data["date"].dt.month

data["is_extreme"] = (
    data["rate_per_mile"] > 5
)


print("\nEXTREME LOADS BY MONTH:")

print(
    data.groupby("month")["is_extreme"]
    .agg(["sum", "count", "mean"])
)


# -------------------------------------------
# Show extreme examples
# -------------------------------------------

print("\nTOP 20 RATE-PER-MILE LOADS:")

columns = [
    "load_id",
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
    "posted_rate",
    "rate_per_mile"
]

print(
    data.sort_values(
        "rate_per_mile",
        ascending=False
    )[columns]
    .head(20)
    .to_string(index=False)
)