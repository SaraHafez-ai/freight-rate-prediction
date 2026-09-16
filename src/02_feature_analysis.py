import pandas as pd


data = pd.read_csv("data/train_test.csv")

data["date"] = pd.to_datetime(data["date"])


# Numeric columns only
numeric_columns = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate"
]


correlation = data[numeric_columns].corr()["posted_rate"].sort_values(
    ascending=False
)

print("CORRELATION WITH POSTED RATE:")
print(correlation)

print("\nAVERAGE RATE BY EQUIPMENT:")
print(
    data.groupby("equipment")["posted_rate"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
)


print("\nTOP PICKUP CITIES BY AVERAGE RATE:")
print(
    data.groupby("pickup")["posted_rate"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTOP DELIVERY CITIES BY AVERAGE RATE:")
print(
    data.groupby("delivery")["posted_rate"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

data["rate_per_mile"] = data["posted_rate"] / data["distance"]

print("\nAVERAGE RATE PER MILE BY EQUIPMENT:")
print(
    data.groupby("equipment")["rate_per_mile"]
    .agg(["mean", "median", "count"])
)