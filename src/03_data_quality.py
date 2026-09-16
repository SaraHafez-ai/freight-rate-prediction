import pandas as pd


data = pd.read_csv("data/train_test.csv")


print("DUPLICATE LOAD IDs:")
print(data["load_id"].duplicated().sum())


print("\nMISSING VALUES:")
print(data.isnull().sum())


print("\nNEGATIVE WEIGHTS:")
print((data["weight"] < 0).sum())


print("\nWEIGHT RANGE:")
print("Min:", data["weight"].min())
print("Max:", data["weight"].max())


print("\nPOSTED RATE SUMMARY:")
print(data["posted_rate"].describe())


print("\nVERY HIGH RATES:")
print(
    data.sort_values("posted_rate", ascending=False)
    [["load_id", "distance", "equipment", "weight", "posted_rate"]]
    .head(10)
)