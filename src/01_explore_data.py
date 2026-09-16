import pandas as pd


train = pd.read_csv("data/train_test.csv")
validation = pd.read_csv("data/validation.csv")
december = pd.read_csv("data/december_chart_inputs.csv")


print("TRAIN SHAPE:")
print(train.shape)

print("\nVALIDATION SHAPE:")
print(validation.shape)

print("\nDECEMBER SHAPE:")
print(december.shape)

print("\nTRAIN COLUMNS:")
print(train.columns.tolist())

print("\nFIRST 5 TRAINING ROWS:")
print(train.head())

print("\nDATA TYPES:")
print(train.dtypes)

print("\nMISSING VALUES:")
print(train.isnull().sum())

print("\nTARGET SUMMARY:")
print(train["posted_rate"].describe())

print("\nEQUIPMENT VALUES:")
print(train["equipment"].value_counts())

print("\nDATE RANGE:")
print(train["date"].min(), "to", train["date"].max())