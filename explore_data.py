import pandas as pd
import json
import os

# Load Sephora dataset (example)
df = pd.read_csv('Data/product_info.csv')
df.head(5)

#data exploration
print("shape:", df.shape)
print("data type\n",df.dtypes)
columns = df.columns.tolist()
print("Columns:", columns)

# missingness check
null_counts = round((df.isnull().sum()/ len(df))*100, 2)
print("null value counts:\n", null_counts)

# unique value check
print("Total Rows:", len(df))
for c in columns:
    unique_values = df[c].nunique()
    print(f"Column: {c}, Unique Values: {unique_values}")