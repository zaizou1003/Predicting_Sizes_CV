import pandas as pd
import numpy as np

# Load both datasets
df_small = pd.read_csv("pred_size/4colums_size.csv")
merged_df = pd.read_csv("pred_size/final_fashion_dataset.csv")

def map_size(size):
    try:
        size = float(size)
    except:
        return np.nan
    if size <= 4:
        return "XS"
    elif size <= 12:
        return "S"
    elif size <= 16:
        return "M"
    elif size <= 22:
        return "L"
    else:
        return "XL"
def normalize_size_string(size):
    size = str(size).upper().strip()

    if size in  ["XXL", "XXXL"]:
        return "XL"
    elif size == "XXS":
        return "XS"
    else:
        return size

# Apply mapping to the size column
merged_df["size"] = merged_df["size"].apply(map_size)
df_small["size"] = df_small["size"].apply(normalize_size_string)

# Rename columns FIRST
df_small.rename(columns={
    "weight": "weight_kg_rent",
    "age": "age_rent",
    "height": "height_cm"
}, inplace=True)

# Fill missing values AFTER renaming
df_small["age_rent"] = df_small["age_rent"].astype(float)  # just in case
df_small["age_rent"].fillna(df_small["age_rent"].mean(), inplace=True)

df_small["height_cm"] = df_small["height_cm"].astype(float)
df_small["height_cm"].fillna(df_small["height_cm"].mean(), inplace=True)

# Add missing columns from the big dataset
for col in merged_df.columns:
    if col not in df_small.columns:
        df_small[col] = np.nan

# Reorder columns to match
df_small = df_small[merged_df.columns]

# Concatenate both datasets
full_df = pd.concat([merged_df, df_small], ignore_index=True)

# Fix numeric missing values after merge
full_df["bust"] = pd.to_numeric(full_df["bust"], errors="coerce")
full_df["bust"].fillna(full_df["bust"].mean(), inplace=True)

full_df["hips_modcloth"] = pd.to_numeric(full_df["hips_modcloth"], errors="coerce")
full_df["hips_modcloth"].fillna(full_df["hips_modcloth"].mean(), inplace=True)

# Drop unused IDs
full_df.drop(columns=["item_id", "user_id"], inplace=True)

print(full_df["size"].unique())


# Check missing values
print(full_df.isnull().sum())
print(full_df["size"].unique())
full_df.to_csv("training.csv", index=False)

