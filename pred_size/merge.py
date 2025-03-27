import pandas as pd




# Load the final cleaned datasets
modcloth_final_df = pd.read_csv("modcloth_final.csv")
renttherunway_final_df = pd.read_csv("renttherunway_final.csv")

# Drop unnecessary columns from ModCloth
modcloth_final_df = modcloth_final_df.drop(columns=["quality", "user_name", "review_text", "waist","bust", "shoe width"], errors="ignore")

# Drop unnecessary columns from Rent The Runway
renttherunway_final_df = renttherunway_final_df.drop(columns=["review_text", "rating", "rented for", "review_date"], errors="ignore")

#filling gaps in the data modcloth
modcloth_final_df["cup size"].fillna("Unknown", inplace=True)
modcloth_final_df["hips"].fillna(modcloth_final_df["hips"].median(), inplace=True)
modcloth_final_df["length"].fillna(modcloth_final_df["length"].mode()[0], inplace=True)  # Most frequent value
modcloth_final_df["review_summary"].fillna("No review", inplace=True)
modcloth_final_df["height_cm"].fillna(modcloth_final_df["height_cm"].median(), inplace=True)
modcloth_final_df["bra size"].fillna(modcloth_final_df["bra size"].median(), inplace=True)
#filling gaps in the data renttherunway
renttherunway_final_df["body type"].fillna("Unknown", inplace=True)
renttherunway_final_df["review_summary"].fillna("No review", inplace=True)
renttherunway_final_df["bust"].fillna(renttherunway_final_df["bust"].median(), inplace=True)
renttherunway_final_df["cup size"].fillna("Unknown", inplace=True)
renttherunway_final_df["weight_kg"].fillna(renttherunway_final_df["weight_kg"].median(), inplace=True)
renttherunway_final_df["height_cm"].fillna(renttherunway_final_df["height_cm"].median(), inplace=True)


# Save the cleaned ModCloth dataset
modcloth_final_df.to_csv("modcloth_last.csv", index=False)

# Save the cleaned RentTheRunway dataset
renttherunway_final_df.to_csv("renttherunway_last.csv", index=False)


# Load the cleaned datasets
modcloth_ = pd.read_csv("modcloth_last.csv")
renttherunway_ = pd.read_csv("renttherunway_last.csv")

# Perform an outer join on 'item_id' to keep all items from both datasets
merged_df = pd.merge(modcloth_, renttherunway_, on="item_id", how="outer", suffixes=('_modcloth', '_rent'))

# Merge common columns by filling missing values from either dataset
common_columns = ["size", "cup size", "category", "fit", "review_summary", "height_cm", "user_id"]

for col in common_columns:
    merged_df[col] = merged_df[f"{col}_modcloth"].fillna(merged_df[f"{col}_rent"])
    merged_df.drop(columns=[f"{col}_modcloth", f"{col}_rent"], inplace=True)


# Merge 'bra size' with 'bust' into a single column
merged_df["bust"] = merged_df["bra size"].fillna(merged_df["bust"])
merged_df.drop(columns=["bra size"], inplace=True)

# Rename unique columns to keep them distinct
merged_df.rename(columns={
    "hips": "hips_modcloth",
    "shoe size": "shoe_size_modcloth",
    "length": "length_modcloth",
    "weight_kg": "weight_kg_rent",
    "body type": "body_type_rent",
    "age": "age_rent"
}, inplace=True)
merged_df = merged_df.drop(columns=[ "shoe_size_modcloth"], errors="ignore")


# Save the final cleaned and merged dataset
merged_df.to_csv("final_merged_fashion_dataset.csv", index=False)

print(merged_df.isnull().sum())

