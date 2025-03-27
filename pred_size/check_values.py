import pandas as pd
import matplotlib.pyplot as plt


merged_df = pd.read_csv("pred_size/final_merged_fashion_dataset.csv")

merged_df["review_summary"].fillna("unknown", inplace=True)
merged_df["body_type_rent"].fillna("unknown", inplace=True)
merged_df["hips_modcloth"].fillna(merged_df["hips_modcloth"].mean(), inplace=True)
merged_df["length_modcloth"].fillna("unknown", inplace=True)
merged_df["age_rent"].fillna(merged_df["age_rent"].mean(), inplace=True)
merged_df["weight_kg_rent"].fillna(merged_df["weight_kg_rent"].mean(), inplace=True)





print(merged_df.isnull().sum())


# List all categorical columns
categorical_columns = ["length_modcloth", "cup size", "category", "fit", "body_type_rent"]

# Display unique values for each categorical column
for col in categorical_columns:
    print(f"Unique values in {col}:")
    print(merged_df[col].unique(), "\n")

print(merged_df.describe())

# Plot histograms for numerical columns
numerical_columns = ["height_cm", "weight_kg_rent", "bust", "hips_modcloth", "age_rent", "size"]
merged_df[numerical_columns].hist(figsize=(10, 6), bins=30)
plt.show()

# Check unique values and their counts
size_counts = merged_df["size"].value_counts().sort_index()
print("Unique values in size and their frequencies:\n", size_counts)

# Plot size distribution
plt.figure(figsize=(10, 5))
plt.bar(size_counts.index, size_counts.values, color='skyblue', edgecolor='black')
plt.xlabel("Size")
plt.ylabel("Frequency")
plt.title("Distribution of Size Values")
plt.xticks(rotation=45)
plt.show()

merged_df.to_csv("final_fashion_dataset.csv", index=False)
