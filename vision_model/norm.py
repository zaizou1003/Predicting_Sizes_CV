from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Load your cleaned data
df = pd.read_csv("vision_model/cleaned_train_df.csv")
df2 = pd.read_csv("vision_model/cleaned_testA_df.csv")
df3 = pd.read_csv("vision_model/cleaned_testB_df.csv")

# Select target columns
target_cols = ["height", "hip", "chest", "weight"]

# Initialize scaler
scaler = StandardScaler()

# Fit only on training data
df[target_cols] = scaler.fit_transform(df[target_cols])
df2[target_cols] = scaler.transform(df2[target_cols])
df3[target_cols] = scaler.transform(df3[target_cols])

# Save the scaler for use during inference/test
joblib.dump(scaler, "vision_model/target_scaler.pkl")

# Save normalized data
df.to_csv("vision_model/normalized_train_df.csv", index=False)
df2.to_csv("vision_model/normalized_testA_df.csv", index=False)
df3.to_csv("vision_model/normalized_testB_df.csv", index=False)

