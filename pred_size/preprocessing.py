import pandas as pd
import json
import re


# def load_json_as_dataframe(filename):
#     data = []
#     with open(filename, "r", encoding="utf-8") as file:
#         for line in file:
#             try:
#                 data.append(json.loads(line))  # Load each JSON object separately
#             except json.JSONDecodeError as e:
#                 print(f"Skipping invalid line: {line} - Error: {e}")
#     return pd.DataFrame(data)

# modcloth_df = load_json_as_dataframe("modcloth_final_data.json")
# renttherunway_df = load_json_as_dataframe("renttherunway_final_data.json")
# modcloth_df.to_csv("modcloth_data.csv", index=False)
# renttherunway_df.to_csv("renttherunway_data.csv", index=False)


# print(modcloth_df.head())
# print(renttherunway_df.head())# Number of rows and columns
load_csv = pd.read_csv("modcloth_data.csv")
load_csv2 = pd.read_csv("renttherunway_data.csv")
# print("ModCloth Dataset Shape:", load_csv.shape)
# print("RentTheRunway Dataset Shape:", load_csv2.shape)

# print("ModCloth Columns:", load_csv.columns.tolist())
# print("RentTheRunway Columns:", load_csv2.columns.tolist())

# print(load_csv.dtypes)
# print(load_csv2.dtypes)
# Convert height (ft/in to cm)

def convert_height_to_cm(height_str):
    """Convert height from feet/inches (e.g., '5ft 2in') to cm."""
    if isinstance(height_str, str):
        match = re.match(r"(\d+)\s*ft\s*(\d*)\s*in?", height_str.lower())  # Match 'ft' and 'in'
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2)) if match.group(2) else 0  # Handle missing inches
            return round((feet * 30.48) + (inches * 2.54), 2)  # Convert to cm
    return None  # Return None for missing or incorrect values

load_csv["bust"] = pd.to_numeric(load_csv["bust"], errors="coerce")
load_csv["height_cm"] = load_csv["height"].apply(convert_height_to_cm)
# Convert shoe size NaNs to mode (most frequent)
load_csv["shoe size"].fillna(load_csv["shoe size"].mode()[0], inplace=True)

load_csv["shoe width"] = load_csv["shoe width"].astype("category")
load_csv["cup size"] = load_csv["cup size"].astype("category")
load_csv["fit"] = load_csv["fit"].astype("category")
load_csv["category"] = load_csv["category"].astype("category")

load_csv.drop(columns=["height"], inplace=True)

# print(load_csv.head())

#renttherunway
def split_bust_size(bust_size):
    """Extracts numeric bust measurement and cup size letter."""
    if isinstance(bust_size, str):
        match = re.match(r"(\d+)([a-zA-Z]*)", bust_size)  # Extract number and letters
        if match:
            bust = int(match.group(1)) if match.group(1) else None
            cup_size = match.group(2).upper() if match.group(2) else None
            return bust, cup_size
    return None, None

load_csv2["bust"], load_csv2["cup size"] = zip(*load_csv2["bust size"].apply(split_bust_size))


# Convert weight to float
def convert_weight_to_kg(weight_str):
    """Convert weight from lbs to kg."""
    if isinstance(weight_str, str) and "lbs" in weight_str:
        try:
            lbs = float(weight_str.replace("lbs", "").strip())
            return round(lbs * 0.453592, 2)  # Convert lbs to kg
        except ValueError:
            return None
    return None

load_csv2["weight_kg"] = load_csv2["weight"].apply(convert_weight_to_kg)


# Convert height to cm
# load_csv2["height_cm"] = load_csv2["height"].apply(convert_height_to_cm)

# Convert review_date to datetime
load_csv2["review_date"] = pd.to_datetime(load_csv2["review_date"], errors="coerce")

# Convert categorical variables
load_csv2["category"] = load_csv2["category"].astype("category")
load_csv2["fit"] = load_csv2["fit"].astype("category")
load_csv2["body type"] = load_csv2["body type"].astype("category")

# Handle incorrect ages (remove rows where age > 100)
load_csv2 = load_csv2[load_csv2["age"] <= 100]

# Drop unnecessary columns
load_csv2.drop(columns=["bust size"], inplace=True)
load_csv2.drop(columns=["height"], inplace=True)
load_csv2.drop(columns=["weight"], inplace=True)

numerical_columns = ["bust", "weight_kg", "height_cm", "waist", "hips", "shoe size", "size", "quality"]
for col in numerical_columns:
    if col in load_csv.columns:
        load_csv[col] = pd.to_numeric(load_csv[col], errors="coerce")
    if col in load_csv2.columns:
        load_csv2[col] = pd.to_numeric(load_csv2[col], errors="coerce")


categorical_columns = ["cup size", "shoe width", "fit", "category", "body type"]
for col in categorical_columns:
    if col in load_csv.columns:
        load_csv[col] = load_csv[col].astype("category")
    if col in load_csv2.columns:
        load_csv2[col] = load_csv2[col].astype("category")


# print(load_csv2.head())
load_csv2.to_csv("renttherunway_cleaned.csv", index=False)
load_csv.to_csv("modcloth_cleaned.csv", index=False)

load_csv3 = pd.read_csv("modcloth_cleaned.csv")
load_csv4 = pd.read_csv("renttherunway_cleaned.csv")
print("ModCloth Dataset Shape:", load_csv3.shape)
print("RentTheRunway Dataset Shape:", load_csv4.shape)

print("ModCloth Columns:", load_csv3.columns.tolist())
print("RentTheRunway Columns:", load_csv4.columns.tolist())

print(load_csv3.dtypes)
print(load_csv4.dtypes)
