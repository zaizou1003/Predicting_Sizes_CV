import pandas as pd

def prepare_bodym_data(split_path):
    map_df = pd.read_csv(f"{split_path}/subject_to_photo_map.csv")
    measurements_df = pd.read_csv(f"{split_path}/measurements.csv")
    metadata_df = pd.read_csv(f"{split_path}/hwg_metadata.csv")

    df = map_df.merge(measurements_df[["subject_id", "height", "hip", "chest"]], on="subject_id")
    df = df.merge(metadata_df[["subject_id", "weight_kg"]], on="subject_id")

    df = df.rename(columns={"weight_kg": "weight"})
    df = df[["photo_id", "height", "hip", "chest", "weight"]]
    return df

train_df = prepare_bodym_data("vision_model/BodyM_Dataset/train")
testA_df = prepare_bodym_data("vision_model/BodyM_Dataset/testA")
testB_df = prepare_bodym_data("vision_model/BodyM_Dataset/testB")

# Save them
train_df.to_csv("vision_model/cleaned_train_df.csv", index=False)
testA_df.to_csv("vision_model/cleaned_testA_df.csv", index=False)
testB_df.to_csv("vision_model/cleaned_testB_df.csv", index=False)