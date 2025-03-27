import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 👇 If BodyM_Dataset, TARGETS, and transform are in train.py
from train import BodyM_Dataset, TARGETS, transform

# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = "vision_model/normalized_testB_df.csv"  # change to test_B.csv if needed
front_img_dir = "vision_model/BodyM_Dataset/testB/mask"
side_img_dir = "vision_model/BodyM_Dataset/testB/mask_left"
SCALER_PATH = "vision_model/target_scaler.pkl"
MODEL_PATH = "vision_model/best_model.pth"
BATCH_SIZE = 32

# -------------------------------
# LOAD DATA
# -------------------------------
dataset = BodyM_Dataset(CSV_PATH, front_img_dir, side_img_dir, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = models.efficientnet_b0(pretrained=False)
model.features[0][0] = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, len(TARGETS))
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -------------------------------
# EVALUATION
# -------------------------------
all_preds, all_targets = [], []

with torch.no_grad():
    for images, targets in dataloader:
        images = images.to(device)
        outputs = model(images)
        all_preds.append(outputs.cpu())
        all_targets.append(targets)

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

# Denormalize
scaler = joblib.load(SCALER_PATH)
preds_denorm = scaler.inverse_transform(all_preds)
targets_denorm = scaler.inverse_transform(all_targets)

# Metrics
mae = mean_absolute_error(targets_denorm, preds_denorm, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm, multioutput='raw_values'))
r2 = r2_score(targets_denorm, preds_denorm, multioutput='raw_values')

# Print results
print("\n📊 Evaluation on Test Set")
print("=" * 40)
for i, t in enumerate(TARGETS):
    print(f"{t.upper():<8} | MAE: {mae[i]:.2f} | RMSE: {rmse[i]:.2f} | R²: {r2[i]:.2f}")
print("=" * 40)
print(f"🔥 Mean MAE: {mae.mean():.2f}")
