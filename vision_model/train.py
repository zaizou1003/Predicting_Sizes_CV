import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm


# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = "vision_model/normalized_train_df.csv"
front_img_dir="vision_model/BodyM_Dataset/train/mask"
side_img_dir="vision_model/BodyM_Dataset/train/mask_left"
SCALER_PATH = "vision_model/target_scaler.pkl"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
TARGETS = ["height", "hip", "chest", "weight"]
best_mae = float("inf")
patience = 10
epochs_no_improve = 0
early_stop = False
# -------------------------------
# DATASET
# -------------------------------
class BodyM_Dataset(Dataset):
    def __init__(self, csv_path, front_img_dir, side_img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.front_img_dir = front_img_dir
        self.side_img_dir = side_img_dir
        self.transform = transform
        self.targets = TARGETS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        for _ in range(len(self.df)):  # retry up to dataset size
            row = self.df.iloc[idx]
            img_name = row["photo_id"]
            if not img_name.endswith(".png"):
                img_name += ".png"        
            
            front_path = os.path.join(self.front_img_dir, img_name)
            side_path = os.path.join(self.side_img_dir, img_name)

            if not os.path.exists(front_path) or not os.path.exists(side_path):
                idx = (idx + 1) % len(self.df)
                continue

            try:
                front_img = Image.open(front_path).convert("RGB")
                side_img = Image.open(side_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
                idx = (idx + 1) % len(self.df)
                continue

            if self.transform:
                front_img = self.transform(front_img)
                side_img = self.transform(side_img)

            combined_img = torch.cat((front_img, side_img), dim=0)
            target = torch.tensor(row[self.targets].values.astype(float), dtype=torch.float32)
            return combined_img, target

        raise RuntimeError("No valid images found in dataset.")

# -------------------------------
# TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # perspective changes
    transforms.RandomRotation(degrees=10),              # small rotations
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # lighting changes
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # slight shifts
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# LOAD DATA
# -------------------------------
dataset = BodyM_Dataset(CSV_PATH, front_img_dir,side_img_dir, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# MODEL
# -------------------------------
model = models.efficientnet_b0(pretrained=True)
model.features[0][0] = nn.Conv2d(
    in_channels=6,
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding=1,
    bias=False
)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, len(TARGETS))
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))
    print("✅ Loaded best model weights to resume training or fine-tune.")
if __name__ == "__main__":
    # -------------------------------
    # TRAINING
    # -------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)

    for epoch in range(EPOCHS):
        if early_stop:
            break
        model.train()
        running_loss = 0.0
        all_preds, all_targets = [], []

        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

        # Denormalize & calculate MAE
        scaler = joblib.load(SCALER_PATH)
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        preds_denorm = scaler.inverse_transform(all_preds)
        targets_denorm = scaler.inverse_transform(all_targets)

        mae = mean_absolute_error(targets_denorm, preds_denorm, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm, multioutput='raw_values'))
        r2 = r2_score(targets_denorm, preds_denorm, multioutput='raw_values')
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(dataloader):.4f}")
        for i, t in enumerate(TARGETS):
            print(f"  {t.upper()} | MAE: {mae[i]:.2f} | RMSE: {rmse[i]:.2f} | R²: {r2[i]:.2f}")

        # Save best model
        mean_mae = mae.mean()
        scheduler.step(mean_mae)
        if epoch == 0 or mean_mae < best_mae:
            best_mae = mean_mae
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  🔥 Saved new best model (Mean MAE: {mean_mae:.2f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement. ({epochs_no_improve}/{patience})")

            if epochs_no_improve >= patience:
                print("🛑 Early stopping triggered.")
                early_stop = True
                break