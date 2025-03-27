from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
import shutil
import os
import torch
from catboost import CatBoostClassifier
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import numpy as np
import joblib


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define model with 6 input channels (2 images concatenated along channel axis)
vision_model = models.efficientnet_b0(weights=None)
vision_model.features[0][0] = nn.Conv2d(
    in_channels=6,
    out_channels=32,
    kernel_size=3,
    stride=2,
    padding=1,
    bias=False
)
vision_model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(vision_model.classifier[1].in_features, 4)  # assuming 4 output target
)
vision_model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu", weights_only=True))
vision_model.eval()

# Load the size prediction model (e.g., sklearn or catboost model)
size_model = CatBoostClassifier()
size_model.load_model("models/size_predictor.cbm")
target_scaler = joblib.load("vision_model/target_scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)

def predict_from_images(file_paths: List[str], age: float) -> List[str]:
    front_tensor = preprocess_image(file_paths[0])
    side_tensor = preprocess_image(file_paths[1])

    # Concatenate along channel axis to form 6-channel input
    combined_tensor = torch.cat((front_tensor, side_tensor), dim=0).unsqueeze(0)

    with torch.no_grad():
        features = vision_model(combined_tensor)  # output from EfficientNet
    
    # Inverse-transform the 4 outputs
    features_np = features.numpy().flatten().reshape(1, -1)
    denorm_features = target_scaler.inverse_transform(features_np).flatten()

    height = denorm_features[0]
    hips = denorm_features[1]
    chest = denorm_features[2]
    weight = denorm_features[3]
        
    # Compute BMI
    bmi = weight / ((height / 100) ** 2)

    # Final input to size model
    final_features = np.array([height, hips, chest, weight, age, bmi]).reshape(1, -1)

    # Get class probabilities
    probs = size_model.predict_proba(final_features)[0]  # Shape: (num_classes,)

    # Get the class with the highest probability
    predicted_class = np.argmax(probs)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    class_probs = dict(zip(label_encoder.classes_, probs))


    print("Raw EfficientNet features:", features_np[:10])  # print first 10 elements
    print("Age:", age)
    print("Computed BMI:", bmi)
    print("Denormalized features:", denorm_features)
    print("Features fed to CatBoost:", final_features )
    print("Predicted Label:", predicted_label)
    print("Class Probabilities:", class_probs)
    # print("Raw CatBoost prediction:", size_prediction)
    # print("Predicted class ID:", size_prediction)
    print("Decoded label:", predicted_label)

    return [f"Predicted size: {predicted_label}"]

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...), age: float = Form(...)):
    if len(files) != 2:
        return JSONResponse(content={"error": "Exactly 2 images are required."}, status_code=400)

    file_paths = []
    for file in files:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(file_location)

    predictions = predict_from_images(file_paths, age)
    return {"predictions": predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)