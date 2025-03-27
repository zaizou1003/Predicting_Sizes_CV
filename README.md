# 👕 Predicting Clothing Sizes from Images + Body Data

This project uses **Computer Vision** + **Tabular Machine Learning** to predict the most likely clothing size for a person based on:

- Front & side images of their body
- Their age (used to estimate size)
- Extracted body features (height, hips, chest, weight)
- BMI (computed internally)

---

## 📂 Project Structure

### 🔧 Core Files

- `back_end.py` — FastAPI backend that:
  - Accepts image uploads + age
  - Uses a pre-trained vision model (EfficientNet)
  - Extracts 4 key features and feeds them to a CatBoost size classifier

- `index.html` — A simple frontend to test the size prediction with file upload.

- `requirements.txt` — All required libraries for the project.

---

### 🤖 Machine Learning & Models

#### 📁 `vision_model/`
- `train.py` — Trains the EfficientNet model to predict height, hips, chest, and weight.
- `eval.py` — Evaluates the performance of the vision model on test data.
- `norm.py` — Normalizes and denormalizes data using `StandardScaler`.
- `clean_data/` — Contains cleaned CSVs for training/test.
- `BodyM_Dataset/` — Custom PyTorch Dataset class and dataloader logic.
- `target_scaler.pkl` — Used to reverse normalization during inference.

#### 📁 `pred_size/`
- `training.py` / `training2.py` — Trains the CatBoost size classifier using tabular data (age, BMI, body features).
- `merge.py` / `last_merge.py` — Scripts used to clean and merge raw CSVs.
- `check_values.py` — For debugging dataset inconsistencies.

---

## 📁 Ignored in `.gitignore`
- `.venv/`, `uploads/`, `catboost_info/`, `old_csv/` — Not committed to GitHub
- `*.csv`, `*.pth`, `*.cbm`, `*.pkl` — Large files and trained models are also ignored.

---

## 🚀 How to Use

1. Clone the repo and install dependencies:
2. Start the FastAPI backend:
3. Open `index.html` in your browser to test the prediction UI.

---

## 🙌 Author

Built with 💻 by [@zaizou1003](https://github.com/zaizou1003)

---

