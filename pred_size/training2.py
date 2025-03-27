import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import optuna
import numpy as np
import joblib


# Load your dataset
df = pd.read_csv("pred_size/training.csv")
df["bmi"] = df["weight_kg_rent"] / ((df["height_cm"] / 100) ** 2)
# ---------------------------------------
# 1. Define features and target
# ---------------------------------------
target_col = "size"

numerical_features = [
    "height_cm", "hips_modcloth", "bust", "weight_kg_rent", "age_rent","bmi"
]

features = numerical_features

# Encode target labels (sizes)
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
joblib.dump(le, "label_encoder.pkl")

# Compute class weights
class_counts = np.bincount(df[target_col])
total = sum(class_counts)
class_weights = [total / (len(class_counts) * count) for count in class_counts]

# Split features/target
X = df[features]
y = df[target_col]

# ---------------------------------------
# 2. Optuna Hyperparameter Optimization
# ---------------------------------------
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 700),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0.1, 1.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
        "task_type": "GPU",
        "verbose": 0,
        "random_seed": 42,
        "class_weights": class_weights
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        train_pool = Pool(X_train_cv, y_train_cv)
        val_pool = Pool(X_val_cv, y_val_cv)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

        preds = model.predict(X_val_cv).flatten().astype(int)
        acc = accuracy_score(y_val_cv, preds)
        scores.append(acc)

    return np.mean(scores)

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# Best results
print("\n✅ Best Accuracy: {:.2%}".format(study.best_value))
print("🔧 Best Parameters:", study.best_params)

# ---------------------------------------
# 3. Final Training
# ---------------------------------------
best_params = study.best_params
best_params.update({
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "task_type": "GPU",
    "verbose": 100,
    "random_seed": 42,
    "class_weights": class_weights
})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

model = CatBoostClassifier(**best_params)
model.fit(train_pool, eval_set=test_pool)

# ---------------------------------------
# 4. Evaluation
# ---------------------------------------
y_pred = model.predict(X_test).flatten().astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------
# 5. Save model
# ---------------------------------------
model.save_model("size_predictor.cbm")
