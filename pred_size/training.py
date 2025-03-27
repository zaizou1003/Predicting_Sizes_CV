import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import optuna


# Load your final cleaned dataset
df = pd.read_csv("pred_size/training.csv")
# ✅ Add BMI
df["bmi"] = df["weight_kg_rent"] / ((df["height_cm"] / 100) ** 2)

# ---------------------------------------
# 1. Define features and target
# ---------------------------------------
target_col = "size"

# Numerical features
numerical_features = [
    "height_cm", "weight_kg_rent", "age_rent", "hips_modcloth", "bust","bmi"
]

# Categorical features
categorical_features = [
    "length_modcloth", "body_type_rent", "cup size", "category", "fit", "review_summary"
]

features = numerical_features + categorical_features

# ---------------------------------------
# 2. Encode target labels (sizes)
# ---------------------------------------
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

# Count samples per class
import numpy as np
class_counts = np.bincount(df[target_col])
total = sum(class_counts)

# Compute inverse frequency weights
class_weights = [total / (len(class_counts) * count) for count in class_counts]
print("Class weights:", class_weights)

# ---------------------------------------
# 3. Train/Test Split
# ---------------------------------------
X = df[features]
y = df[target_col]


# ---------------------------------------
# CLEAN NaNs in categorical features
# ---------------------------------------
for col in categorical_features:
    X[col] = X[col].astype(str).fillna("missing")
    
cat_features_index = [X.columns.get_loc(col) for col in categorical_features]

# ---------------------------------------
# 6. Optuna Hyperparameter Optimization
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

        for col in categorical_features:
            X_train_cv.loc[:, col] = X_train_cv[col].astype(str).fillna("missing")
            X_val_cv.loc[:, col] = X_val_cv[col].astype(str).fillna("missing")

        train_pool = Pool(X_train_cv, y_train_cv, cat_features=cat_features_index)
        val_pool = Pool(X_val_cv, y_val_cv, cat_features=cat_features_index)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

        preds = model.predict(X_val_cv).flatten().astype(int)
        acc = accuracy_score(y_val_cv, preds)
        scores.append(acc)

    return np.mean(scores)


# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Print best results
print("\n✅ Best Accuracy: {:.2%}".format(study.best_value))
print("🔧 Best Parameters:", study.best_params)


# ---------------------------------------
# 5. Train the CatBoostClassifier
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

# Split one last time for final model training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

for col in categorical_features:
    X_train[col] = X_train[col].astype(str).fillna("missing")
    X_test[col] = X_test[col].astype(str).fillna("missing")

train_pool = Pool(X_train, y_train, cat_features=cat_features_index)
test_pool = Pool(X_test, y_test, cat_features=cat_features_index)

model = CatBoostClassifier(**best_params)
model.fit(train_pool, eval_set=test_pool)

# ---------------------------------------
# 6. Evaluation
# ---------------------------------------
y_pred = model.predict(X_test)
y_pred = y_pred.flatten().astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------
# 7. Save model (optional)
# ---------------------------------------
model.save_model("size_predictor.cbm")
