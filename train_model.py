
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_xgboost_model(data_path="training_data.json", model_output_path="xgboost_heading_model.joblib"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract features and labels
    features = []
    labels = []
    for item in data:
        # For simplicity, using font_size and is_bold as features for now.
        # More sophisticated features (relative position, indentation, etc.) can be added later.
        features.append([
            item["features"]["font_size"],
            1 if item["features"]["is_bold"] else 0,
            item["features"]["bbox"][0], # x0
            item["features"]["bbox"][1], # y0
            item["features"]["bbox"][2], # x1
            item["features"]["bbox"][3], # y1
            item["features"]["page_width"],
            item["features"]["page_height"]
        ])
        labels.append(item["label"])

    X = pd.DataFrame(features, columns=["font_size", "is_bold", "x0", "y0", "x1", "y1", "page_width", "page_height"])
    y = pd.Series(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train XGBoost Classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Model Precision: {precision:.4f}")
    print(f"Model Recall: {recall:.4f}")
    print(f"Model F1-Score: {f1:.4f}")

    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    train_xgboost_model()


