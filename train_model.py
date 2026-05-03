import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import joblib
import argparse
import json
import os
import numpy as np
import subprocess
from preprocessing import preprocess_data

def train_model(train_data, model_output, config_file, metrics_file):
    data = pd.read_csv(train_data)
    X, y = preprocess_data(data)

    with open(config_file) as f:
        config = json.load(f)
    model_params = config['model_params']

    # Set model parameters from config file
    model = RandomForestClassifier(
        n_estimators=model_params.get('n_estimators', 100),
        max_depth=model_params.get('max_depth', None),
        min_samples_split=model_params.get('min_samples_split', 2),
        min_samples_leaf=model_params.get('min_samples_leaf', 1),
        random_state=model_params.get('random_state', 42)
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Custom labels for the report
    target_names = ['Predicted Not Survive', 'Predicted Survive']

    # Generate classification reports
    training_report = classification_report(y_train, y_train_pred, target_names=target_names)
    validation_report = classification_report(y_val, y_val_pred, target_names=target_names)

    # Write classification reports to text files
    with open("training_classification_report.txt", "w") as f:
        f.write(training_report)
    with open("validation_classification_report.txt", "w") as f:
        f.write(validation_report)

    # Calculate MSE, RMSE, and MAE for training and validation
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = float(np.sqrt(train_mse))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = float(np.sqrt(val_mse))
    val_mae = mean_absolute_error(y_val, y_val_pred)

    print(f"\nTraining   MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    print(f"Validation MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

    # Save the model
    joblib.dump(model, model_output)
    print(f"Model saved to {model_output}")

    # Write metrics JSON
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    metrics = {
        "report_type": "training",
        "hyperparameters": model_params,
        "training": {
            "mse": round(float(train_mse), 6),
            "rmse": round(train_rmse, 6),
            "mae": round(float(train_mae), 6),
            "classification_report": training_report
        },
        "validation": {
            "mse": round(float(val_mse), 6),
            "rmse": round(val_rmse, 6),
            "mae": round(float(val_mae), 6),
            "classification_report": validation_report
        }
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    # Call generate_pdf.py to create a PDF report
    subprocess.run(["python3", "generate_pdf.py", "--report", "training_report.pdf", "--metrics", metrics_file])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model.")
    parser.add_argument('--train', type=str, help="Path to the training dataset.")
    parser.add_argument('--model_output', type=str, help="Path to save the trained model.")
    parser.add_argument('--config', type=str, help="Path to the configuration JSON file.")
    parser.add_argument('--metrics', type=str, help="Path to save the metrics JSON file.")

    args = parser.parse_args()
    train_model(args.train, args.model_output, args.config, args.metrics)
