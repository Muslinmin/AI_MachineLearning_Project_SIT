import pandas as pd
import importlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import joblib
import argparse
import json
import os
import numpy as np
import subprocess
from preprocessing import preprocess_data


def build_model(config, registry):
    """
    Resolve and instantiate a scikit-learn model from the registry.

    Resolution order:
      1. Look up model_type in registry to get class path, seed_param, default_params.
      2. Merge: registry default_params first, then experiment config model_params on top.
      3. Inject seed: model-level seed in registry > global default_seed in registry.

    Returns
    -------
    model : fitted-ready scikit-learn estimator
    resolved_params : dict — the full merged parameter set actually used
    """
    model_type = config.get('model_type')
    if not model_type:
        raise ValueError("'model_type' key is missing from the experiment config.")

    if model_type not in registry['models']:
        available = list(registry['models'].keys())
        raise ValueError(
            f"Model type '{model_type}' not found in registry. "
            f"Available models: {available}"
        )

    model_entry = registry['models'][model_type]

    # Dynamically import the model class from its fully qualified path
    module_path, class_name = model_entry['class'].rsplit('.', 1)
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    # Merge params: registry defaults first, experiment config overrides on top
    resolved_params = {
        **model_entry.get('default_params', {}),
        **config.get('model_params', {})
    }

    # Inject seed: model-level seed > global default_seed
    seed_param = model_entry.get('seed_param', 'random_state')
    seed = model_entry.get('seed', registry.get('default_seed', 42))
    resolved_params[seed_param] = seed

    return ModelClass(**resolved_params), resolved_params


def train_model(train_data, model_output, config_file, registry_file, metrics_file):
    data = pd.read_csv(train_data)
    X, y = preprocess_data(data)

    with open(config_file) as f:
        config = json.load(f)
    with open(registry_file) as f:
        registry = json.load(f)

    # Build model via registry — no hardcoded model class
    model, resolved_params = build_model(config, registry)
    print(f"\nModel     : {config['model_type']}")
    print(f"Parameters: {resolved_params}")

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

    # Write metrics JSON with fully resolved params
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    metrics = {
        "report_type": "training",
        "model_type": config['model_type'],
        "hyperparameters": resolved_params,
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
    parser = argparse.ArgumentParser(description="Train a model defined by the registry and experiment config.")
    parser.add_argument('--train', type=str, help="Path to the training dataset.")
    parser.add_argument('--model_output', type=str, help="Path to save the trained model.")
    parser.add_argument('--config', type=str, help="Path to the experiment configuration JSON file.")
    parser.add_argument('--registry', type=str, help="Path to the model registry JSON file.")
    parser.add_argument('--metrics', type=str, help="Path to save the metrics JSON file.")

    args = parser.parse_args()
    train_model(args.train, args.model_output, args.config, args.registry, args.metrics)
