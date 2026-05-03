import pandas as pd
import importlib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import joblib
import argparse
import json
import os
import numpy as np
import subprocess
from preprocessing import preprocess_data


SEARCH_STRATEGIES = {
    "random_search": RandomizedSearchCV,
    "grid_search": GridSearchCV,
}


def build_model(config, registry):
    """
    Resolve and instantiate a scikit-learn model from the registry.

    Resolution order:
      1. Look up model_type in registry — class path, seed_param, default_params.
      2. Merge: registry default_params first, then experiment config fixed_params on top.
      3. Inject seed: model-level seed in registry > global default_seed in registry.

    Returns
    -------
    model       : unfitted scikit-learn estimator
    base_params : dict — merged params before search (registry defaults + fixed_params + seed)
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

    # Dynamically import the model class
    module_path, class_name = model_entry['class'].rsplit('.', 1)
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    # Merge: registry defaults + fixed_params overrides
    base_params = {
        **model_entry.get('default_params', {}),
        **config.get('fixed_params', {})
    }

    # Inject seed: model-level seed > global default_seed
    seed_param = model_entry.get('seed_param', 'random_state')
    seed = model_entry.get('seed', registry.get('default_seed', 42))
    base_params[seed_param] = seed

    return ModelClass(**base_params), base_params


def run_search(base_model, config, X_train, y_train):
    """
    Run a hyperparameter search using the strategy specified in the experiment config.

    Parameters
    ----------
    base_model  : unfitted scikit-learn estimator (with fixed_params + seed applied)
    config      : experiment config dict
    X_train     : feature matrix
    y_train     : target vector

    Returns
    -------
    searcher    : fitted CV search object (RandomizedSearchCV or GridSearchCV)
    """
    strategy = config.get('search_strategy')
    if strategy not in SEARCH_STRATEGIES:
        available = list(SEARCH_STRATEGIES.keys())
        raise ValueError(
            f"Search strategy '{strategy}' not recognised. "
            f"Available strategies: {available}\n"
            f"Please correct 'search_strategy' in your experiment config."
        )

    search_space = config.get('search_space', {})
    scoring = config.get('scoring', 'neg_mean_squared_error')
    cv = config.get('cv', 5)

    SearchClass = SEARCH_STRATEGIES[strategy]

    # Build kwargs — n_iter only applies to RandomizedSearchCV
    kwargs = dict(
        estimator=base_model,
        param_distributions=search_space if strategy == "random_search" else None,
        param_grid=search_space if strategy == "grid_search" else None,
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=-1,
        verbose=1,
    )
    # Remove the inapplicable param key
    if strategy == "random_search":
        del kwargs['param_grid']
        kwargs['n_iter'] = config.get('n_iter', 20)
    else:
        del kwargs['param_distributions']

    try:
        searcher = SearchClass(**kwargs)
        searcher.fit(X_train, y_train)
    except ValueError as e:
        raise ValueError(
            f"Search failed — check 'scoring' and 'search_space' in your experiment config.\n"
            f"sklearn error: {e}"
        )

    return searcher


def extract_search_results(searcher):
    """Extract per-iteration results from cv_results_, sorted by rank."""
    cv_results = searcher.cv_results_
    results = []
    for i in range(len(cv_results['params'])):
        results.append({
            "rank": int(cv_results['rank_test_score'][i]),
            "params": cv_results['params'][i],
            "mean_cv_score": round(float(cv_results['mean_test_score'][i]), 6),
            "std_cv_score": round(float(cv_results['std_test_score'][i]), 6),
        })
    results.sort(key=lambda x: x['rank'])
    return results


def train_model(train_data, model_output, config_file, registry_file, metrics_file):
    data = pd.read_csv(train_data)
    X, y = preprocess_data(data)

    with open(config_file) as f:
        config = json.load(f)
    with open(registry_file) as f:
        registry = json.load(f)

    # Build base model (fixed params + seed from registry)
    base_model, base_params = build_model(config, registry)

    print(f"\nModel           : {config['model_type']}")
    print(f"Search strategy : {config.get('search_strategy')}")
    print(f"Scoring         : {config.get('scoring')}")
    print(f"Fixed params    : {base_params}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run hyperparameter search on training split
    searcher = run_search(base_model, config, X_train, y_train)

    best_params = searcher.best_params_
    best_cv_score = round(float(searcher.best_score_), 6)
    search_results = extract_search_results(searcher)

    # Full resolved params for the best model: base + best found by search
    resolved_params = {**base_params, **best_params}

    print(f"\nBest CV score   : {best_cv_score:.6f}")
    print(f"Best params     : {best_params}")

    # Evaluate best model on train and validation splits
    best_model = searcher.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    target_names = ['Predicted Not Survive', 'Predicted Survive']
    training_report = classification_report(y_train, y_train_pred, target_names=target_names)
    validation_report = classification_report(y_val, y_val_pred, target_names=target_names)

    with open("training_classification_report.txt", "w") as f:
        f.write(training_report)
    with open("validation_classification_report.txt", "w") as f:
        f.write(validation_report)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = float(np.sqrt(train_mse))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = float(np.sqrt(val_mse))
    val_mae = mean_absolute_error(y_val, y_val_pred)

    print(f"\nTraining   MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    print(f"Validation MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

    # Save best model
    joblib.dump(best_model, model_output)
    print(f"Model saved to {model_output}")

    # Write extended metrics JSON
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    metrics = {
        "report_type": "training",
        "model_type": config['model_type'],
        "search_strategy": config.get('search_strategy'),
        "scoring": config.get('scoring'),
        "n_iter": config.get('n_iter'),
        "cv": config.get('cv'),
        "best_cv_score": best_cv_score,
        "best_params": best_params,
        "hyperparameters": resolved_params,
        "search_results": search_results,
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

    # Generate PDF report
    subprocess.run(["python3", "generate_pdf.py", "--report", "training_report.pdf", "--metrics", metrics_file])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model via hyperparameter search.")
    parser.add_argument('--train', type=str, help="Path to the training dataset.")
    parser.add_argument('--model_output', type=str, help="Path to save the trained model.")
    parser.add_argument('--config', type=str, help="Path to the experiment configuration JSON file.")
    parser.add_argument('--registry', type=str, help="Path to the model registry JSON file.")
    parser.add_argument('--metrics', type=str, help="Path to save the metrics JSON file.")

    args = parser.parse_args()
    train_model(args.train, args.model_output, args.config, args.registry, args.metrics)
