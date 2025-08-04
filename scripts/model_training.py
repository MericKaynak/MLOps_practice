import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

mlflow.set_tracking_uri("http://localhost:5000/")

train_path = "../data/train.csv"
val_path = "../data/validation.csv"
test_path = "../data/test.csv"

# Daten laden (dieser Teil bleibt gleich)
train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)
test_data = pd.read_csv(test_path)

# Datenaufbereitung (dieser Teil bleibt gleich)
X_train = train_data.drop("medhouseval", axis=1)
y_train = train_data["medhouseval"]

X_val = val_data.drop("medhouseval", axis=1)
y_val = val_data["medhouseval"]

X_test = test_data.drop("medhouseval", axis=1)
y_test = test_data["medhouseval"]


def objective(trial):
    # Hyperparameter, die Optuna optimiert
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_preds)

    # MLflow tracking for each trial
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        })
        mlflow.log_metric("val_mse", val_mse)

    return val_mse


if __name__ == "__main__":
    mlflow.set_experiment("RandomForest_Housing")

    # Start main run for the entire experiment
    with mlflow.start_run(run_name="hyperparameter_optimization_and_final_model") as parent_run:

        train_dataset_log = mlflow.data.from_pandas(
            train_data,
            source=train_path,
            targets="medhouseval",
            name="housing_train_data"
        )
        val_dataset_log = mlflow.data.from_pandas(
            val_data,
            source=val_path,
            targets="medhouseval",
            name="housing_validation_data"
        )
        test_dataset_log = mlflow.data.from_pandas(
            test_data,
            source=test_path,
            targets="medhouseval",
            name="housing_test_data"
        )

        mlflow.log_input(train_dataset_log, context="training")
        mlflow.log_input(val_dataset_log, context="validation")
        mlflow.log_input(test_dataset_log, context="testing")

        mlflow.log_param("train_data_path", train_path)
        mlflow.log_param("val_data_path", val_path)
        mlflow.log_param("test_data_path", test_path)

        # Dataset statistics logging (bleibt gleich)
        mlflow.log_param("train_samples", len(train_data))
        mlflow.log_param("val_samples", len(val_data))
        mlflow.log_param("test_samples", len(test_data))
        mlflow.log_param("n_features", len(X_train.columns))
        mlflow.log_param("feature_names", list(X_train.columns))

        # Hyperparameter optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)  # n_trials auf 5 reduziert für schnellere Tests

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Log Optuna results
        mlflow.log_metric("best_val_mse", trial.value)
        mlflow.log_param("n_trials", 5)  # Angepasst an die tatsächliche Anzahl
        mlflow.log_param("optimization_direction", "minimize")

        # Train best model with optimal hyperparameters
        best_params = trial.params
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        best_model.fit(X_train_val, y_train_val)
        test_preds = best_model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_preds)
        test_r2 = best_model.score(X_test, y_test)

        # Log final model metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_param("final_training_samples", len(X_train_val))

        # ### GEÄNDERT ### DVC-Info aus Metadaten und Tags entfernt
        model_info = mlflow.sklearn.log_model(
            best_model,
            "random_forest_model",
            signature=mlflow.models.infer_signature(X_test, test_preds),
            input_example=X_test.head(5),
            metadata={
                "optimization_trials": 5,
                "best_val_mse": trial.value,
                "training_data_size": len(X_train_val)
            }
        )

        run_id = parent_run.info.run_id
        model_uri = f"runs:/{run_id}/random_forest_model"

        model_name = "RandomForestHousingModel"
        model_details = mlflow.register_model(
            model_uri,
            model_name,
            tags={
                "data_source": "local_csv",  # Geändert von "DVC_managed"
                "optimization_method": "optuna",
                "validation_metric": "mse"
            }
        )

        print(f"Model registered: {model_name}")
        print(f"   Version: {model_details.version}")
        print(f"   Run ID: {run_id}")

    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Experiment completed successfully!")