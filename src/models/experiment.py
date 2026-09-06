import mlflow


EXPERIMENT_NAME = "Predictive-Maintenance-Baselines"


def setup_mlflow():
    """
    Configure the MLflow experiment for baseline model tracking.
    """

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow experiment set to: {EXPERIMENT_NAME}")

def log_model_run(model_name, model, metrics):
    """
    Log a model run, parameters, and metrics to MLflow.

    Parameters
    ----------
    model_name : str
        Name of the model.

    model :
        Trained machine learning model.

    metrics : dict
        Cross-validation metrics.
    """

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)

        model_params = model.get_params()

        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        print(f"MLflow run logged successfully: {model_name}")