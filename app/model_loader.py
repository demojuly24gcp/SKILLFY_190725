import mlflow
import mlflow.pyfunc

def load_model():
    # Get the latest run from the default experiment (ID = '0')
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")

    if not experiment:
        raise ValueError("No experiment named 'Default' found in MLflow.")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["start_time DESC"],
                              max_results=1)

    if not runs:
        raise ValueError("No runs found in MLflow for the default experiment.")

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Loading model from latest run: {run_id}")
    return mlflow.pyfunc.load_model(model_uri)
