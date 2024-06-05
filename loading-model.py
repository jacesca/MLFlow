import mlflow
import json
from environment import prepare_environment, print


prepare_environment()

# Load model from MLFLow local filesystem
experiment_name = '50_Startups'
path_model = f'local-saved-models/{experiment_name}'
model = mlflow.sklearn.load_model(path_model)
print('Loaded model from MLFlow File System:', model)


# Load model from MLFlow Tracking
score_filter = "metrics.score > 0.60"
df = mlflow.search_runs(
        experiment_names=['Iris Experiment'],
        filter_string=score_filter,
        order_by=["metrics.score DESC"]
    )
if len(df) > 0:
    # To get the artifact path, we can do
    artifact_json = json.loads(df['tags.mlflow.log-model.history'].loc[0])
    artifact_path = artifact_json[0]['artifact_path']
    run_id = df['run_id'].loc[0]
    print("Run ID:", run_id, artifact_path)

    # model = mlflow.sklearn.load_model(f'runs:/{run_id}/{artifact_path}')
    model = mlflow.sklearn.load_model(f'runs:/{run_id}/model')
    print('Loaded model from MLFlow Tracking:', model)
