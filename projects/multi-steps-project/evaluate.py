# Import needed libraries
import pandas as pd
import mlflow
import sys

from pprint import pprint


# Testing Data
X_test = pd.read_parquet("X_test.parquet")
y_test = pd.read_parquet("y_test.parquet")

# Parameters
run_id = str(sys.argv[1])
print('Run ID (3):', run_id)

# Eval the model
eval_data = X_test
eval_data["sex"] = y_test

results = mlflow.evaluate(f"runs:/{run_id}/model",
                          data=eval_data,
                          targets="sex",
                          model_type="classifier")
print('Metrics:')
pprint(results.metrics)
