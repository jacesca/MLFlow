# Importing libraries
import pandas as pd
import mlflow
import shutil
import os
import logging

from environment import prepare_environment, print, hprint, SEED
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature


# Preparing the environment
hprint('Custom Model Creation')
# logging.getLogger("mlflow").setLevel(logging.DEBUG)
logging.getLogger("mlflow").setLevel(logging.ERROR)

prepare_environment()
experiment_name = 'custom_model'
saved_dir = 'local-saved-models'
lr_model = 'lr_model'
for dir_name in [experiment_name, lr_model]:
    if os.path.exists(f'{saved_dir}/{dir_name}'):
        shutil.rmtree(f'{saved_dir}/{dir_name}')
experiments = mlflow.search_experiments(
    filter_string=f"name = '{experiment_name}'")
for exp_data in experiments:
    exp_id = exp_data.experiment_id
    mlflow.delete_experiment(exp_id)
    print('Experiment cleaned!')


# Reading the data
print('Reading the data')
df = pd.read_csv('data/insurance.csv', index_col=0)
df['smoker'] = df.smoker == 'yes'
df['sex_encoded'] = df.sex == 'male'

# .astype('float64') >> To avoid:
#                       UserWarning: Hint: Inferred schema contains integer
#                       column(s). Integer columns in Python cannot represent
#                       missing values.
X = df[['age', 'bmi', 'children', 'smoker', 'charges']].astype('float64')
y = df['sex_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                    random_state=SEED,
                                                    stratify=y)
y_test_sex = df.loc[y_test.index]['sex']

# Set experiment
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
experiment = mlflow.set_experiment(experiment_name)
hprint(f'Experiment ID: {experiment.experiment_id}')

# Start a run # with mlflow.start_run() as run:
run = mlflow.start_run()
run_id = run.info.run_id
hprint(f'Run Id: {run_id}')

# Linear Regression model
print('Training the LogisticRegression Model...')
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

# Predict on the test set
print('Predicting...')
y_pred = lr.predict(X_test)

# Calculate metrics
print('Calculating the score...')
score = lr.score(X_test, y_test)
print(f'Score: {score}')

# Infer the model signature
signature = infer_signature(X_test, lr.predict(X_test))

# Save model to local filesystem
print('Saving the LogisticRegression Model...')
mlflow.sklearn.save_model(lr, f'{saved_dir}/{lr_model}')

# Log model to MLflow Tracking
print('Logging the LogisticRegression Model...')
mlflow.sklearn.log_model(
    sk_model=lr,
    # The convention set artifact_path='model',
    artifact_path='model',
    signature=signature,
    input_example=X_train,
    # registered_model_name=str(lr),  # Create a separate folder inside mlruns
)


#############################################################
# Creating a custom Python Class
#############################################################
class CustomPredict(mlflow.pyfunc.PythonModel):
    # Set method for loading model
    def load_context(self, context):
        print('Creating a custom Model...')
        self.model = mlflow.sklearn.load_model(f"./{lr_model}/")

    # Set method for custom inference
    def predict(self, context, model_input):
        predictions = self.model.predict(model_input)
        decoded_predictions = []
        for prediction in predictions:
            if prediction == 0:
                decoded_predictions.append("female")
            else:
                decoded_predictions.append("male")
        return decoded_predictions


print('Saving the Custom Model...')
mlflow.pyfunc.save_model(path=f'{saved_dir}/{experiment_name}',
                         python_model=CustomPredict())

print('Logging the Custom Model...')
mlflow.pyfunc.log_model(artifact_path=experiment_name,
                        artifacts={lr_model: lr_model},
                        python_model=CustomPredict())

#############################################################
# Loading a custom model
#############################################################
hprint('Loading a custom model')

# Load model from local filesystem
model = mlflow.pyfunc.load_model(f'{saved_dir}/{experiment_name}')
print("Loaded model from local filesystem:", model)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test_sex, y_pred)
print(f'Score: {accuracy}')

# Load model from MLflow Tracking
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{experiment_name}")
print("Loaded model from MLflow Tracking:", model)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test_sex, y_pred)
print(f'Score: {accuracy}')


#############################################################
# Evaluating the model
#############################################################
hprint('Evaluating the model')

# Evaluating the lr model
eval_data = X_test
eval_data["sex"] = y_test

results = mlflow.evaluate(f"runs:/{run_id}/model",
                          eval_data,
                          targets="sex",
                          model_type="classifier")
print(dir(results))
print('Baseline Model Metrics:', results.baseline_model_metrics)
print('Metrics:', results.metrics)


# Evaluating the custom model
eval_data = X_test
eval_data["sex"] = y_test_sex

results = mlflow.evaluate(f"runs:/{run_id}/{experiment_name}",
                          eval_data,
                          targets="sex",
                          model_type="classifier")
print('Metrics:', results.metrics)

# End run
mlflow.end_run()
