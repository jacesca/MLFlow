# Import needed libraries
import pandas as pd
import numpy as np
import mlflow
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# prepare the environment
SEED = 42
np.random.seed(SEED)

# Reading data
df = pd.read_csv('insurance.csv')
df = df.drop(columns=['region'])
df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(float)
df['smoker'] = df['smoker'].map({'yes': 0, 'no': 1})
print(df.head())

X = df[["age", "bmi", "children", "smoker", "charges"]].astype(float)
y = df[['sex']]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=SEED)

# Save data
X_train.to_parquet("X_train.parquet")
X_test.to_parquet("X_test.parquet")
y_train.to_parquet("y_train.parquet")
y_test.to_parquet("y_test.parquet")

# Training Data
X_train = pd.read_parquet("X_train.parquet")
y_train = pd.read_parquet("y_train.parquet")
y_train = y_train.sex

# Parameters
n_jobs_param = int(sys.argv[1])  # 1
intercept_param = bool(sys.argv[2])  # True

# Start a run # with mlflow.start_run() as run:
run = mlflow.start_run()
run_id = run.info.run_id
print(f'Run Id (1): {run_id}')

# Train the model
model = LogisticRegression(n_jobs=n_jobs_param,
                           fit_intercept=intercept_param,
                           max_iter=200)
model.fit(X_train, y_train)
score = model.score(X_train, y_train)
print(model)
print('Train Score:', score)

# Save the model
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path='model',
)

# Line command
# $ python train_insurance_model.py 1 True
