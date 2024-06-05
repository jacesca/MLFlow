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
df['smoker'] = df['smoker'].map({'yes': 0, 'no': 1}).astype(float)
print(df.info())

X = df[["age", "bmi", "children", "smoker", "charges"]].astype(float)
y = df[['sex']]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=SEED)

# Set Auto logging for Scikit-learn flavor
mlflow.sklearn.autolog()

# Parameters
n_jobs_param = int(sys.argv[1])  # 1
intercept_param = bool(sys.argv[2])  # True

# Train the model
lr = LogisticRegression(n_jobs=n_jobs_param,
                        fit_intercept=intercept_param)
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print(lr)
print('Score:', score)

# Line command
# $ python train_insurance_model.py 1 True
