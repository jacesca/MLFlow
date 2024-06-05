# Import needed libraries
import pandas as pd
import mlflow
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# prepare the environment
SEED = 42
np.random.seed(SEED)

# Reading data
df = pd.read_csv('Salary_predict.csv')
print(df.info())

X = df[['experience', 'age', 'interview_score']].astype(float)
y = df[['Salary']]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=SEED)

# Set Auto logging for Scikit-learn flavor
mlflow.sklearn.autolog()

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print(lr)
print('Score:', score)
