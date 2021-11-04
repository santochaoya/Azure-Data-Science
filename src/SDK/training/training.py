from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from scipy.sparse.construct import rand

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Get the experiment context
run = Run.get_context()

# Get the training model
data = pd.read_csv('data/wine.csv')
data['WineVariety'] = data['WineVariety'].replace(2, 1)

X, y = data[['Alcohol', 'Malic_acid']], data['WineVariety']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

reg = 0.1
model = LogisticRegression(C=1/reg, solver='liblinear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# log the accurancy
run.log('Accuracy', np.float(np.average(y_pred==y_test)))

# Save the model
joblib.dump(model, filename='outputs/model.pkl')

# Complete the experiment
run.complete()