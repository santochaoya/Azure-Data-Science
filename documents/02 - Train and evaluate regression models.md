This document is to apply a regression model to a daily bike sales dataset.



# [Explore the Data](./Explore and analyze data with python.md)

Before applying the machine learning models which known as train the model, we need to explore the dataset we will use. This may require a butch of technicals to process the data. 

## Distributions

* Histogram to numeric values
* Boxplot to categorical values

## Relationships between features and labels

* Scatter plot to numeric values

* Boxplot show relationship between some category values

  

# Train a Regression Model

## Split training and testing dataset

Split the data into two subsets **randomly**.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)
```



## Train a model

Here, we use linear regression model, a common starting point for regression, which is easy to interpret and find a linear relationship between the features and labels.

```python
from sklearn.linear_model import LinearRegression

# Fit the linear regression model on training set
model = LinearRegression(X_train, y_train)
```



## Evaluate the Trained Model

Apply the model to the testing dataset to predict the labels. Compared the predicted labels to the actual labels to evaluate how well the model performed.

* Set predict labels and actual labels

  ```python
  import numpy as np
  
  # set predictions and actual labels
  predictions = model.predict(X_teset)
  ```

* Compare predict labels to the actual labels

  * Graphic - Scatter plot

    ```python
    import matplotlib.pyplot as plt
    
    # Plot scatter plot between predict labels and actual labels
    plt.scatter(y_test, predictions)
    
    # Customize
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Compared of Predictions')
    
    # Overlay the regression line
    z = np.polyfit(y_test, predictions, 1) # all the coefficient of the polynomial
    p = np.ploy1d(z)
    plt.plot(y_test, p(y_test), color='green')
    
    # Show
    plt.show()
    ```

  * Quantify the residual

    ```python
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print('MSE: {}\nRMSE: {}\nR2:{}'.format(mse, rmse, r2))
    ```



# More Regression Models

## Lasso

```python
from sklearn.linear_model import Lasso

# Fit a lasso model on the traning dataset
model = Lasso().fit(X_train, y_train)
```



## Decision Tree - an alternative linear model

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text

# Train the model
model = DecisionTreeRegressor().fit(X_train, y_train)

# Visualize the model tree
tree = export_text(model)
print(tree)
```



## Ensemble Algorithm

Ensemble algorithms work by combinng multiple base algorithms or aggregate functions to get an optimal model.

* Applying an aggregate function to a collection of base models(e.g. bagging)
* Building a sequence of models that build one another to improve predictive performance(e.g. boosting)



### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor().fit(X_train, y_train)
```



### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor().fit(X_train, y_train)
```



# Improve Models 

## Hyperparameters

Hyperparameters are values that change the way the model is fitting during the loops. For example, the [learning rate](../Machine Learning/Preparetions.md). 



### Optimize Hyperparameters

Looping for a parameter dictionary contains a variety of learning rate and estimators. Using ```GridSearchCV``` from ```sklearn.model_selection```to find the best result for the given performance metric.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

# Using a Gradient Boosting algorithm
alg = GradientBoostingRegressor()

# Identify the hyperparameter values
params = {
    'learning_rate': [0.1, 0.5, 1.0],
    'n_estimators': [50, 100, 150]
}

# Find the best hyperparameter combination to get the optimal R2
score = make_scorer(r2_score)
grid_search = GridSearchCV(alg, params, scoring=score, cv=3, return_train_score=True)
grid_search.fix(X_train, y_train)
print('The best parameter combination:{}\n'.format(grid_search.best_params_))

# Get best model with the best parameter combination
model = grid_search.best_params_
print(model, "\n")
```



There are some parameters templates for models:

* **Decision Tree**

  ```
  
  ```

  

## Preprocessing Data

### Scaling features

#### Scaling Numeric features

#### Encoding Categorical features

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.prepocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np

# Define preprocessing for numeric features
numeric_features = [6, 7, 8, 9]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScale())
])

# Define preprocessing for categorical features
categorical_features = [0, 1, 2, 3, 4, 5]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
	transformer=[
        ('num', numeric_tranformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', GradientBoostingRegressor)])

# fit the pipelie to train a linear regression model
model = pipeline.fit(X_train, y_train)
```



### Example

Preprocessing, grid search cross-validation, regression model, evaluation, and graphic together.

```python
# Train the model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Define preprocessing for numeric columns (scale them)
numeric_features = [6,7,8,9]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Define preprocessing for categorical features (encode them)
categorical_features = [0,1,2,3,4,5]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', GradientBoostingRegressor())])

# Identify the hyperparameter values
params = {
    'regressor__learning_rate': [0.1, 0.5, 1.0],
    'regressor__n_estimators': [50, 100, 150]
}

# Find the best hyperparameter combination to get the optimal R2
score = make_scorer(r2_score)
grid_search = GridSearchCV(pipeline, params, scoring=score, cv=3, return_train_score=True)
grid_search.fit(X_train, y_train)
print('The best parameter combination:{}\n'.format(grid_search.best_params_))

# Get predictions with testing dataset
predictions = grid_search.predict(X_test)

# Display metrics
mse = mean_squared_error(y_test, predictions)
print('MSE: ', mse)
rmse = np.sqrt(mse)
print('RMSE: ', rmse)
r2 = r2_score(y_test, predictions)
print('R2: ', r2)

# Plot predicted and actual labels
fig = plt.figure(figsize(10 ,8))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')

# Overlay the regression line
z = np.ployfit(y_test, predictions, 1)
p = np.ploy1d(z)
plt.plot(y_test, p(y_test), color='green')

# show the graphic
plt.show()

# Save models to file
import joblib

filename = './data/trained_model.pkl'
joblib.dump(grid_search, filename)

# Load the model from file
model = joblib.load(finename)

# Predict new array
X_new = np.array([[16.2,289.3248,5,24.98203,121.54348],
                  [13.6,4082.015,0,24.94155,121.5038]])

predictions_new = model.predict(X_new)
```



> Linear Regression might not need to use GridSearchCV.  If have to, can use parameters I found on the internet:
>
> ```python
> ...
> # define search space
> space = dict()
> space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
> space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
> space['fit_intercept'] = [True, False]
> space['normalize'] = [True, False]
> ```
>
> 
