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
import sklearn.tree import DecisionTreeRegressor
import sklearn.tree import export_text

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

## Preprocessing Data

### Feature Engine

### Scaling features

