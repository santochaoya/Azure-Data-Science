# Training and Evaluating A Classification Model

Classification model is a form of supervised learning in which you predicted a appropriate label based on calculating the probability of the observed cases belong to each of a number of a possible classes. 



## Evaluate Classification Models

### Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

# or
print('Overall Precision: ', precision_score(y_test, predictions))
print('Overall Recall: ', recall_score(y_test, predictions))
```



![Classification1](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\Classification1.PNG)

* **Precision**

  of all the cases that the model predicted to be positive, how many actually are positive?
  $$
  Precision = \frac{TP}{(TP + FP)}
  $$

* **Recall**

  of all the cases that are actually positive, how many did the model identify?
  $$
  Recall = \frac{TP}{(TP + FN)}
  $$

* **Accurancy**

  out of all the predictions, how many are correct?
  $$
  Accurancy = \frac{(TP + TN)}{(TP + TN + FP + FN)}
  $$

* **F1-score**

  The balance between precision and recall
  $$
  F1 & = 2 \times \frac{Precision \times Recall} {(Precision + Recall)} \\
  & = \frac {TP}{TP + \frac{1}{2}(FP + FN)}
  $$

* **Support**

  How many instances of this class are there in the test dataset?



### Precision & Recall

* *True Positive*: The predicted label and actual label are both positive

* *False Positive*: The predicted label is positive, the actual label is negative.

* *False Negative*: The predicted label is negative, the actual label is positive.

* *True Negative*: The predicted label and actual label are both negative.

  ```python
  from sklean.metrics import confusion_matrix
  
  cm = confusion_matrix(y_test, predictions)
  ```



### Probabilities of each case

```python
y_scores = model.predict_proba(X_test)
```



### ROC & AUC

* **ROC** - Received Operator Characteristic

  The relationship between True Positive Rate(**Recall**) and False Positive Rate for a range of possible thresholds.

  The more likely the graphic to a right triangle.

  ```python
  from sklearn.metrics import roc_curve
  from sklearn.metrics import confusion_matrix
  import matplotlib.pyplot as plt
  
  # Calculate ROC curve with label in testing dataset and probabilities of positive cases
  fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
  
  # Plot ROC curve
  fig = plt.figure(figsize=(8, 8))
  
  % Plot the diagonal 50% line
  plt.plot([0, 1], [0, 1], 'k--')
  
  # Plot the FPR and TPR achieved by our model
  plt.plot(fpr, tpr)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  
  # Show the graphic
  plt.show()
  ```



![Classification2](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\Classification2.PNG)

* **AUC** - Area Under the Curve

  A value between 0 and 1 that quantifies the overall performance of the model. The closer to 1 the value is, the better the model performed.

  ```python
  from sklearn.metric import roc_auc_score
  
  auc = roc_auc_score(y_test, y_scores[:, 1])
  ```



## Multiclass Classification Models

The multiclass classification means label with more than two classes. It can be treated as a combination of multiple binary classifiers.



### Create Multiclass Classification Models

* **One vs Rest (OVR)**
  * Class 1 or not
  * Class 2 or not
  * Class 3 or not

* **One vs One (OVO)**
  * Class 1 or Class 2
  * Class 1 or Class 3
  * Class 2 or Class 3

## Train and Evaluate

Most of the steps are the same as the Binary Classification Models. When evaluate multiclass Classification models, we can use a heat map to identify the confusion matrix.

```python
import numpy as np
import mapplotlib.pyplot as plt

# Train the multiclass logistic regression model
multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000).fit(X_train, y_train)

# Predict on testing dataset
predictions = multi_model.predict(X_test)

# Get the confusion matrix
mcm = confusion_matrix(y_test, predictions)

# Plot the heat map graphic
plt.imshow(mcm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()

# Make a tick marks of density of the total number in all classes
tick_marks = np.arrange(len(y_classes))

# Costumized the graphic
plt.xticks(tick_marks, y_classes, rotation=90)
plt.yticks(tick_marks, y_classes)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")

# Show the graphic
plt.show()
```



