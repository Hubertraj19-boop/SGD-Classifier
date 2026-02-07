# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the Iris dataset and separate the feature variables and target class labels.
2.Split the dataset into training and testing sets using an appropriate ratio.
3.Train the Stochastic Gradient Descent (SGD) Classifier using the training data.
4.Predict the Iris species for test data and evaluate the model using accuracy and confusion matrix

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Hubert Raj.I
RegisterNumber:  25018951
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris()


df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

X = df.drop('target', axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train)


y_pred = sgd_clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Iris Species Prediction")
plt.show()
```
```
```


## Output:
<img width="988" height="452" alt="image" src="https://github.com/user-attachments/assets/5f964290-f9c6-4975-811d-ade5f92de187" />
<img width="828" height="614" alt="image" src="https://github.com/user-attachments/assets/deab4ed2-65f0-4f2d-a61d-444298f29a77" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
