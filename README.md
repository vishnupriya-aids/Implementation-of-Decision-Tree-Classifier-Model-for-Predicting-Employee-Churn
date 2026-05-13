# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Loading: Import the dataset and convert text labels (like Salary) into numbers.

2. Data Splitting: Divide the data into Features (X) and Target (y), then split into Training and Testing sets.

3. Model Training: Build the DecisionTreeClassifier using the Entropy criterion and train it with the training data.

4. Evaluation: Predict outcomes for the test set, calculate Accuracy, and visualize the results via a Confusion Matrix.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:VISHNUPRIYA E 
RegisterNumber: 212225230308
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file_path =("C:/Users/acer/Downloads/Employee.csv")
df = pd.read_csv(file_path)


print("Dataset Columns:", df.columns.tolist())


if 'salary' in df.columns:
    salary_mapping = {'low': 0, 'medium': 1, 'high': 2}
    df['salary'] = df['salary'].map(salary_mapping)


df = pd.get_dummies(df, drop_first=True)


target_col = 'left' 
X = df.drop(target_col, axis=1)
y = df[target_col]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




plt.figure(figsize=(20, 10))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=['Stayed', 'Left'], 
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title("Employee Churn Decision Tree Logic")
plt.show()
*/
```

## Output:
<img width="1002" height="543" alt="Screenshot 2026-05-13 091417" src="https://github.com/user-attachments/assets/2169c5cc-6b18-4748-8166-0f48f7329252" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
