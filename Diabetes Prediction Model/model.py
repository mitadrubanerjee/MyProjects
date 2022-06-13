import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, plot_roc_curve
import pickle

# Loading Data
df = pd.read_csv("diabetes.csv")
# print(df.head())
y = df["Outcome"]
# print(y)
# Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
# Factors in this model, Glucose, BloodPressure, Insulin, BMI, Age
X = df.drop(columns=["Pregnancies", "Outcome", "SkinThickness", "DiabetesPedigreeFunction"])
# print(X)
# Data Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train.values.ravel())
RandomForestClassifier()
pickle.dump(random_forest, open("model.pkl", "wb"))

y_pred = random_forest.predict(X_test)
print(random_forest.score(X_test, y_test))
print(accuracy_score(y_test, y_pred))



