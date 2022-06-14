# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, plot_roc_curve
import pickle

# Reading Data
df = pd.read_csv("BSESENSEX_2012-2022.csv")
# Modifying the Dataset: I will be adding the Daily Price Change from which I will be calculating the Label (Up/Down)
# Calculating the Daily Percentage change in the Close price of the Index
# Label 1 would mean that the stock went up that day
# Label of 0 would mean that the stock either remained the same or went down.
df['Daily_Percentage_Change'] = df['Close'].pct_change()*100
df['Label'] = df['Daily_Percentage_Change'].apply(lambda x: 0 if x < 1 else 1)
print(df.head(100))
# Dropping all rows with a NA value
df = df.dropna()
y = df["Label"]
X = df.drop(columns=["Date", "Label", "Daily_Percentage_Change"])

print(X.head())
# Running Classification Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train.values.ravel())
RandomForestClassifier()
pickle.dump(random_forest, open("model.pkl", "wb"))

y_pred = random_forest.predict(X_test)
print(random_forest.score(X_test, y_test))
print(accuracy_score(y_test, y_pred))

# I am running the classification model using Open, High, Low, Close

