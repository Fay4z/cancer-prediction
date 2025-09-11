import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score
)
import seaborn as sns
import pickle as pkl


df = pd.read_csv("Cancer_Data.csv")
print(df.head())

df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print(df.info())
print(df.head())

print(df.isnull().any())


X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

plt.figure(figsize=(6,4))
sns.boxplot(df)
plt.title("Boxplot - Radius Mean")
plt.show()

for column in df.columns:
  q1 = df[column].quantile(0.25)
  q3 = df[column].quantile(0.75)
  iqr = q3 - q1
  lb = q1 - 1.5*iqr
  ub = q3 + 1.5*iqr
  df = df[(df[column] >= lb) & (df[column] <= ub)]

plt.figure(figsize=(6,4))
sns.boxplot(df)
plt.title("Boxplot - Radius Mean")
plt.show()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Before scaling (first row):", X_train.iloc[0].values[:5])
print("After scaling (first row):", X_train_scaled[0][:5])

log_reg = LogisticRegression()

log_reg.fit(X_train_scaled, y_train)

print("Model trained successfully!")

y_pred = log_reg.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc*100))

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"]))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

auc = roc_auc_score(y_test, y_pred)
print("ROC AUC Score: {:.2f}".format(auc))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(5,4))
plt.scatter(
    df[df['diagnosis']==0]['radius_mean'], 
    df[df['diagnosis']==0]['texture_mean'], 
    color='blue', label='Benign', alpha=0.6
)

plt.scatter(
    df[df['diagnosis']==1]['radius_mean'], 
    df[df['diagnosis']==1]['texture_mean'], 
    color='green', label='Malignant', alpha=0.6
)

plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.title('Scatter Plot: Radius vs Texture')
plt.legend()
plt.show()

pkl.dump(log_reg, open("model.pkl", "wb"))