import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import os
 
# loading and cleaning
df = pd.read_csv('dataset/Admission_Predict(1).csv')
df.drop(columns=['Serial No.'], inplace=True)        # Removes irrelvant ID column
df.columns = df.columns.str.strip()                  # strips off extra whitespaces from column names

# feature enginnering and target creation
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
df['Admitted'] = (df['Chance of Admit'] >= 0.75).astype(int)  
df.drop(columns=['Chance of Admit'], inplace=True)
X = df.drop('Admitted', axis=1)
y = df['Admitted']
''' 
Converts the continuous Chance of Admit score into a binary label; 1 = Admitted (≥ 0.75), 0 = Not Admitted. 
Separates features (X) from the target (y)
'''

feature_names = X.columns.tolist()
print(f"\nFeatures : {feature_names}")
print(f"Target   : Admitted (0 = No, 1 = Yes)")
print(f"Class distribution:\n{y.value_counts()}")

# TTS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Splits data into 80% training and 20% testing. random_state=42 ensures reproducibility.

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing  Samples : {X_test.shape[0]}")
 
 # Model Training 
model = DecisionTreeClassifier(
    criterion='entropy',   # This is what makes it ID3
    max_depth=5,
    random_state=42)
# ID3 uses entropy as the split criterion, measuring disorder in each node.
# The tree picks the feature that reduces entropy (uncertainty) the most at every split called Information Gain.

model.fit(X_train, y_train)
print("Model trained successfully")
 
# Print the tree rules in text format
print("\n Decision Tree Rules (ID3):")
tree_rules = export_text(model, feature_names=feature_names)
print(tree_rules)
 
# Makes predictions on unseen test data and measures performance using accuracy, precision/recall/F1, and the confusion matrix.
y_pred = model.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy : {accuracy * 100:.2f}%")
 
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Not Admitted', 'Admitted']))
 
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# visualizations
os.makedirs("model", exist_ok=True)
 
# Ploting Decision Tree 
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=feature_names,
    class_names=['Not Admitted', 'Admitted'],
    filled=True,
    rounded=True,
    fontsize=10)
plt.title("ID3 Decision Tree — College Admission Predictor", fontsize=16)
plt.tight_layout()
plt.savefig("model/decision_tree.png", dpi=150)
plt.close()
print("Decision Tree saved → model/decision_tree.png")
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
    print(" Model saved → model/model.pkl")
    print("\n Training Complete")