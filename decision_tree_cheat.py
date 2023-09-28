import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_text

dataTrain = pd.read_csv('heart_train.tsv', sep='\t')
dataVal = pd.read_csv('heart_val.tsv', sep='\t')

# Load the Iris dataset and split into training and testing sets
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = dataTrain.iloc[:, :-1], dataVal.iloc[:, :-1], dataTrain.iloc[:, -1], dataVal.iloc[:, -1]

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Evaluate the classifier on the test data
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy before pruning:", accuracy)

# Prune the decision tree by setting ccp_alpha
# Larger values of ccp_alpha lead to more pruning
pruned_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=0.01)
pruned_clf.fit(X_train, y_train)

# Evaluate the pruned classifier on the test data
y_pred_pruned = pruned_clf.predict(X_test)
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print("Accuracy after pruning:", accuracy_pruned)

feature_names = ['sex', 'chest_pain','high_blood_sugar','abnormal_ecg','angina','flat_ST','fluoroscopy','thalassemia']
# Display the pruned decision tree as text
tree_text = export_text(pruned_clf, feature_names=feature_names)
print("Pruned Decision Tree (text representation):\n", tree_text)