import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine

wineData = load_wine()

# Features
X = wineData.data

# Labels
y = wineData.target

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model with k=3
#k=1 -> Accuracy: 77.78% , k= 3 -> Accuracy: 80.56% , k= 5 ->Accuracy: 80.56%, 
#, k= 7 -> Accuracy: 69.44% , k =9 -> Accuracy: 72.22%
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# Predict on the test data
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.2%}")