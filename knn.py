import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


irisData = load_iris()

# Features
X = irisData.data

# Labels
y = irisData.target

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# Value of k
k = 3

# Predict all test samples
predictions = []

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

def predict(X_train, y_train, test_point, k):
    distances = []

    # Calculate distance to every training sample
    for i in range(len(X_train)):
        dist = euclidean_distance(test_point, X_train[i])
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])

    # Get k nearest neighbors
    neighbors = distances[:k]

    votes = {}
    for _, label in neighbors:
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
	
    prediction = max(votes, key=votes.get)
    return prediction

for test_point in X_test:
    predictions.append(predict(X_train, y_train, test_point, k))




# Convert predictions to NumPy array for cleaner printing
predictions = np.array(predictions)

print("Predicted classes:")
print(predictions)

print("\nActual classes:")
print(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=irisData.target_names
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
