import math
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

irisData = load_iris()

#Features
X = irisData.data

#labels
Y = irisData.target
#print(X)

#split data into training(80%) and Testing(20%).

X_train , X_test, Y_train, Y_test = train_test_split(
	X, Y,
	test_size= 0.2,
	random_state=42
)
k=3 # must be odd Number
prediction = []

def euclidean_distance(point1,point2):
	distance=0
	for i in range(len(point1)):
		distance+=(point1[i] - point2[i])**2
	return math.sqrt(distance)


def predict(X_train,Y_train,test_point,k):
	distances =[]
	for i in range(len(X_train)):
		dist= euclidean_distance(test_point,X_train[i])
		distances.append((dist,Y_train[i]))
	distances.sort(key=lambda x:x[0])
	neighbours = distances[:k]
	votes = {}
	for _, label in neighbours :
		if label in votes:
			votes[label] +=1
		else:
			votes[label] = 1
			
	prediction = max(votes, key=votes.get)
	return prediction
		
	

for test_point in X_test:
	prediction.append(predict(X_train,Y_train,test_point,k))
	
accuracy = accuracy_score(Y_test,prediction)
print(accuracy)

cm = confusion_matrix(Y_test,prediction)
print(cm)