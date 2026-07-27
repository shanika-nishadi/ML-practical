import math
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree




irisData = load_iris()

x = irisData.data

# Labels
y = irisData.target

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)






model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

plt.figure(figsize=(12, 8))

plot_tree(
    model,
    feature_names=["Outlook", "Temperature", "Humidity", "Wind"],
    class_names=["No", "Yes"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree- should I play Tennis ? ")
plt.show()



encoders ={}
for col in df.columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])
    
 

x = 





new_day = pd.DataFrame({
"Outlook" : ["Sunny"],
"Temperature" : ["Cool"],
"Humidity" : ["Normal"],
"Wind" : ["Weak"],

})
for col in new_day.columns:
    new_day[col] = encoders[col].transform(new_day[col])
    
prediction = model.predict(new_day)

if prediction[0] == 1:
    print("Prediction: play Tennis")
else:
    print("Prediction: Do not play Tennis")