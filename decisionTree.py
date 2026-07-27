import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("PlayTennis.csv")
print(df)

encoders ={}
for col in df.columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])
    
 
print(df)

x= df[["Outlook", "Temperature", "Humidity", "Wind"]]
y = df["Play_Tennis"]
model = DecisionTreeClassifier(criterion="entropy")
model.fit(x, y)

plt.figure(figsize=(7, 14))

plot_tree(
    model,
    feature_names=["Outlook", "Temperature", "Humidity", "Wind"],
    class_names=["No", "Yes"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree- should I play Tennis ? ")
plt.show()

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