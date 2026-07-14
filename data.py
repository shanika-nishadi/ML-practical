# import pandas as pd
# df=pd.read_csv('C:/Users/dcsuser/Desktop/shanika/student_data (1).csv')
# print("Display few row: ")
# print(df.head())

# from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder
# #label encoding

# le = LabelEncoder()
# df['Department'] = le.fit_transform(df['Department'])
# print(df)

# #method B - One-hot Encoding for Nominal data (Gender and city)

# df = pd.get_dummies(df, columns=['Gender'], dtype=int)
# print("\n--- 6. After Categerical Encoding ---")
# print(df.info())

# df.to_csv('C:/Users/dcsuser/Desktop/shanika/student_data_cleaned (1).csv')
# print("\n cleaned dataset saved\n")

import pandas as pd  # working with tables and datasetts
import matplotlib.pyplot as plt  # creating charts and graphs
import seaborn as sb # colorful and attreactive garphs

df = pd.read_csv('C:/Users/dcsuser/Desktop/shanika/student_data_cleaned.csv')
print("cleaned Dataset")
print(df)

plt.figure(figsize=(6, 5))
#count department value

gender_count = df['Gender'].value_counts()
# create bar chart
plt.bar(gender_count.index, gender_count.values, color=['skyblue', 'pink'])
plt.title('Bar Chart - Count of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# department_count = df['department'].value_counts()

# plt.pie(department_count.values, )
# plt.title('Pie Chart - count of department')
# plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(df['Age'], df['GPA'], color='purple')
plt.title('Scatter Plot - Age vs GPA')
plt.show()

#Multivariate Analysis
# correlation Heatmap

plt.figure(figsize=(8, 6))
corr = df[['Age', 'GPA', 'Attendance', 'Salary']].corr()
sb.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Heatmap (Correlation)')
plt.show()