import pandas as pd
df=pd.read_csv('C:/Users/dcsuser/Desktop/shanika/student_data.csv')
print("Display few row: ")
print(df.head())

print('description:', df.describe())
print('Number of rows and column :', df.shape)

#Remove fully repeated rows
df.drop_duplicates(inplace=True)
print("Duplicates removed.")
print(df)

#If the same student appears multiple times,keep only the first one
df=df.drop_duplicates(subset=['Student_ID'],keep='first')
print("Duplicates removed.")
print(df)


#fix inconsistent formating
df['Gender'] = df['Gender'].str.lower().str.strip()
gender_map = {'male': 'Male', 'female': 'Female', 'f': 'Female', 'm': 'Male'}
df['Gender']= df['Gender'].map(gender_map)
print("Formating standerdized.")
print(df)

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
print("Fix incoreect data type.")
print(df)

import numpy as np
df['Department'] = df['Department'].replace('0', np.nan)
print("Data type corrected.")
print(df)


#missing Values
#df['Age'] = df['Age'].fillna(df['Age'].median())
df['Age'] = df['Age'].fillna(30)
df['Attendance']= df['Attendance'].fillna(df['Attendance'].mean())
df['Department']= df['Department'].fillna(df['Department'].mode()[0])

# fillna - fill missing valus.  It replace Nan with another value..

print("Missing values handled")
print(df)

# handling noice data
df['Department'] = df['Department'].replace('COmputer Since', 'Computer Science')
print("Noise handled")
print(df)

#Define the salary limit

upper_limit = 60000
lower_limit = 30000
#cap salary values

df['Salary'] = np.where(
    df['Salary'] > upper_limit,
    upper_limit,
    np.where(
        df['Salary'] < lower_limit,
        lower_limit,
        df['Salary']
    )
)
print(df)


upper_Age_limit = 100
lower_Age_limit = 20
#

df['Age'] = np.where(
    df['Age'] > upper_Age_limit,
    upper_Age_limit,
    np.where(
        df['Age'] < lower_Age_limit,
        lower_Age_limit,
        df['Age']
    )
)
print(df)

#Data transformation

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Min-Max Normalization
scaler_minmax = MinMaxScaler()
df['Attendance_Normalized'] = scaler_minmax.fit_transform(df[['Attendance']])

print(df)

# Standardization
scaler_std = StandardScaler()
df['Salary_Standardized'] = scaler_std.fit_transform(df[['Salary']])

print(df)
