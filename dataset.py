import pandas as pd
import numpy as np

# ==========================================
# Step 1: Load Dataset
# ==========================================

df = pd.read_csv("student_data.csv")

print("========== Original Dataset ==========")
print(df)

print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ==========================================
# Step 2: Remove Duplicate Records
# ==========================================

# Remove fully duplicated rows
df.drop_duplicates(inplace=True)

# Remove duplicate Student IDs (keep first)
df = df.drop_duplicates(subset=['Student_ID'], keep='first')

print("\n========== After Removing Duplicates ==========")
print(df)

# ==========================================
# Step 3: Standardize Gender Formatting
# ==========================================

df['Gender'] = df['Gender'].str.lower().str.strip()

gender_map = {
    'male': 'Male',
    'female': 'Female',
    'm': 'Male',
    'f': 'Female'
}

df['Gender'] = df['Gender'].map(gender_map)

print("\n========== After Standardizing Gender ==========")
print(df)

# ==========================================
# Step 4: Correct Data Types
# ==========================================

df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

print("\n========== After Converting Age ==========")
print(df)

# ==========================================
# Step 5: Replace Invalid Values
# ==========================================

df['Department'] = df['Department'].replace('0', np.nan)

print("\n========== After Replacing Invalid Department ==========")
print(df)

# ==========================================
# Step 6: Handle Missing Values
# ==========================================

# Fill missing Age
df['Age'] = df['Age'].fillna(30)

# Fill missing Attendance
df['Attendance'] = df['Attendance'].fillna(df['Attendance'].mean())

# Fill missing Department
df['Department'] = df['Department'].fillna(df['Department'].mode()[0])

print("\n========== After Handling Missing Values ==========")
print(df)

# ==========================================
# Step 7: Handle Noisy Data
# ==========================================

df['Department'] = df['Department'].replace(
    'Computer Since',
    'Computer Science'
)

print("\n========== After Correcting Department ==========")
print(df)

# ==========================================
# Step 8: Handle Salary Outliers
# ==========================================

upper_salary = 60000
lower_salary = 30000

df['Salary'] = np.where(
    df['Salary'] > upper_salary,
    upper_salary,
    np.where(
        df['Salary'] < lower_salary,
        lower_salary,
        df['Salary']
    )
)

print("\n========== After Salary Outlier Handling ==========")
print(df)

# ==========================================
# Step 9: Handle Age Outliers
# ==========================================

upper_age = 100
lower_age = 20

df['Age'] = np.where(
    df['Age'] > upper_age,
    upper_age,
    np.where(
        df['Age'] < lower_age,
        lower_age,
        df['Age']
    )
)

print("\n========== After Age Outlier Handling ==========")
print(df)

# ==========================================
# Step 10: Data Transformation
# ==========================================

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Normalize Attendance
minmax = MinMaxScaler()

df['Attendance_Normalized'] = minmax.fit_transform(
    df[['Attendance']]
)

# Standardize Salary
standard = StandardScaler()

df['Salary_Standardized'] = standard.fit_transform(
    df[['Salary']]
)

print("\n========== After Data Transformation ==========")
print(df)

# ==========================================
# Step 11: Label Encoding
# ==========================================

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['Department'] = encoder.fit_transform(df['Department'])

print("\n========== After Label Encoding ==========")
print(df)

# ==========================================
# Step 12: One-Hot Encoding
# ==========================================

df = pd.get_dummies(df, columns=['Gender'], dtype=int)

print("\n========== After One-Hot Encoding ==========")
print(df)

# ==========================================
# Step 13: Save Cleaned Dataset
# ==========================================

df.to_csv("student_data_cleaned.csv", index=False)

print("\n===================================")
print("Cleaned dataset saved successfully!")
print("File Name: student_data_cleaned.csv")
print("===================================")