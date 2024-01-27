# Creating a prediction model for the survival rate on the Titanic.
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Getting path to the files
train_path = os.path.join(os.path.dirname(__file__), 'train.csv')
test_path = os.path.join(os.path.dirname(__file__), 'test.csv')

# Get the directory where the current script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path for the output file in the same directory
output_file_path = os.path.join(script_directory, 'submission.csv')

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Data Cleaning and Preprocessing
age_imputer = SimpleImputer(strategy='median')
train_df['Age'] = age_imputer.fit_transform(train_df[['Age']])
test_df['Age'] = age_imputer.transform(test_df[['Age']])
embarked_mode = train_df['Embarked'].mode()[0]
train_df['Embarked'].fillna(embarked_mode, inplace=True)
test_df['Embarked'].fillna(embarked_mode, inplace=True)
train_df['Cabin_Unknown'] = train_df['Cabin'].isnull().astype(int)
test_df['Cabin_Unknown'] = test_df['Cabin'].isnull().astype(int)
train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
train_df['Embarked'] = train_df['Embarked'].map(embarked_mapping)
test_df['Embarked'] = test_df['Embarked'].map(embarked_mapping)

# Feature Engineering
train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Col": 7, "Major": 7, "Mlle": 8, 
                 "Countess": 9, "Ms": 2, "Lady": 9, "Jonkheer": 10, "Don": 10, "Dona": 10, "Mme": 8, "Capt": 7, "Sir": 9}
train_df['Title'] = train_df['Title'].map(title_mapping)
test_df['Title'] = test_df['Title'].map(title_mapping)
train_df['Title'].fillna(0, inplace=True)
test_df['Title'].fillna(0, inplace=True)
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
train_df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test_passenger_ids = test_df['PassengerId']
test_df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Handling missing value in Fare in test dataset
fare_median = train_df['Fare'].median()
test_df['Fare'].fillna(fare_median, inplace=True)

# Model Training
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions on the test dataset
test_predictions = rf_model.predict(test_df)

# Preparing the submission dataframe
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': test_predictions
})

# Saving the submission to a CSV file
submission_df.to_csv(output_file_path, index=False)
