import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data
college_data = pd.read_csv('Final_Extended_Real_College_Dataset.csv')

# Encode categorical data
encoder = LabelEncoder()
college_data['Location'] = encoder.fit_transform(college_data['Location'])
college_data['Weather'] = encoder.fit_transform(college_data['Weather'])
college_data['Public/Private'] = encoder.fit_transform(college_data['Public/Private'])
college_data['Population Size'] = encoder.fit_transform(college_data['Population Size'])
college_data['Tuition Preference'] = encoder.fit_transform(college_data['Tuition Preference'])
college_data['Orientation'] = encoder.fit_transform(college_data['Orientation'])