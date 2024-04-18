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

# Hypothetical target: Let's say 1 if 'Public/Private' is Public and 'Tuition Preference' is low, else 0
college_data['Target'] = (college_data['Public/Private'] == 1) & (college_data['Tuition Preference'] == 0)


# Features and target
X = college_data[['Location', 'Weather', 'Public/Private', 'Population Size', 'Tuition Preference', 'Orientation']]
y = college_data['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy:', accuracy)
