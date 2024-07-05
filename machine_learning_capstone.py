import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv('horse_colic.csv')


# Drop any columns that are not relevant
df = df.drop(['id', 'lesion_1', 'lesion_2', 'lesion_3', 'cp_data'], axis=1)


# Drop any rows with null values
df = df.dropna()



# Split the train.csv into training and testing
# Test size to be 20% of the overall
X = df.drop('outcome', axis=1)
y = df['outcome']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Map 'euthanized' to 'died' in the 'outcome' column
y_train = y_train.replace({'euthanized': 'died'})
y_val = y_val.replace({'euthanized': 'died'})



# One-Hot Encoding
categorical_columns = X_train.select_dtypes(include=['object']).columns

# One-hot encode the categorical columns in the training set
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns)

# One-hot encode the categorical columns in the validation set
X_val_encoded = pd.get_dummies(X_val, columns=categorical_columns)

# Ensure the columns in X_train_encoded and X_val_encoded match
X_train_encoded, X_val_encoded = X_train_encoded.align(X_val_encoded, join='left', axis=1, fill_value=0)



# Encoding the Target Variable
# Initialize the encoder
label_encoder = LabelEncoder()

# Fit and transform the training target variable
y_train_encoded = label_encoder.fit_transform(y_train)

# Transform the validation target variable
y_val_encoded = label_encoder.transform(y_val)



# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data
numerical_columns = X_train_encoded.select_dtypes(include=['float64', 'int64']).columns
X_train_encoded[numerical_columns] = scaler.fit_transform(X_train_encoded[numerical_columns])

# Transform the validation data
X_val_encoded[numerical_columns] = scaler.transform(X_val_encoded[numerical_columns])



# Training the Logistic Regression Model
# Initialize the model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train_encoded, y_train_encoded)

# Predict on the validation set
y_pred = model.predict(X_val_encoded)

# Evaluate the model for accuracy
from sklearn.metrics import accuracy_score
print(f"Validation Accuracy: {accuracy_score(y_val_encoded, y_pred)}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_val_encoded, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_val_encoded, y_pred, target_names=label_encoder.classes_)
print("Classification Report:")
print(class_report)








'''

# CONFUSION MATRIX VISUALIZATION
# Confusion matrix values
conf_matrix = confusion_matrix(y_val_encoded, y_pred)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)

# Labels, title and ticks
label_font = {'size':'16'}  # Adjust to fit
ax.set_xlabel('Predicted labels', fontdict=label_font)
ax.set_ylabel('Actual labels', fontdict=label_font)
ax.set_title('Confusion Matrix', fontdict=label_font)
ax.xaxis.set_ticklabels(['Died', 'Lived'])
ax.yaxis.set_ticklabels(['Died', 'Lived'])

plt.show()



# FEATURE IMPORTANCE VISUALIZATION
importance = model.coef_[0]
plt.figure(figsize=(10, 8))
indices = np.argsort(importance)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X_train_encoded.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# PIE CHART VISUALIZATION - Distribution of pain level for horses that died
# Combine X_val_encoded and y_val_encoded into a single DataFrame
df_encoded = pd.concat([X_val_encoded, pd.Series(y_val_encoded, name='outcome')], axis=1)

# Filter the dataset to include only rows where the outcome is 'died'
lived_df = df_encoded[df_encoded['outcome'] == 1]

# List of pain level columns
pain_columns = ['pain_alert', 'pain_depressed', 'pain_extreme_pain', 'pain_mild_pain', 'pain_severe_pain']

# Sum the occurrences of each pain level in the filtered dataset
pain_counts = lived_df[pain_columns].sum()

# Mapping of original column names to display names
column_mapping = {
    'pain_alert': 'Alert',
    'pain_depressed': 'Depressed',
    'pain_extreme_pain': 'Extreme Pain',
    'pain_mild_pain': 'Mild Pain',
    'pain_severe_pain': 'Severe Pain'
}

# Apply the mapping to the index of pain_counts
pain_counts.index = pain_counts.index.map(column_mapping)

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(pain_counts, labels=pain_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
plt.title('Distribution of Pain Levels for Horses that Died')
plt.show()


'''