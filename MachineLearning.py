import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



class MachineLearning:

    def __init__(self):
        pass

    def machine_learning():

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

        return df, model, X_train_encoded, X_val_encoded, y_train_encoded, y_val_encoded, y_pred, label_encoder



    
        #Classification Report
        #class_report = classification_report(y_val_encoded, y_pred, target_names=label_encoder.classes_)
        #print("Classification Report:")
        #print(class_report)
        