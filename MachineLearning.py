import pandas as pd

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

        # Data cleaning
        # Drop any columns that are not relevant
        # Drop any rows with null values
        df = df.drop(['id', 'lesion_1', 'lesion_2', 'lesion_3', 'cp_data'], axis=1)
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
        # Identify which columns in the dataset are categorical
        # One-hot encode the categorical columns in the training and validation sets
        # and verify the columns in X_train_encoded and X_val_encoded match
        categorical_columns = X_train.select_dtypes(include=['object']).columns
        X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns)
        X_val_encoded = pd.get_dummies(X_val, columns=categorical_columns)
        X_train_encoded, X_val_encoded = X_train_encoded.align(X_val_encoded, join='left', axis=1, fill_value=0)

        # Encoding the Target Variable
        # Initialize the encoder, fit and transform the training target variable
        # and transform the validation target variable
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)

        # Scaling the features in the dataset
        # Initialize the scaler, fit the scaler on the training data
        # and transform the validation data
        scaler = StandardScaler()
        numerical_columns = X_train_encoded.select_dtypes(include=['float64', 'int64']).columns
        X_train_encoded[numerical_columns] = scaler.fit_transform(X_train_encoded[numerical_columns])
        X_val_encoded[numerical_columns] = scaler.transform(X_val_encoded[numerical_columns])

        # Training the Logistic Regression Model
        # Initialize  and train the model
        # and predict on the validation set
        model = LogisticRegression(max_iter=200)
        model.fit(X_train_encoded, y_train_encoded)
        y_pred = model.predict(X_val_encoded)

        return df, model, X_train_encoded, X_val_encoded, y_train_encoded, y_val_encoded, y_pred, label_encoder