from flask import Flask, render_template

from MachineLearning import *
from ConfusionMatrix import confusion_matrix_visual
from AccuracyReport import accuracy_report
from FeatureImportances import feature_importances_visual
from PieChart import pie_chart_visual
from BarGraph import bar_graph_visual
from SurvivalCalculator import survival_calculator
from Prediction import prediction


app = Flask(__name__)


# Call the machine_learning() method from the MachineLearning class to execute the machine learning algorithm.
# This method returns these needed variables:
# df: The original DataFrame used for the machine learning process.
# model: The trained machine learning model.
# X_train_encoded: The encoded training feature set.
# X_val_encoded: The encoded validation feature set.
# y_train_encoded: The encoded training target variable.
# y_val_encoded: The encoded validation target variable.
# y_pred: The predicted target variable for the validation set.
# label_encoder: The label encoder used for encoding the target variable.

df, model, X_train_encoded, X_val_encoded, y_train_encoded, y_val_encoded, y_pred, label_encoder = MachineLearning.machine_learning()


# Render index
@app.route('/')
def index():
    return render_template('index.html')


# Accuracy report route
@app.route('/accuracy_report')
def accuracy_report_route():

    return accuracy_report(y_val_encoded, y_pred, label_encoder)


# Confusion matrix route
@app.route('/confusion_matrix')
def confusion_matrix_route():

    return confusion_matrix_visual(y_val_encoded, y_pred)


# Feature importances route
@app.route('/feature_importances')
def feature_importances_route():

    return feature_importances_visual(model, X_train_encoded)


# Pie chart route
@app.route('/pie_chart')
def pie_chart_route():

    return pie_chart_visual(X_val_encoded, y_val_encoded)


# Bar graph route
@app.route('/bar_graph')
def bar_graph_route():

    return bar_graph_visual(df)


# Survival calculator route
@app.route('/survival_calculator')
def survival_calculator_route():

    return survival_calculator(df)


# Prediction route
@app.route('/prediction', methods=['POST'])
def prediction_route():

    return prediction(X_train_encoded, model)


# Run app
if __name__=='__main__':
    app.run()
    
    



