from sklearn.metrics import accuracy_score, classification_report
from flask import render_template


# Accuracy report function
def accuracy_report(y_val_encoded, y_pred, label_encoder):

    # Calculate the accuracy score
    # and generate the classification report
    accuracy = str(round(accuracy_score(y_val_encoded, y_pred) * 100, 2))
    class_report = classification_report(y_val_encoded, y_pred, target_names=label_encoder.classes_)
    class_report_lst = class_report.split()
    class_report_lst = [report.capitalize() if not report.isdigit() else report for report in class_report_lst]

    return render_template('accuracy.html', accuracy=accuracy, class_report=class_report_lst)