from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from flask import render_template



# Accuracy report function
def accuracy_report(y_val_encoded, y_pred, label_encoder, model, X_val_encoded):

    # Calculate the accuracy score, generate the classification report
    # and calculate the AUC-ROC curve
    accuracy = str(round(accuracy_score(y_val_encoded, y_pred) * 100, 2))
    class_report = classification_report(y_val_encoded, y_pred, target_names=label_encoder.classes_)
    class_report_lst = class_report.split()
    class_report_lst = [report.capitalize() if not report.isdigit() else report for report in class_report_lst]
    y_pred_proba = model.predict_proba(X_val_encoded)[:, 1]
    roc_auc = roc_auc_score(y_val_encoded, y_pred_proba)

    return render_template('accuracy.html', accuracy=accuracy, class_report=class_report_lst, roc_auc=round(roc_auc, 3))