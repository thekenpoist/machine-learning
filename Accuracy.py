




@app.route('/accuracy')
def accuracy():

    accuracy = str(round(accuracy_score(y_val_encoded, y_pred) * 100, 2))
    class_report = classification_report(y_val_encoded, y_pred, target_names=label_encoder.classes_)
    class_report_lst = class_report.split()
    class_report_lst = [class_report.capitalize() if not class_report.isdigit() else class_report for class_report in class_report_lst]

    return render_template('accuracy.html', accuracy=accuracy, class_report=class_report_lst)