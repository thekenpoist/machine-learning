from sklearn import model_selection
from MachineLearning import *

import os
from flask import Flask, render_template, send_file
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns



app = Flask(__name__)

model, X_train_encoded, X_val_encoded, y_train_encoded, y_val_encoded, y_pred, label_encoder = MachineLearning.machine_learning()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/accuracy')
def accuracy():

    return render_template('accuracy.html', accuracy = str(round(accuracy_score(y_val_encoded, y_pred) * 100, 2)))

@app.route('/class_report')
def class_report():

    return render_template('class_report.html', class_report = classification_report(y_val_encoded, y_pred, target_names=label_encoder.classes_))




@app.route('/confusion_matrix')
def confusion_matrix_visual():

        conf_matrix = confusion_matrix(y_val_encoded, y_pred)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)

        # Labels, title and ticks
        label_font = {'size': '16'}  # Adjust to fit
        ax.set_xlabel('Predicted labels', fontdict=label_font)
        ax.set_ylabel('Actual labels', fontdict=label_font)
        ax.set_title('Confusion Matrix', fontdict=label_font)
        ax.xaxis.set_ticklabels(['Died', 'Lived'])
        ax.yaxis.set_ticklabels(['Died', 'Lived'])

        # Save plot to file as image
        image_path = os.path.join('static', 'images', 'confusion_matrix.png')
        plt.savefig(image_path)
        plt.close(fig)

        return render_template('confusion_matrix_visual.html')


@app.route('/feature_importance')
def feature_importance_visual():
    # FEATURE IMPORTANCE VISUALIZATION
    importance = model.coef_[0]

    fig, ax = plt.subplots(figsize=(14, 10))
    indices = np.argsort(importance)
    ax.set_title('Feature Importances')
    ax.barh(range(len(indices)), importance[indices], color='b', align='center')
    ax.set_yticks(range(len(indices)), [X_train_encoded.columns[i] for i in indices])
    ax.set_xlabel('Relative Importance')

    # Save plot to file as image
    image_path = os.path.join('static', 'images', 'feature_importance.png')
    plt.savefig(image_path)
    plt.close(fig)

    return render_template('feature_importance_visual.html')


@app.route('/pie_chart')
def pie_chart_visual():
    # PIE CHART VISUALIZATION - Distribution of pain level for horses that died
    # Combine X_val_encoded and y_val_encoded into a single DataFrame
    df_encoded = pd.concat([X_val_encoded, pd.Series(y_val_encoded, name='outcome')], axis=1)

    # Filter the dataset to include only rows where the outcome is 'died'
    died_df = df_encoded[df_encoded['outcome'] == 1]

    # List of pain level columns
    pain_columns = ['pain_alert', 'pain_depressed', 'pain_extreme_pain', 'pain_mild_pain', 'pain_severe_pain']

    # Sum the occurrences of each pain level in the filtered dataset
    pain_counts = died_df[pain_columns].sum()

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
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(pain_counts, labels=pain_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    ax.set_title('Distribution of Pain Levels for Horses that Died')
    
    # Save plot to file as image
    image_path = os.path.join('static', 'images', 'pain_distribution_pie_chart.png')
    plt.savefig(image_path)
    plt.close(fig)

    return render_template('pie_chart_visual.html')



if __name__=='__main__':
    app.run()
    
    



