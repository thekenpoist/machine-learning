from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os

from MachineLearning import *


app = Flask(__name__)


df, model, X_train_encoded, X_val_encoded, y_train_encoded, y_val_encoded, y_pred, label_encoder = MachineLearning.machine_learning()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/accuracy')
def accuracy():

    return render_template('accuracy.html', accuracy = str(round(accuracy_score(y_val_encoded, y_pred) * 100, 2)))

@app.route('/class_report')
def class_report():

    return render_template('class_report.html', class_report = classification_report(y_val_encoded, y_pred, target_names=label_encoder.classes_))




@app.route('/survival_calculator')
def survival_calculator():
    unique_hospital_number = df['hospital_number'].unique()
    unique_temp_of_extremities = df['temp_of_extremities'].unique()
    unique_peripheral_pulse = df['peripheral_pulse'].unique()
    unique_mucous_membrane = df['mucous_membrane'].unique()
    unique_capillary_refill_time = df['capillary_refill_time'].unique()
    unique_pain = df['pain'].unique()
    unique_peristalsis = df['peristalsis'].unique()
    unique_abdominal_distention = df['abdominal_distention'].unique()
    unique_nasogastric_tube = df['nasogastric_tube'].unique()
    unique_nasogastric_reflux = df['nasogastric_reflux'].unique()
    unique_rectal_exam_feces = df['rectal_exam_feces'].unique()
    unique_abdomen = df['abdomen'].unique()
    unique_abdomo_appearance = df['abdomo_appearance'].unique()



    return render_template('survival_calculator.html', dataframe = df.columns, unique_hospital_number=unique_hospital_number, 
                           unique_temp_of_extremities=unique_temp_of_extremities, unique_peripheral_pulse=unique_peripheral_pulse,
                           unique_mucous_membrane=unique_mucous_membrane, unique_capillary_refill_time=unique_capillary_refill_time,
                           unique_pain=unique_pain, unique_peristalsis=unique_peristalsis, unique_abdominal_distention=unique_abdominal_distention,
                           unique_nasogastric_tube=unique_nasogastric_tube, unique_nasogastric_reflux=unique_nasogastric_reflux,
                           unique_rectal_exam_feces=unique_rectal_exam_feces, unique_abdomen=unique_abdomen, 
                           unique_abdomo_appearance=unique_abdomo_appearance)


@app.route('/predict', methods=['POST'])
def predict():

    form_data = {
        'surgery': [request.form['surgery']],
        'age': [request.form['age']],
        'hospital_number': [request.form['hospital_number']],
        'rectal_temp': [request.form['rectal_temp']],
        'pulse': [request.form['pulse']],
        'respiratory_rate': [request.form['respiratory_rate']],
        'temp_of_extremities': [request.form['temp_of_extremities']],
        'peripheral_pulse': [request.form['peripheral_pulse']],
        'mucous_membrane': [request.form['mucous_membrane']],
        'capillary_refill_time': [request.form['capillary_refill_time']],
        'pain': [request.form['pain']],
        'peristalsis': [request.form['peristalsis']],
        'abdominal_distention': [request.form['abdominal_distention']],
        'nasogastric_tube': [request.form['nasogastric_tube']],
        'nasogastric_reflux': [request.form['nasogastric_reflux']],
        'nasogastric_reflux_ph': [request.form['nasogastric_reflux_ph']],
        'rectal_exam_feces': [request.form['rectal_exam_feces']],
        'abdomen': [request.form['abdomen']],
        'packed_cell_volume': [request.form['packed_cell_volume']],
        'total_protein': [request.form['total_protein']],
        'abdomo_appearance': [request.form['abdomo_appearance']],
        'abdomo_protein': [request.form['abdomo_protein']],
        'surgical_lesion': [request.form['surgical_lesion']]
    }

    # Create a DataFrame from the form data
    df_form = pd.DataFrame(form_data)

    # One-hot encode the form data to match the training data
    df_form_encoded = pd.get_dummies(df_form)

    # Add missing columns with zeros to match X_train_encoded
    for col in X_train_encoded.columns:
        if col not in df_form_encoded.columns:
            df_form_encoded[col] = 0

    # Reorder columns to match X_train_encoded
    df_form_encoded = df_form_encoded[X_train_encoded.columns]

    # Predict the probability of survival
    probability = model.predict_proba(df_form_encoded)[:, 1]

    return render_template('prediction.html', probability=str(round(probability[0] * 100, 2)))


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


@app.route('/bar_chart')
def hospital_outcome_bar_chart():
    # BAR CHART VISUALIZATION - Top 20 hospitals with the highest death rate.
    # Filter the dataset to include only rows where the outcome is 'died'
    died_df = df[df['outcome'] == 'died']

    # Count deaths per hospital
    death_counts = died_df['hospital_number'].value_counts()

    # Select the top 10 hospitals with the highest death counts
    top_20_hospitals = death_counts.nlargest(20)

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_20_hospitals.index, y=top_20_hospitals.values, palette='viridis')
    plt.xlabel('Hospital Number')
    plt.ylabel('Number of Deaths')
    plt.title('Top 20 Hospitals with the Highest Death Rates')
    plt.xticks(rotation=45)

    # Save plot to a file
    image_path = os.path.join('static', 'images', 'hospital_death_correlation_bar_chart.png')
    plt.savefig(image_path)
    plt.close()

    return render_template('bar_chart_visual.html')







if __name__=='__main__':
    app.run()
    
    



