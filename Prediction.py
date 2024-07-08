import pandas as pd
from flask import request, render_template


# Prediction function
def prediction(X_train_encoded, model):

    form_data = {
        'surgery': request.form['surgery'],
        'age': request.form['age'],
        'hospital_number': request.form['hospital_number'],
        'rectal_temp': request.form['rectal_temp'],
        'pulse': request.form['pulse'],
        'respiratory_rate': request.form['respiratory_rate'],
        'temp_of_extremities': request.form['temp_of_extremities'],
        'peripheral_pulse': request.form['peripheral_pulse'],
        'mucous_membrane': request.form['mucous_membrane'],
        'capillary_refill_time': request.form['capillary_refill_time'],
        'pain': request.form['pain'],
        'peristalsis': request.form['peristalsis'],
        'abdominal_distention': request.form['abdominal_distention'],
        'nasogastric_tube': request.form['nasogastric_tube'],
        'nasogastric_reflux': request.form['nasogastric_reflux'],
        'nasogastric_reflux_ph': request.form['nasogastric_reflux_ph'],
        'rectal_exam_feces': request.form['rectal_exam_feces'],
        'abdomen': request.form['abdomen'],
        'packed_cell_volume': request.form['packed_cell_volume'],
        'total_protein': request.form['total_protein'],
        'abdomo_appearance': request.form['abdomo_appearance'],
        'abdomo_protein': request.form['abdomo_protein'],
        'surgical_lesion': request.form['surgical_lesion']
    }

    # Create a DataFrame from the form data
    df_form = pd.DataFrame([form_data])

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

    return render_template('prediction.html', probability=str(round(probability[0] * 100, 2)), form_data=form_data)