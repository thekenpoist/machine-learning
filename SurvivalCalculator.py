import pandas as pd
from flask import render_template


# Survival calculator function
def survival_calculator(df):
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

    return render_template('survival_calculator.html', unique_hospital_number=unique_hospital_number, unique_pain=unique_pain,
                           unique_temp_of_extremities=unique_temp_of_extremities, unique_peripheral_pulse=unique_peripheral_pulse,
                           unique_mucous_membrane=unique_mucous_membrane, unique_capillary_refill_time=unique_capillary_refill_time,
                           unique_peristalsis=unique_peristalsis, unique_abdominal_distention=unique_abdominal_distention,
                           unique_nasogastric_tube=unique_nasogastric_tube, unique_nasogastric_reflux=unique_nasogastric_reflux,
                           unique_rectal_exam_feces=unique_rectal_exam_feces, unique_abdomen=unique_abdomen, 
                           unique_abdomo_appearance=unique_abdomo_appearance)