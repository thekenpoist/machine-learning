import os
import matplotlib.pyplot as plt
import pandas as pd
from flask import render_template


# Pie chart function
def pie_chart_visual(X_val_encoded, y_val_encoded):

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
    ax.set_title('Distribution of Pain Levels for Horses that Died from Colic')
    
    # Save chart to file as image
    image_path = os.path.join('static', 'images', 'pain_distribution_pie_chart.png')
    plt.savefig(image_path)
    plt.close(fig)

    return render_template('pie_chart_visual.html')