import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import render_template


# Bar graph function
def bar_graph_visual(df):

    # BAR GRAPH VISUALIZATION - Top 20 hospitals with the highest death rate.
    # Filter the dataset to include only rows where the outcome is 'died'
    died_df = df[df['outcome'] == 'died']

    # Count deaths per hospital
    death_counts = died_df['hospital_number'].value_counts()

    # Select the top 20 hospitals with the highest death counts
    top_20_hospitals = death_counts.nlargest(20)

    # Convert to NumPy arrays for sorting
    indices = np.array(top_20_hospitals.index)
    values = np.array(top_20_hospitals.values)

    # Sort by values
    sorted_order = np.argsort(values)[::-1]
    sorted_indices = indices[sorted_order]
    sorted_values = values[sorted_order]

    # arrange the bars in descending order
    y_pos = np.arange(len(sorted_indices))

    # Create the bar graph
    plt.figure(figsize=(12, 8))
    plt.bar(y_pos, sorted_values, align='center', alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(sorted_indices))))
    plt.xticks(y_pos, sorted_indices, rotation=45)
    plt.xlabel('Hospital Number')
    plt.ylabel('Number of Deaths')
    plt.title('Top 20 Hospitals with the Highest Death Rates')

    # Save graph to file
    image_path = os.path.join('static', 'images', 'hospital_death_correlation_bar_graph.png')
    plt.savefig(image_path)
    plt.close()

    return render_template('bar_graph_visual.html')