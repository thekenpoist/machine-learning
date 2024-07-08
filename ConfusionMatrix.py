import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from flask import render_template


# Confusion matrix function
def confusion_matrix_visual(y_val_encoded, y_pred):

    # CONFUSION MATRIX VISUALIZATION
    conf_matrix = confusion_matrix(y_val_encoded, y_pred)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)

    # Labels, title and ticks
    label_font = {'size': '16'}  # Adjust to fit
    ax.set_xlabel('Predicted', fontdict=label_font)
    ax.set_ylabel('Actual', fontdict=label_font)
    ax.set_title('Confusion Matrix', fontdict=label_font)
    ax.xaxis.set_ticklabels(['Died', 'Lived'])
    ax.yaxis.set_ticklabels(['Died', 'Lived'])

    # Save to file as image
    image_path = os.path.join('static', 'images', 'confusion_matrix.png')
    plt.savefig(image_path)
    plt.close(fig)

    return render_template('confusion_matrix_visual.html')