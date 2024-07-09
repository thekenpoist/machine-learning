import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from flask import render_template


# Feature importances function
def feature_importances_visual(model, X_train_encoded):

    # FEATURE IMPORTANCES VISUALIZATION
    importance = model.coef_[0]

    fig, ax = plt.subplots(figsize=(14, 10))
    indices = np.argsort(importance)
    ax.set_title('Feature Importances')
    fig.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.1)
    ax.barh(range(len(indices)), importance[indices], color='b', align='center')
    ax.set_yticks(range(len(indices)), [X_train_encoded.columns[i] for i in indices])
    ax.set_xlabel('Relative Importance')

    # Save to file as image
    image_path = os.path.join('static', 'images', 'feature_importances.png')
    plt.savefig(image_path)
    plt.close(fig)

    return render_template('feature_importances_visual.html')