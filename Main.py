from flask import Flask, render_template
from MachineLearning import *


app = Flask(__name__)

X_train_encoded, X_val_encoded, y_train_encoded, y_val_encoded, y_pred = MachineLearning.machine_learning()

@app.route('/')
def index():
    acc = MachineLearning.accuracy_score(y_val_encoded, y_pred)
    return render_template('accuracy.html', accuracy = acc)
    # return render_template('confusion_matrix.html')



if __name__=='__main__':
    app.run()
    
    



