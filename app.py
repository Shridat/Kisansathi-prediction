from __future__ import division, print_function
import os
import numpy as np
import codecs
import textract

#with open('D:/M.Tech/S.Y/DP3/diseases/marathi1.txt', encoding='utf-8') as f:
    #input1 = f.read()
#print(input1)

#with open('D:/M.Tech/S.Y/DP3/diseases/marathi2.txt', encoding='utf-8') as f:
    #input2 = f.read()

#with open('D:/M.Tech/S.Y/DP3/diseases/marathi3.txt', encoding='utf-8') as f:
    #input3 = f.read()

#with open('D:/M.Tech/S.Y/DP3/diseases/marathi4.txt', encoding='utf-8') as f:
    #input4 = f.read()

#with open('D:/M.Tech/S.Y/DP3/diseases/marathi5.txt', encoding='utf-8') as f:
    #input5 = f.read()
#with open('D:/M.Tech/S.Y/DP3/diseases/marathi6.txt', encoding='utf-8') as f:
    #input6 = f.read()

#text = textract.process("D:/M.Tech/S.Y/DP3/diseases/marathi2.docx")
#text = text.decode("utf-8")

# Keras

from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename # use to store the file name
# Flask utils
from flask import Flask, redirect, url_for, request, render_template


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/maize model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# print('Model loaded. Start serving...')

#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = x * 1./255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x]) #Stack arrays in sequence vertically (row wise).
    res = np.argmax(model.predict(images)) #Returns the indices of the maximum values .
    return res

#@app.route(’/’) , where @app is the name of the object containing our Flask app
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__) # current working directory 
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model) # call prediction function
	
	

        disease_list=['Fall Army Worm','Green Stink Bug','Maize Cob Borer','Maize Stem Borer','Pink Stem Borer', 'Spodoptera litura','No Leaf', 'Healthy Corn Leaf']    

        result = disease_list[int(preds)]
        return result
    return None



if __name__ == '__main__':
    
    app.run(debug=True)
