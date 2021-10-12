from __future__ import division, print_function
import os
import numpy as np
import codecs
import textract

with open('Recommondations/fall army worm.txt', encoding='utf-8') as f:
    input1 = f.read()
#print(input1)

#with open('D:/M.Tech/S.Y/DP3/diseases/marathi2.txt', encoding='utf-8') as f:
    #input2 = f.read()

#with open('D:/M.Tech/S.Y/DP3/diseases/marathi3.txt', encoding='utf-8') as f:
    #input3 = f.read()

with open('Recommondations/maize stem borer.txt', encoding='utf-8') as f:
    input4 = f.read()

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
from PIL import Image, ImageOps
import streamlit as st

# Define a flask app
app = Flask(__name__)


def model_predict(img_data, model):
    size=(224,224)
    img = ImageOps.fit(img_data, size)

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

@app.route('/maize_predict', methods=['GET', 'POST'])
def maize_upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        if f is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(f)
            st.image(image, use_column_width=True)
        # Model saved with Keras model.save()
        MODEL_PATH = 'models/maize model.h5'

        # Load your trained model
        model = load_model(MODEL_PATH)

        # Make prediction
        preds = model_predict(file_path, model) # call prediction function
	
	

        pest_list=['फॉल आर्मी वर्म','ग्रीन स्टिंक बग','मक्का बेधक','तना छेदक','गुलाबी तना बेधक (Pink Stem Corer)', 'स्पोडोप्टेरा लिटुरा','अपलोड की गई छवि डेटासेट पर उपलब्ध नहीं है', 'स्वस्थ मकई का पत्ता, अगर प्रभावित हिस्सा है तो फिर से तस्वीर अपलोड करें']    

        result = pest_list[int(preds)]
        return result
    return None

@app.route('/rice_predict', methods=['GET', 'POST'])
def rice_upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        if f is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(f)
            st.image(image, use_column_width=True)
        # Model saved with Keras model.save()
        MODEL_PATH = 'models/rice model.h5'

        # Load your trained model
        model = load_model(MODEL_PATH)

        # Make prediction
        preds = model_predict(file_path, model) # call prediction function
	
	

        pest_list=['बीट वेबवर्म','ब्राउन प्लांट हूपर','यूबलम्मा','गांधी बग','हरा हूपर','हॉर्न कैटरपिलर','लीफ फोल्डर','राइस मील्यबग','राइस स्किपर','स्टेम बोरर','येलो टेल मोत','अपलोड की गई छवि डेटासेट पर उपलब्ध नहीं है', 'स्वस्थ चावल का पत्, अगर प्रभावित हिस्सा है तो फिर से तस्वीर अपलोड करे']    

        result = pest_list[int(preds)]
        return result
    return None

@app.route('/rice_disease', methods=['GET', 'POST'])
def riceD_upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        if f is None:
            st.text("Please upload an image file")
        else:
            image = Image.open(f)
            st.image(image, use_column_width=True)
        # Model saved with Keras model.save()
        MODEL_PATH = 'models/Rice disease.h5'

        # Load your trained model
        model = load_model(MODEL_PATH)

        # Make prediction
        preds = model_predict(file_path, model) # call prediction function
	
	

        disease_list=['बैक्टीरियल ब्लाइट','बैक्टीरियल लीफ स्ट्रीक','ब्राउन लीफ स्पॉट','फॉल्स स्मुट','राइस ब्लास्ट रोग','शेअथ ब्लाईट ऑफ राइस','शेअथ रोट ऑफ राइस','अपलोड की गई छवि डेटासेट पर उपलब्ध नहीं है', 'स्वस्थ चावल का पत्, अगर प्रभावित हिस्सा है तो फिर से तस्वीर अपलोड करें']    

        result = disease_list[int(preds)]
        return result
    return None

if __name__ == '__main__':
    
    app.run(debug=True)
