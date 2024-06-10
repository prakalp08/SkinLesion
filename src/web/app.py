from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.kearas.preprocessing.image import img_to_array

app = Flask(__name__)

MODEL_PATH='models/skin_cancer_model.h5'
model=load_model(MODEL_PATH)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['files']
    image=cv2.imdecode(np.frombuffer(file.read{},np.unit8),cv2.IMREAD_COLOR)
    image=cv2.resize(image,(128,128))
    image= img_to_array(image)
    image=np.expand_dims(image,axis=0)
    predictions=model.predict(image)
    predicted_class=np.argmax(predictions,axis=1)
    return str(predicted_class[0])
if __name__=='__main__':
    app.run(debug=True)