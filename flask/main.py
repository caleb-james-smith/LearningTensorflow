# main.py
import uuid
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES

app = Flask(__name__)

# image classifier from keras
model = ResNet50(weights='imagenet')

# configure location to save images
photos = UploadSet(name='photos', extensions=IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, upload_sets=photos)

# get: loading website
# post: uploading to website

# default url
@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        pass
    return render_template('upload.html')
