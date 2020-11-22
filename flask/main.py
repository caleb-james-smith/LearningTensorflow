# main.py
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES

# problem on mac regarding OpenMP:
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
# https://github.com/dmlc/xgboost/issues/1715
# fix:
# As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results.
# For more information, please see http://openmp.llvm.org/
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# fix python crash by setting matplotlib to a non-interactive backend
# https://github.com/matplotlib/matplotlib/issues/14304
plt.switch_backend('Agg')

app = Flask(__name__, static_url_path='/static')
NOISE_DIM = 100
GENERATE_IMG_PATH = 'static/img_generate'
UPLOAD_IMG_PATH = 'static/img_upload'

# Auxiliary Classifier Generative Adversarial Network (ACGAN) trained on MNIST dataset of handwritten digits 0-9
generator = load_model('static/models/acgan_generator_100.h5')

# ResNet image classifier from keras trained on ImageNet
model = ResNet50(weights='imagenet')

# configure location to save images
# this path needs to exist
photos = UploadSet(name='photos', extensions=IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = UPLOAD_IMG_PATH
configure_uploads(app, upload_sets=photos)

# GET: loading website (read)
# POST: sending data to webiste (write)

# main page
@app.route('/')
def main():
    return render_template('main.html')

# about page
@app.route('/about')
def about():
    return render_template('about.html')

# choose digit to generate
@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        label = request.form['value']
        if not label:
            label = 0
        label = int(label)
        noise = np.random.normal(0, 1, size=[1, NOISE_DIM])
        label = np.array([[label]])
        generated_img = generator.predict([noise, label])
        generated_img = generated_img.reshape(28, 28)
        filename = uuid.uuid4().hex[:8] + '.png'
        filepath = GENERATE_IMG_PATH + '/' + filename
        plt.imshow(generated_img, interpolation='nearest', cmap='gray')
        plt.savefig(filepath)
        return redirect(url_for('show_generate', filename=filename))
    return render_template('gen_input.html')

# generated image
@app.route('/gen_photo/<filename>')
def show_generate(filename):
    filepath = '/' + GENERATE_IMG_PATH + '/' + filename
    return render_template('gen_output.html', url=filepath)

# upload an image
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        # save image using a unique id for filename and redirect to view image
        filename = photos.save(request.files['photo'], name=uuid.uuid4().hex[:8] + '.')
        return redirect(url_for('show_upload', filename=filename))
    # render page
    return render_template('upload_input.html')

# show uploaded image and classification
@app.route('/upload_photo/<filename>')
def show_upload(filename):
    debug = False
    # load image, resize, convert to numpy array
    img_path = app.config['UPLOADED_PHOTOS_DEST'] + '/' + filename
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x[np.newaxis, ...]
    x = preprocess_input(x)
    # predict
    y_pred = model.predict(x)
    predictions = decode_predictions(y_pred, top=5)[0]
    url = photos.url(filename)
    # get mostly likely prediction
    indices = np.argmax(predictions, axis=0)
    max_index = indices[2]
    max_class = predictions[max_index][1]
    max_prob  = "{:.1f}".format(100 * predictions[max_index][2])
    if debug:
        for i in range(len(predictions)):
            print("prediction {0}: {1}, {2}".format(i, predictions[i][1], predictions[i][2]))
        print("indices: {0}".format(indices))
        print("max_index: {0}, max_class: {1}, max_prob: {2}".format(max_index, max_class, max_prob))

    # render page
    return render_template('upload_output.html', filename=filename, url=url, predictions=predictions, max_class=max_class, max_prob=max_prob)
