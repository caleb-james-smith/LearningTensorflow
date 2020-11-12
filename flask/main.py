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
# this path needs to exist
photos = UploadSet(name='photos', extensions=IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, upload_sets=photos)

# get: loading website
# post: uploading to website

# default url: upload an image
@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        # save image using a unique id for filename and redirect to view image
        filename = photos.save(request.files['photo'], name=uuid.uuid4().hex[:8] + '.')
        return redirect(url_for('show', filename=filename))
    # render page
    return render_template('upload.html')

# show specific image
@app.route('/photo/<filename>')
def show(filename):
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
    #print("max_index: {0}, max_class: {1}, max_prob: {2}".format(max_index, max_class, max_prob))
    # render page
    return render_template('view_results.html', filename=filename, url=url, predictions=predictions, max_class=max_class, max_prob=max_prob)
