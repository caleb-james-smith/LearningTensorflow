# main.py
from flask import Flask

app = Flask(__name__)

# default url
@app.route('/')
def hello_world():
    return 'Hello and welcome!'

# testing images
@app.route('/images/')
def images():
    return 'Where are all the pictures?'


