# LearningTensorflow


## Running Flask Applications

Documentation: https://flask.palletsprojects.com/

Installation: https://flask.palletsprojects.com/en/1.1.x/installation/#installation

Quickstart: https://flask.palletsprojects.com/quickstart/

Installing Flask:
```
pip install Flask
```

Before running Flask, run these setup commands. FLASK_APP needs to be assigned, but FLASK_ENV and FLASK_DEBUG are optional and can be assigned for development and debugging mode, respectively.
```
export FLASK_APP=main.py
export FLASK_ENV=development
export FLASK_DEBUG=1
```

Running Flask:
```
cd flask
flask run
```

To view the main page after running Flask, open a web browser and go to http://127.0.0.1:5000/.
