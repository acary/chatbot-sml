# Chatbot application for intent classification

Serving a simple machine learning model as a webservice using [flask](http://flask.pocoo.org/).

## Getting Started

### Clone repository to your local machine:

1. From your intended location, run: ```git clone [repository]```

2. Cd to the folder

###  Set up virtual environment:

1. Run: ```python3 -m venv env```
    - If you do not have virtualenv set up, run  ```pip install virtualenv ```

2. To activate, run: ```source env/bin/activate```
    - (To deactivate, run: deactivate)

3. Ensure your virtual environment is running Python 3.x,
    - verify by running: ```python --version```

### Install requirements and run application

1. Run: ```pip install -r requirements.txt``` to install requirements

2. Run: ```python main.py``` to serve the model as a REST Web Service
