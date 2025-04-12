# Create a project folder and open it in terminal

## Make sure the folder structure is maintained:-

    project/
    ├── app.py
    ├── templates/
    │ ├── index.html
    │ └── result.html
    ├── static/
    │ ├── style.css # Put CSS files
    │ ├── script.js # Put JavaScript files
    │ └── images/ # Images (optional)
    ├── readme.md
    └── .gitignore
    └── requirements.txt

# Then Run these commands:-

## Creates a python environment

    python -m venv env

### env is the name of the environment

## Activates the environment

    .\env\Scripts\activate

## Decativate the environment

    deactivate

## Install Dependencies

    pip install -r requirements.txt

## Update the requirements.txt file each time when a new module or libary is installed using pip

# Link Js and CSS Like this:-

    <!DOCTYPE html>
    <html>
    <head>
        <title>My Flask App</title>
        <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
    </head>
    <body>
        Your HTML content here
        <script src="{{ url_for('static', filename='script.js') }}"></script>
    </body>
    </html>

# Link the css and js with flask like this :-

    app = Flask(__name__)
    app.config['STATIC_FOLDER'] = 'static'

# Run this to start the program at port 127.0.0.1:5000 or ctrl click on the link:-

    python app.py
