from flask import Flask

UPLOAD_FOLDER = '/home/xcoder963/Desktop/csc/Cervical Spinal Cord/uploads/'
RESULT_FOLDER = '/home/xcoder963/Desktop/csc/Cervical Spinal Cord/yolov5/runs/detect/exp/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
