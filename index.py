from flask import Flask, render_template, request, redirect, flash, url_for
#import main
import urllib.request
from app import app
from werkzeug.utils import secure_filename
import pydicom
import numpy as np
import os
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
#from model import EffNet

def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data




def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im

@app.route('/')
def main():
    file_path = os.path.join(app.config['RESULT_FOLDER'], 'image.png')
    return render_template("main.html")

@app.route('/report')
def report_main():
    #file_path = os.path.join(app.config['RESULT_FOLDER'], 'image.png')
    return render_template("main.html")

@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],"file.dcm"))
            # Do the conversion here maybe and then save idk
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'],"file.dcm"))
            #ds = pydicom.dcmread('uploads/file.dcm')
            #new_image = ds.pixel_array.astype(float)

            data = read_xray('uploads/file.dcm')

            #scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
            #scaled_image = np.uint8(scaled_image)
            final_image = Image.fromarray(data)
            os.system("rm -rf runs/detect/exp/labels/*")
            os.system("rm -rf runs/detect/exp/*.png")
            final_image.save('uploads/image1.png')
            os.system("python yolov5/detect.py  --source 'uploads/image1.png' --weights 'yolov5/best.pt' --img 512 --save-txt --save-conf --exist-ok")
            file_path = os.path.join(app.config['RESULT_FOLDER'], 'file.dcm')
            os.system("rm uploads/file.dcm")
            cmd_str = "cp runs/detect/exp/image1.png static/"
            os.system(cmd_str)
            return render_template("main.html", user_image = "static/image1.png")

@app.route('/report', methods=['POST'])
def report():
    #print("Hello")
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            data_input = secure_filename(file.filename)
            #print(data_input)
            data_input = data_input[:-4]
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],"file.zip"))
            os.system("rm -rf data/*")
            os.system("unzip uploads/file.zip -d data/")
            os.system("rm -rf runs/detect/exp/labels/*")
            os.system("rm -rf runs/detect/exp/*.png")
            from model import EffNet
            # Do the conversion here maybe and then save idk
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'],"file.dcm"))
            model = EffNet()
            output_data = model.predict(data_input)
            #print(output_data.to_string())
            #name = output_data["StudyInstanceUID"]
            #aff_ver = str(output_data["C1_fracture"]) + "," + str(output_data["C2_fracture"]) + "," + str(output_data["C3_fracture"]) + "," + str(output_data["C4_fracture"]) + "," + str(output_data["C5_fracture"]) + "," + str(output_data["C6_fracture"]) + "," + str(output_data["C7_fracture"]);
            #patient_overall = output_data["patient_overall"]
            #output_data_string = output_data.to_string(index=False, header=False)
            #output_data = output_data[:80] + '\n' + output_data[80:]
            return render_template("main.html", name = output_data[0], C1 = output_data[1], C2 = output_data[2], C3 = output_data[3], C4 = output_data[4], C5 = output_data[5], C6 = output_data[6], C7 = output_data[7], img_arr_path=output_data[8])

if __name__ == "__main__":
    app.run()
