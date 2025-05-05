from flask import Flask, render_template, request, url_for, Response, send_from_directory, session
import os
import cv2
import time
import torch
from werkzeug.utils import secure_filename
from ultralytics import YOLO  
import pickle

# Import your feature extraction and age estimation functions
from image_processing import extract_features_from_bboxes, estimate_age


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session usage

# Load your trained YOLO model (change path if needed)
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'model-yolo.pt')
model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
RESULT_FOLDER = "static/results"
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/djipath")
def djipath():
    return render_template("djipath.html")

@app.route("/optical-images")
def optical_images():
    return render_template("optical-images.html")

@app.route('/data-visualization')
def data_visualization():
    return render_template('data-visualization.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        file = request.files["file"]
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        result_path = os.path.join(RESULT_FOLDER, filename)

        file.save(file_path)  # Save uploaded file

        # Run prediction
        results = model(file_path)

        # Save result image
        results[0].save(result_path)

        # Get bounding boxes for the detected birds
        bboxes = results[0].boxes.xyxy.numpy()

        # Extract features from the detected areas
        features = extract_features_using_bboxes(file_path, bboxes)
        if features:
            print("Features extracted:", features)  # Debugging step
            age_estimation = estimate_age(features)
        else:
            print("No valid features found")  # Debugging step

        # Estimate the age of the birds from the extracted features
        ages = []
        if features:
            for feature in features:
                age = estimate_age(feature)
                ages.append(age)

        # Generate URL for displaying image
        image_url = url_for("static", filename=f"results/{filename}")

        return render_template("index.html", image_path=image_url, ages=ages)

    return render_template("index.html", image_path=None, ages=None)

@app.route("/download/<filename>")
def download_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route("/optical-images", methods=["GET", "POST"])
def predict_page2():
    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        result_path = os.path.join(RESULT_FOLDER, filename)

        file.save(file_path)
        results = model(file_path)
        results[0].save(result_path)

        # Pull out bounding boxes directly
        # results[0].boxes.xyxy is a tensor; convert to numpy
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        # Extract features straight from those boxes
        features = extract_features_from_bboxes(file_path, bboxes)

        # Estimate an age for each box
        ages = [estimate_age(f) for f in features] if features else []

        image_url = url_for("static", filename=f"results/{filename}")
        return render_template(
            "optical-images.html",
            image_path=image_url,
            # Right:
            detection_result = len(bboxes) > 0,
            ages=ages
        )

    return render_template("optical-images.html", image_path=None, detection_result=None, ages=None)

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_frame():
    uploaded_filename = session.get('uploaded_filename')
    if not uploaded_filename:
        return
    
    subfolders = [f for f in os.listdir(RESULTS_FOLDER) if os.path.isdir(os.path.join(RESULTS_FOLDER, f))]
    if subfolders:
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(RESULTS_FOLDER, x)))
        video_path = os.path.join(RESULTS_FOLDER, latest_subfolder, uploaded_filename)
    else:
        video_path = os.path.join(UPLOAD_FOLDER, uploaded_filename)

    video = cv2.VideoCapture(video_path)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

if __name__ == "__main__":
    app.run(debug=True)
