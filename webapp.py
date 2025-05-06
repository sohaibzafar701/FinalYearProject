from flask import Flask, render_template, request, url_for, Response, send_from_directory, session, jsonify, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import cv2
import time
import torch
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import pickle
import uuid
import json
import logging
import zipfile
from io import BytesIO
from datetime import datetime
from sqlalchemy import func, extract

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import your feature extraction and age estimation functions
from image_processing import extract_features_from_bboxes, estimate_age, extract_features_and_estimate_age

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/project_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# DronePath model
class DronePath(db.Model):
    __tablename__ = 'drone_paths'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    polygon_coordinates = db.Column(db.Text, nullable=False)
    altitude = db.Column(db.Float, nullable=False)
    speed = db.Column(db.Float, nullable=False)
    gimbal_angle = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# ImageUpload model
class ImageUpload(db.Model):
    __tablename__ = 'image_uploads'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_image_path = db.Column(db.String(255), nullable=False)
    detection_image_path = db.Column(db.String(255), nullable=False)
    labeled_image_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# DetectionResult model
class DetectionResult(db.Model):
    __tablename__ = 'detection_results'
    id = db.Column(db.Integer, primary_key=True)
    image_upload_id = db.Column(db.Integer, db.ForeignKey('image_uploads.id'), nullable=False)
    species = db.Column(db.String(100), nullable=False)
    detection_count = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# AgeEstimation model
class AgeEstimation(db.Model):
    __tablename__ = 'age_estimations'
    id = db.Column(db.Integer, primary_key=True)
    detection_result_id = db.Column(db.Integer, db.ForeignKey('detection_results.id'), nullable=False)
    age = db.Column(db.Enum('Young', 'Middle-Aged', 'Old'), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# Load your trained YOLO model
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'model-yolo.pt')
model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
RESULT_FOLDER = "static/results"
DRONE_PATHS_FOLDER = "static/drone-paths"
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DRONE_PATHS_FOLDER, exist_ok=True)

# Initialize database
with app.app_context():
    db.create_all()

# Authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists!', 'error')
            return render_template('register.html')
        
        password_hash = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=password_hash)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password!', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

# Protect routes
def login_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/djipath")
@login_required
def djipath():
    user_id = session['user_id']
    previous_paths = DronePath.query.filter_by(user_id=user_id).all()
    previous_paths_data = []
    for path in previous_paths:
        try:
            polygon_coordinates = json.loads(path.polygon_coordinates)
            previous_paths_data.append({
                'id': path.id,
                'polygon_coordinates': polygon_coordinates,
                'altitude': path.altitude,
                'speed': path.speed,
                'gimbal_angle': path.gimbal_angle,
                'created_at': path.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for path ID {path.id}: {e}")
            flash('Some previous paths could not be loaded due to data corruption.', 'error')
            continue
    return render_template("djipath.html", previous_paths=previous_paths_data)

@app.route("/save_drone_path", methods=['POST'])
@login_required
def save_drone_path():
    data = request.get_json()
    user_id = session['user_id']
    
    logger.debug(f"Received data: {data}")
    
    try:
        # Validate user exists
        user = User.query.filter_by(id=user_id).first()
        if not user:
            logger.error(f"User with ID {user_id} does not exist")
            return jsonify({'error': 'User does not exist'}), 400
        
        # Validate required fields
        if not all(key in data for key in ['polygon_coordinates', 'altitude', 'speed', 'gimbal_angle']):
            logger.error("Missing required fields in request data")
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Convert and validate JSON data
        polygon_coordinates = json.dumps(data['polygon_coordinates'], separators=(',', ':'))
        
        # Check data size
        if len(polygon_coordinates.encode('utf-8')) > 65535:
            logger.error("Polygon coordinates data too large")
            return jsonify({'error': 'Polygon coordinates data too large.'}), 400
        
        # Convert numeric fields
        altitude = float(data['altitude'])
        speed = float(data['speed'])
        gimbal_angle = float(data['gimbal_angle'])
        
        # Create new path
        new_path = DronePath(
            user_id=user_id,
            polygon_coordinates=polygon_coordinates,
            altitude=altitude,
            speed=speed,
            gimbal_angle=gimbal_angle
        )
        db.session.add(new_path)
        db.session.commit()
        
        # Save KMZ file if kml_content is provided
        if 'kml_content' in data:
            kml_content = data['kml_content']
            kmz_filename = f"path_{new_path.id}.kmz"
            kmz_path = os.path.join(DRONE_PATHS_FOLDER, kmz_filename)
            
            # Create KMZ file
            with zipfile.ZipFile(kmz_path, 'w', zipfile.ZIP_DEFLATED) as kmz:
                kmz.writestr('waylines.kml', kml_content)
            
            logger.info(f"KMZ file saved to {kmz_path}")
        
        logger.info(f"Path saved successfully for user_id {user_id}, path_id {new_path.id}")
        return jsonify({'message': 'Path saved successfully', 'path_id': new_path.id})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving path: {str(e)}")
        return jsonify({'error': f'Failed to save path: {str(e)}'}), 500

@app.route("/download_kmz/<int:path_id>")
@login_required
def download_kmz(path_id):
    user_id = session['user_id']
    
    try:
        # Verify the path exists and belongs to the user
        path = DronePath.query.filter_by(id=path_id, user_id=user_id).first()
        if not path:
            logger.error(f"Path ID {path_id} not found or does not belong to user ID {user_id}")
            flash('Path not found or you do not have access.', 'error')
            return redirect(url_for('djipath'))
        
        # Check if KMZ file exists
        kmz_filename = f"path_{path_id}.kmz"
        kmz_path = os.path.join(DRONE_PATHS_FOLDER, kmz_filename)
        if not os.path.exists(kmz_path):
            logger.error(f"KMZ file {kmz_path} not found")
            flash('KMZ file not found.', 'error')
            return redirect(url_for('djipath'))
        
        # Serve the KMZ file
        return send_from_directory(DRONE_PATHS_FOLDER, kmz_filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading KMZ for path ID {path_id}: {str(e)}")
        flash('Failed to download KMZ file.', 'error')
        return redirect(url_for('djipath'))

@app.route("/optical-images", methods=["GET", "POST"])
@login_required
def optical_images():
    if request.method == "POST":
        results_dict = {}
        user_id = session['user_id']
        
        if 'files[]' not in request.files:
            flash("No files uploaded", 'error')
            return render_template("optical-images.html")
            
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            flash("No files selected", 'error')
            return render_template("optical-images.html")
        
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            orig_result_path = os.path.join(RESULT_FOLDER, filename)
            
            # Save the original image
            file.save(file_path)
            
            # Run YOLO model for species detection
            results = model(file_path)
            results[0].save(orig_result_path)
            
            # Extract bounding boxes
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            
            # Extract features and estimate age
            tree_data = extract_features_and_estimate_age(file_path, bboxes)
            
            # Create labeled image with age annotations
            labeled_img = cv2.imread(file_path)
            for item in tree_data:
                bbox = item['bbox']
                age = item['age']
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if age == "Young" else (0, 165, 255) if age == "Middle-Aged" else (0, 0, 255)
                cv2.rectangle(labeled_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(labeled_img, age, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            labeled_result_path = os.path.join(RESULT_FOLDER, f"labeled_{filename}")
            cv2.imwrite(labeled_result_path, labeled_img)
            
            # Prepare ages for display
            ages = [item['age'] for item in tree_data]
            
            # Save to database
            try:
                # Create ImageUpload record
                image_upload = ImageUpload(
                    user_id=user_id,
                    filename=filename,
                    original_image_path=file_path,
                    detection_image_path=orig_result_path,
                    labeled_image_path=labeled_result_path
                )
                db.session.add(image_upload)
                db.session.flush()  # Get image_upload.id before committing
                
                # Create DetectionResult record (assuming "Palm Tree" is the species)
                detection_result = DetectionResult(
                    image_upload_id=image_upload.id,
                    species="Palm Tree",
                    detection_count=len(bboxes)
                )
                db.session.add(detection_result)
                db.session.flush()  # Get detection_result.id
                
                # Create AgeEstimation records
                for age in ages:
                    age_estimation = AgeEstimation(
                        detection_result_id=detection_result.id,
                        age=age
                    )
                    db.session.add(age_estimation)
                
                db.session.commit()
                logger.info(f"Saved detection results for image {filename} for user_id {user_id}")
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error saving detection results for image {filename}: {str(e)}")
                flash(f"Failed to save results for {filename}.", 'error')
            
            # Prepare results for display
            results_dict[filename] = {
                'original_image_url': url_for("static", filename=f"uploads/{filename}"),
                'detection_image_url': url_for("static", filename=f"results/{filename}"),
                'labeled_image_url': url_for("static", filename=f"results/labeled_{filename}"),
                'detection_result': len(bboxes) > 0,
                'tree_data': tree_data,
                'ages': ages
            }
        
        session['results'] = results_dict
        return render_template("optical-images.html", results=results_dict)

    return render_template("optical-images.html", results=None)

@app.route('/data-visualization')
@login_required
def data_visualization():
    user_id = session['user_id']
    
    try:
        # Species Distribution (Pie Chart)
        species_data = db.session.query(
            DetectionResult.species,
            func.sum(DetectionResult.detection_count).label('total_count')
        ).join(ImageUpload).filter(
            ImageUpload.user_id == user_id
        ).group_by(DetectionResult.species).all()
        
        species_labels = [row.species for row in species_data]
        species_counts = [row.total_count for row in species_data]
        
        # If no data, provide default values
        if not species_labels:
            species_labels = ['Palm Tree', 'Others']
            species_counts = [0, 0]
        
        # Age Distribution (Bar Chart)
        age_data = db.session.query(
            AgeEstimation.age,
            func.count(AgeEstimation.id).label('age_count')
        ).join(DetectionResult).join(ImageUpload).filter(
            ImageUpload.user_id == user_id
        ).group_by(AgeEstimation.age).all()
        
        age_labels = ['Young', 'Middle-Aged', 'Old']
        age_counts = [0, 0, 0]
        for row in age_data:
            if row.age == 'Young':
                age_counts[0] = row.age_count
            elif row.age == 'Middle-Aged':
                age_counts[1] = row.age_count
            elif row.age == 'Old':
                age_counts[2] = row.age_count
        
        # Tree Count Over Time (Line Chart)
        # Group by month, ensure at least 6 months for better visualization
        from datetime import datetime, timedelta
        from dateutil.relativedelta import relativedelta
        
        # Get the earliest and latest dates to define the range
        date_range = db.session.query(
            func.min(ImageUpload.created_at).label('min_date'),
            func.max(ImageUpload.created_at).label('max_date')
        ).filter(ImageUpload.user_id == user_id).first()
        
        if date_range.min_date and date_range.max_date:
            min_date = date_range.min_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            max_date = date_range.max_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + relativedelta(months=1)
        else:
            # Default to last 6 months if no data
            max_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            min_date = max_date - relativedelta(months=6)
        
        # Generate all months in the range
        time_labels = []
        current_date = min_date
        while current_date <= max_date:
            time_labels.append(current_date.strftime('%Y-%m'))
            current_date += relativedelta(months=1)
        
        # Query tree counts by month
        time_data = db.session.query(
            func.date_format(ImageUpload.created_at, '%Y-%m').label('month'),
            func.sum(DetectionResult.detection_count).label('total_count')
        ).join(DetectionResult).filter(
            ImageUpload.user_id == user_id
        ).group_by('month').order_by('month').all()
        
        # Map counts to months, fill missing months with 0
        time_counts = [0] * len(time_labels)
        for row in time_data:
            try:
                index = time_labels.index(row.month)
                time_counts[index] = row.total_count
            except ValueError:
                continue
        
        # If no data, provide default
        if not time_data and not time_labels:
            time_labels = ['No Data']
            time_counts = [0]
        
        logger.debug(f"Time chart data: labels={time_labels}, counts={time_counts}")
        
        # Pass data to template
        chart_data = {
            'species': {
                'labels': species_labels,
                'counts': species_counts
            },
            'age': {
                'labels': age_labels,
                'counts': age_counts
            },
            'time': {
                'labels': time_labels,
                'counts': time_counts
            }
        }
        
        logger.debug(f"Chart data sent to template: {chart_data}")
        return render_template('data-visualization.html', chart_data=chart_data)
    except Exception as e:
        logger.error(f"Error fetching visualization data: {str(e)}")
        flash('Failed to load visualization data.', 'error')
        return render_template('data-visualization.html', chart_data=None)

@app.route("/", methods=["GET", "POST"])
@login_required
def predict_img():
    if request.method == "POST":
        results_dict = {}
        
        if 'files[]' not in request.files:
            flash("No files uploaded", 'error')
            return render_template("index.html")
            
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            flash("No files selected", 'error')
            return render_template("index.html")
        
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            result_path = os.path.join(RESULT_FOLDER, filename)
            
            file.save(file_path)
            results = model(file_path)
            results[0].save(result_path)
            
            bboxes = results[0].boxes.xyxy.numpy()
            features = extract_features_from_bboxes(file_path, bboxes)
            ages = [estimate_age(feature) for feature in features] if features else []
            
            results_dict[filename] = {
                'image_url': url_for("static", filename=f"results/{filename}"),
                'detection_count': len(bboxes),
                'ages': ages
            }
        
        session['results'] = results_dict
        return render_template("index.html", results=results_dict)

    return render_template("index.html", results=None)

@app.route("/download/<filename>")
@login_required
def download_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_frame():
    uploaded_filename = session.get('uploaded_filename')
    if not uploaded_filename:
        return
    
    try:
        subfolders = [f for f in os.listdir(RESULT_FOLDER) if os.path.isdir(os.path.join(RESULT_FOLDER, f))]
        if subfolders:
            latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(RESULT_FOLDER, x)))
            video_path = os.path.join(RESULT_FOLDER, latest_subfolder, uploaded_filename)
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
    except Exception as e:
        print(f"Error in video feed: {e}")
        yield b''

@app.route('/batch-process', methods=['POST'])
@login_required
def batch_process():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
        
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    results_dict = {}
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        result_path = os.path.join(RESULT_FOLDER, filename)
        
        file.save(file_path)
        results = model(file_path)
        results[0].save(result_path)
        
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        features = extract_features_from_bboxes(file_path, bboxes)
        ages = [estimate_age(f) for f in features] if features else []
        
        results_dict[filename] = {
            'image_url': url_for("static", filename=f"results/{filename}", _external=True),
            'detection_count': len(bboxes),
            'ages': ages
        }
    
    return jsonify({'results': results_dict})

if __name__ == "__main__":
    app.run(debug=True)