Final Year Project
A Flask-based web application developed as a Final Year Project, featuring dynamic routes, HTML templates, and backend processing. This repository contains two versions: Version 1 (old) and Version 2 (updated).
Setup Guidelines
Prerequisites

Python 3.8 or higher
Git
A web browser

Installation

Clone the Repository:
git clone https://github.com/sohaibzafar701/FinalYearProject.git
cd FinalYearProject


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download the YOLO Model (Optional):

The yolo-model.pt file is not included due to size constraints. Download it from [Google Drive link] and place it in the models/ folder.
Update your application code to use this model if required.


Set Up Environment Variables (Optional):

Create a .env file for sensitive configurations (e.g., Flask secret key):FLASK_ENV=development
SECRET_KEY=your-secret-key





Running the Application

Start the Flask Server:python app.py


Access the Application:
Open a browser and navigate to http://localhost:5000.



Usage

Home Page: Visit http://localhost:5000 to access the main interface.
Routes: Explore endpoints defined in app.py (e.g., /, /about, /predict).
Templates: The application uses 6 HTML templates in the templates/ folder for rendering dynamic content.
Model Integration: If using yolo-model.pt, ensure it’s in models/ and referenced in your code.
Static Files: CSS, JS, and images are in the static/ folder.

Project Structure
FinalYearProject/
├── app.py           # Main Flask application
├── templates/       # 6 HTML templates
├── static/          # CSS, JS, images
├── models/          # Model files (yolo-model.pt not included)
├── requirements.txt # Python dependencies
└── other_files/     # Additional project files

Versions

Version 1.0: Original project (tagged as v1.0).
Version 2.0: Updated project with improved features (tagged as v2.0).

Notes

Ensure yolo-model.pt is downloaded if your application requires it.
Check requirements.txt for all dependencies.
For issues, refer to the GitHub Issues section.

