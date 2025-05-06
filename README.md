# Final Year Project

A **Flask-based web application** developed as a Final Year Project, featuring dynamic routes, HTML templates, and backend processing for advanced functionality. This repository contains two versions: **Version 1** (original) and **Version 2** (updated).

## Features

- **Dynamic Routes**: Supports multiple endpoints for user interaction
- **HTML Templates**: Renders 6 responsive templates for a seamless web interface
- **Backend Processing**: Integrates with external models (e.g., YOLO) for enhanced functionality
- **Modular Structure**: Organized codebase with routes, templates, and static assets
- **Versioned Releases**: Includes Version 1 (legacy) and Version 2 (improved)

## Project Structure

```
FinalYearProject/
├── .env                  # Environment variables (optional)
├── .gitignore            # Ignored files (e.g., venv, yolo-model.pt)
├── requirements.txt      # Python dependencies
├── app.py                # Main Flask application
├── templates/            # 6 HTML templates
│   ├── index.html
│   ├── about.html
│   └── [other templates]
├── static/               # CSS, JS, images
│   ├── css/
│   ├── js/
│   └── images/
├── models/               # Model files (yolo-model.pt not included)
└── other_files/          # Additional project files (50+)
```

## Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/sohaibzafar701/FinalYearProject.git
cd FinalYearProject
```

2. **Set up a Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate  # On Linux/Mac: source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the YOLO Model (Optional)**

- The `yolo-model.pt` file is not included due to size constraints. Download it from [Google Drive link](https://drive.google.com/your-model-link) and place it in the `models/` folder.
- Update your application code to reference this model if required.

5. **Configure Environment Variables (Optional)**

Create a `.env` file in the project root:

```
FLASK_ENV=development
SECRET_KEY=your-secret-key
```

6. **Run the Application**

```bash
python app.py
```

The application will be available at `http://localhost:5000`.

## Usage

- **Home Page**: Visit `http://localhost:5000` to access the main interface.
- **Routes**: Explore endpoints defined in `app.py` (e.g., `/`, `/about`, `/predict`).
- **Templates**: The application uses 6 HTML templates in the `templates/` folder for dynamic content.
- **Model Integration**: If using `yolo-model.pt`, ensure it’s in `models/` and referenced in your code.
- **Static Files**: CSS, JS, and images are served from the `static/` folder.

### Example Routes

- **GET /**: Renders the home page (`index.html`)
- **GET /about**: Displays the about page
- **POST /predict**: Processes input data (if applicable)

## Version Information

- **Version 1.0**: Original project (tagged as `v1.0`)
- **Version 2.0**: Updated project with enhanced features (tagged as `v2.0`)

## Notes

- Ensure `yolo-model.pt` is downloaded if required by your application.
- Verify all dependencies are listed in `requirements.txt`.
- For issues, refer to the [GitHub Issues](https://github.com/sohaibzafar701/FinalYearProject/issues) section.
- Contact the repository owner for the `yolo-model.pt` Google Drive link if not provided.
