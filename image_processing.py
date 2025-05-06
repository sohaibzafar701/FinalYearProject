import cv2
import numpy as np
import os
import pickle
from skimage.feature import graycomatrix, graycoprops
import joblib


# Load the age estimation model (KMeans)
AGE_MODEL_PATH = os.path.join(os.getcwd(), 'models', 'model-age-estimation.pkl')
age_model = joblib.load(AGE_MODEL_PATH)

def extract_features_from_bboxes(image_path, bboxes):
    """
    Given an image and a list/array of bounding boxes [x_min, y_min, x_max, y_max],
    extract crown diameter, green intensity, and GLCM features for each box.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    features_list = []
    for (x_min, y_min, x_max, y_max) in bboxes:
        cropped = img[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            continue

        # 1) Contour-based crown diameter
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        crown_diameter = 2 * radius

        # 2) Green intensity
        green = cropped[:, :, 1]
        green_intensity = np.mean(green) / 255.0

        # 3) GLCM texture features
        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]

        features_list.append([crown_diameter, green_intensity, contrast, dissimilarity])




    return features_list

def estimate_age(features):
    """
    Map a feature vector to one of your three age clusters.
    """
    cluster = age_model.predict([features])[0]
    age_groups = {0: "Young", 1: "Middle-Aged", 2: "Old"}
    return age_groups.get(cluster, "Unknown")


# // added latetr

import cv2
import numpy as np
import os
import pickle
from skimage.feature import graycomatrix, graycoprops
import joblib

# Load the age estimation model (KMeans)
AGE_MODEL_PATH = os.path.join(os.getcwd(), 'models', 'model-age-estimation.pkl')
age_model = joblib.load(AGE_MODEL_PATH)

def extract_features_from_bboxes(image_path, bboxes):
    """
    Given an image and a list/array of bounding boxes [x_min, y_min, x_max, y_max],
    extract crown diameter, green intensity, and GLCM features for each box.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    features_list = []
    for (x_min, y_min, x_max, y_max) in bboxes:
        cropped = img[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            continue

        # 1) Contour-based crown diameter
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        crown_diameter = 2 * radius

        # 2) Green intensity
        green = cropped[:, :, 1]
        green_intensity = np.mean(green) / 255.0

        # 3) GLCM texture features
        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]

        features_list.append([crown_diameter, green_intensity, contrast, dissimilarity])

    return features_list

def estimate_age(features):
    """
    Map a feature vector to one of your three age clusters.
    """
    cluster = age_model.predict([features])[0]
    age_groups = {0: "Young", 1: "Middle-Aged", 2: "Old"}
    return age_groups.get(cluster, "Unknown")

# ✅ NEW FUNCTION to return bbox with age info
def extract_features_and_estimate_age(image_path, bboxes):
    """
    Extract features from each bbox and estimate the age, returning both bbox and age
    """
    results = []
    img = cv2.imread(image_path)
    if img is None:
        return []

    for (x_min, y_min, x_max, y_max) in bboxes:
        cropped = img[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            continue

        # 1) Contour-based crown diameter
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        crown_diameter = 2 * radius

        # 2) Green intensity
        green = cropped[:, :, 1]
        green_intensity = np.mean(green) / 255.0

        # 3) GLCM texture features
        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]

        features = [crown_diameter, green_intensity, contrast, dissimilarity]
        age = estimate_age(features)

        results.append({
            'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
            'age': age
        })

    return results
