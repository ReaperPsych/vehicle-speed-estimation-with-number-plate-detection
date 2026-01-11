# TrafficAI - Vehicle Speed Estimation & Flow Analysis

TrafficAI is a computer vision system designed to detect vehicles, estimate their real-world speeds, and analyze traffic flow levels from video footage. Built with **Django**, **YOLOv8**, and **OpenCV**, it utilizes homography matrix transformation to map 2D video pixels to real-world coordinates for accurate speed calculation.

## 🚀 Key Features

* **Vehicle Detection:** Utilizes **YOLOv8** to identify cars, motorcycles, buses, and trucks.
* **Speed Estimation:** Calculates vehicle speed (km/h) using **Perspective Transformation (Homography)** to map screen pixels to real-world distance (meters).
* **Traffic Flow Analysis:** Categorizes traffic density into **Light**, **Medium**, or **Heavy** based on vehicles per minute.
* **Number Plate Recognition:** Integrated **EasyOCR** for capturing license plate data.
* **Web Dashboard:** User-friendly Django interface for uploading videos and viewing analysis results.
* **Secure Access:** Restricted login system for authorized users.

## 🛠️ Tech Stack

* **Backend:** Python 3.x, Django 5.x
* **Computer Vision:** OpenCV, Ultralytics YOLOv8, NumPy
* **OCR:** EasyOCR
* **Frontend:** HTML5, Bootstrap 5
* **Database:** SQLite (Default)

## 📂 Project Structure

```text
traffic_project/
├── core/                   # Main Application Logic (Views, Detector, Models)
├── traffic_project/        # Project Settings (ASGI, WSGI, URLs)
├── media/                  # Storage for Uploaded & Processed Videos
├── manage.py               # Django Command Utility
├── yolov8n.pt              # YOLOv8 Nano Model Weights (Auto-downloads on first run)
└── requirements.txt        # List of dependencies
