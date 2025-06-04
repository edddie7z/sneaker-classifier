# Sneaker Classifier

## Overview

This project is a Sneaker Classifier capable of identifying 50 different classes of sneakers. It utilizes a deep learning model trained on a comprehensive dataset of sneaker images. The application consists of a Python backend for model training and inference, and a web-based frontend for user interaction.

## Features

*   Classifies 50 different types of sneakers.
*   Web interface for uploading images and viewing classification results.
*   Backend built with Python, potentially using Flask for the API.
*   Frontend built with HTML, CSS, and JavaScript.

## Dataset

The model was trained on the "Sneakers Classification" dataset available on Kaggle, created by Nikolas Gegenava.
*   **Dataset Link:** [https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification/data](https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification/data)
*   The dataset contains images for 50 classes of sneakers.
*   For the current model (`sneaker_classifier.pth`):
    *   Total images used: 5953
    *   Training images: 4763
    *   Validation images: 1190

## Model

The classification model is based on the ResNet18 architecture.
*   **Model File:** `backend/sneaker_classifier.pth`
*   **Training Details (from `log3.txt`):**
    *   The model was trained for 20 epochs.
    *   The final layer of the model is `Linear(in_features=512, out_features=50, bias=True)`.
    *   Best validation accuracy achieved: **83.19%**.

## Project Structure

```
├── backend/
│   ├── app.py             # Main backend application (e.g., Flask API)
│   ├── train_model.py     # Script for training the model
│   ├── sneaker_classifier.pth # Trained model weights
│   ├── class_names.json   # List of sneaker class names
│   ├── class_names.py     # Helper script for class names
│   ├── requirements.txt   # Python dependencies
│   ├── data/              # Image dataset (organized by class)
│   └── logs/              # Training logs (e.g., log3.txt)
├── frontend/
│   ├── index.html         # Main HTML page for the frontend
│   ├── script.js          # JavaScript for frontend logic
│   └── style.css          # CSS for styling the frontend
└── README.md              # This file
```

## Setup and Installation

### Backend

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Frontend

The frontend consists of static HTML, CSS, and JavaScript files. It can be served by any simple HTTP server or by the backend if configured to do so.

## Usage

### Training the Model

1.  Ensure your dataset is correctly placed in the `backend/data/` directory, organized into subdirectories for each class.
2.  Run the training script:
    ```bash
    cd backend
    python train_model.py
    ```
    The trained model will be saved as `sneaker_classifier.pth`. Training logs will be saved in the `backend/logs/` directory.

### Running the Application

1.  **Start the backend server (e.g., if `app.py` is a Flask app):**
    ```bash
    cd backend
    python app.py
    ```
2.  **Open the frontend:**
    Open `frontend/index.html` in your web browser.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

Specify your project's license here (e.g., MIT, Apache 2.0). If not specified, assume it's proprietary.
