from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import json

# Configurations - Parameters
MODEL_PATH = 'sneaker_classifier.pth'
CLASS_NAMES_PATH = 'class_names.json'
NUM_CLASSES = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and saved state
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load class names
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
except FileNotFoundError:
    print(f"Error: {CLASS_NAMES_PATH} not found.")
    exit(1)


# Transform image
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])

# Initialize Flask
app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    try:
        # Process image
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image_tensor = image_transforms(
            image).unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)

        # Prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            # Get probabilities
            prob = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(prob, 1)

        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item()

        # Format predicted class
        if isinstance(predicted_class, str):
            predicted_class = ' '.join(
                [word.capitalize() for word in predicted_class.split('_')])
        else:
            predicted_class = str(predicted_class)

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.4f}",
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Model and class names loaded. Starting Flask app\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
