# app.py
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from model import CNN
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
model = CNN(num_classes=36)
model.load_state_dict(torch.load('model_epoch3.pth', map_location=torch.device('cpu')))
model.eval()

# Class labels
class_names = ["Ain", "Alf_Hamza_Above", "Alf_Hamza_Under", "Alf", "Baa", "Daad", "Dal", "Faa", "Gem", "Gen", "Ha", "Haa", "Hamza", "Kaf", "Khaa", "Lam_Alf_Hamza", "Lam_Alf", "Lam_Alf_Mad", "Lam", "Mem", "Non", "Qaf", "Raa", "Saad", "Shen", "Sin", "Taa", "Tah", "Thaa", "Waw_Hamza", "Waw", "Yaa_Dot", "Yaa", "Zah", "Zal", "Zin"]

# Preprocess image
def preprocess_image(image):
    image = image.convert('L')  # grayscale
    image = image.resize((64, 64))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64]
    return image

# Route to predict
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream)

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
