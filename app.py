from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

train_datagen = ImageDataGenerator()
training_set = train_datagen.flow_from_directory(
    'D:\\CaChua\\tomato\\train',  # Path to the training data folder
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='categorical')
label_map = training_set.class_indices
class_labels = {v: k for k, v in label_map.items()}

# Load mô hình đã huấn luyện
model = load_model('D:\\CaChua\\tomato\\keras_tomato_trained_model2.keras')
labels = class_labels  # Thay bằng nhãn thực tế

def preprocess_image(image):
    """Xử lý ảnh trước khi đưa vào mô hình."""
    image = image.resize((224, 224))  # Kích thước đầu vào của mô hình
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Chuẩn hóa
    return image

@app.route("/", methods=["GET"])
def index():
    """Trang chính."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Dự đoán dựa trên ảnh tải lên."""
    if "file" not in request.files:
        return jsonify({"error": "Không có file được tải lên."})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "File rỗng."})
    
    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        class_idx = np.argmax(predictions[0])
        label = labels[class_idx]
        confidence = float(predictions[0][class_idx])
        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
