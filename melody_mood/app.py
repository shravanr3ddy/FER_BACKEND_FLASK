from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('../emotion_detection_model.h5')  # Make sure to provide the correct path

# Dictionary to label all emotion classes
class_labels = {
    0: 'Angry',
    1: 'Happy',
    2: 'Sad'
}

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the image
        img = image.load_img(file, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict the emotion`
        predictions = model.predict(img_array)
        max_index = np.argmax(predictions[0])
        predicted_emotion = class_labels[max_index]

        return jsonify({'predicted_emotion': predicted_emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
