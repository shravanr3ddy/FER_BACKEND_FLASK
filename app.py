from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import io
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load your trained model
model = load_model('./emotion_detection_model.h5')  # Make sure to provide the correct path

# Dictionary to label all emotion classes
class_labels = {
    0: 'angry',
    1: 'happy',
    2: 'sad'
}

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/', methods=['GET'])
def testing_emotion():
    return jsonify({'message': 'I am happy to connect'})

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    app.logger.debug("Predict emotion called")
    if 'file' not in request.files:
        app.logger.error("No file part")
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        app.logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    try:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)
        img = image.load_img(in_memory_file, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)
        max_index = np.argmax(predictions[0])
        predicted_emotion = class_labels[max_index]
        app.logger.debug(f"Predicted emotion: {predicted_emotion}")
        return jsonify({'predicted_emotion': predicted_emotion})
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
