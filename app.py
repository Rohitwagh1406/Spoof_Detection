from flask import Flask, render_template, Response, request, send_file
import cv2
import numpy as np
from keras.models import model_from_json
import os
from io import BytesIO
from PIL import Image
from flask_cors import CORS
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))
CORS(app)
face_cascade_path = r"models/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print(f"Error loading face cascade from {face_cascade_path}")

model_json_path = r'antispoofing_models/antispoofing_model.json'
model_weights_path = r'antispoofing_models/antispoofing_model.h5'

with open(model_json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights(model_weights_path)
print("Model loaded from disk")

def detect_and_predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y - 5:y + h + 5, x - 5:x + w + 5]
        resized_face = cv2.resize(face, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)

        preds = model.predict(resized_face)[0]
        print(f"Predictions: {preds}")  # Debug: Print predictions

        label = 'real' if preds <= 0.5 else 'spoof'
    #     color = (0, 255, 0) if preds <= 0.5 else (0, 0, 255)

    #     cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # return frame
    return label
    


def generate_frames():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        success, frame = video.read()
        if not success:
            print("Error: Failed to read frame from video source.")
            break

        frame = detect_and_predict(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            continue

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    if 'frame' not in request.files:
        return "No frame found", 400

    frame = request.files['frame'].read()
    npimg = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    processed_frame = detect_and_predict(img)
    # label = detect_and_predict(img)
    # _, buffer = cv2.imencode('.jpg', processed_frame)
    # io_buf = BytesIO(buffer)

    del frame
    # return label
    return processed_frame
    # return send_file(io_buf, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)
