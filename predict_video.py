import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model.keras')
class_names = ['Bacterial blight', 'Brown spot', 'Leaf smut']

def preprocess_frame(frame, img_size=(128, 128)):
    img = cv2.resize(frame, img_size)
    img = img / 255.0 
    return np.expand_dims(img, axis=0) 

def predict_on_video(video_path=None):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame)

        predictions = model.predict(preprocessed_frame)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        output_text = f"{class_names[predicted_class]}: {confidence:.2f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, output_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Video Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = None  # Set to None for webcam or provide video file path
    predict_on_video(video_path)
