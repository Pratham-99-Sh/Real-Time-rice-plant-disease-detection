import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys


model = load_model('model.keras')
class_names = ['Bacterial blight', 'Brown spot', 'Leaf smut']


def preprocess_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0 
    return np.expand_dims(img, axis=0) 


def predict_image(image_path):
    image = preprocess_image(image_path)

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")

    original_img = cv2.imread(image_path)
    output_text = f"{class_names[predicted_class]}: {confidence:.2f}%"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_img, output_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Prediction", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
    else:
        image_path = sys.argv[1]
        predict_image(image_path)

# Command to use this script
# python predict_image.py /path/to/image.jpg