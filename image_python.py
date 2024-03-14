import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pyttsx3
import time
engine = pyttsx3.init() 
rate = engine.getProperty('rate')                       
engine.setProperty('rate', 190)
engine.say("Welcome to traffic sign recognition system!")
engine.runAndWait()

model_path = 'keras_model.h5'  
teachable_model = load_model(model_path)

with open('labels.txt', 'r') as file:
    labels = file.read().splitlines()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    img = cv2.resize(frame, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = teachable_model.predict(img_array)

    predicted_label_index = np.argmax(predictions)
    predicted_label = labels[predicted_label_index]
    confidence = predictions[0][predicted_label_index]  

    label_text = f"Predicted Label: {predicted_label}"
    confidence_text = f"Confidence: {confidence:.2f}"
    print(label_text)
    if float(confidence_text[13:]) > 0.79 and label_text[19:]!="Give Way":
        engine.say(label_text[19:])
        engine.runAndWait()
        time.sleep(1)
    
    cv2.putText(frame, label_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(frame, confidence_text, (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Teachable Machine Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()