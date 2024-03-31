#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from keras.models import load_model
import streamlit as st

# Load the pre-trained model
model = load_model('sign_language_model.h5')
class_labels = ['A', 'B', 'C', ...]  # Your class labels

# Streamlit app layout
st.title('Sign Language Recognition')

# Open webcam
cap = None

# Function to make predictions
def predict_sign(frame):
    frame = cv2.resize(frame, (200, 200)) / 255.0
    predictions = model.predict(np.expand_dims(frame, axis=0))
    predicted_class = np.argmax(predictions)
    
    # Print predicted class index for debugging
    print("Predicted Class Index:", predicted_class)
    
    if predicted_class < len(class_labels):
        return class_labels[predicted_class]
    else:
        return "Unknown Class"  # Handle out-of-range predictions

# Main function to run the app
def main():
    global cap

    # Check if the camera is opened successfully
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to access the camera. Please ensure that the camera is connected and accessible.")
        st.stop()  # Stop execution of the app

    # Button to toggle camera on/off
    if st.button("Toggle Camera"):
        if cap.isOpened():
            cap.release()
            st.write("Camera is Off")
        else:
            cap = cv2.VideoCapture(0)
            st.write("Camera is On")

    # Main loop to capture frames and make predictions
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to capture frame from the camera.")
            break

        # Make prediction
        predicted_label = predict_sign(frame)

        # Display the frame and predicted label
        st.image(frame, caption='Sign Language Gesture')
        st.write('Predicted Label:', predicted_label)

        # Break the loop if 'Quit' button is pressed
        if st.button('Quit'):
            break

    # Release the webcam
    if cap is not None:
        cap.release()

# Run the main function
if __name__ == "__main__":
    main()
