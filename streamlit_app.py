#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from keras.models import load_model
import streamlit as st

# Load the pre-trained model
model = load_model('https://github.com/vijay011/sign/blob/46d2c1a882831a230c111bb040c44d594eefaa96/sign_language_model.h5')
class_labels = ['A', 'B', 'C', ...]  # Your class labels

# Streamlit app layout
st.title('Sign Language Recognition')

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


# Open webcam
cap = cv2.VideoCapture(0)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error capturing frame from the camera.")
        break

    # Make prediction
    predicted_label = predict_sign(frame)

    # Display the frame and predicted label
    st.image(frame, caption='Sign Language Gesture')
    st.write('Predicted Label:', predicted_label)

    # Break the loop if 'q' is pressed
    if st.button('Quit'):
        break

# Release the webcam and close the application
cap.release()

