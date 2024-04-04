# Import necessary libraries
import cv2
import numpy as np
from keras.models import load_model
import streamlit as st

# Function to load the pre-trained model
def load_sign_language_model(model_path):
    return load_model(model_path)

# Function to make predictions
def predict_sign(frame, model, class_labels):
    # Preprocess the frame
    frame = cv2.resize(frame, (200, 200)) / 255.0
    # Make predictions using the model
    predictions = model.predict(np.expand_dims(frame, axis=0))
    # Get the predicted class index
    predicted_class = np.argmax(predictions)
    # Get the predicted label
    if predicted_class < len(class_labels):
        return class_labels[predicted_class]
    else:
        return "Unknown Class"  # Handle out-of-range predictions

# Streamlit app layout
st.title('Sign Language Recognition')

# Option to use pre-captured sample data
use_sample_data = st.checkbox('Use Pre-captured Sample Data')

if use_sample_data:
    # Allow user to upload pre-captured sample data
    uploaded_file = st.file_uploader('Upload Pre-captured Sample Data', type=['mp4', 'avi', 'jpg', 'png'])
    
    if uploaded_file is not None:
        # Check the file extension
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension in ['mp4', 'avi']:
            # Process video file
            st.write("Processing video file...")
            # Add video processing logic here
        elif file_extension in ['jpg', 'png']:
            # Process image file
            st.write("Processing image file...")
            # Add image processing logic here
else:
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to access the camera.")
        st.stop()

    # Load the pre-trained model and class labels
    model_path = 'sign_language_model.h5'  # Update with the correct path
    model = load_sign_language_model(model_path)
    class_labels = ['A', 'B', 'C', ...]  # Update with your class labels

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to capture frame from the camera.")
            break

        # Make prediction
        predicted_label = predict_sign(frame, model, class_labels)

        # Display the frame and predicted label
        st.image(frame, caption='Sign Language Gesture')
        st.write('Predicted Label:', predicted_label)

        # Break the loop if 'Quit' button is pressed
        if st.button('Quit'):
            break

    # Release the webcam and close the application
    cap.release()
