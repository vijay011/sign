import cv2
import numpy as np
from keras.models import load_model
import streamlit as st

# Load the pre-trained model
model = load_model('sign_language_model.h5')
class_labels = ['A', 'B', 'C', ...]  # Your class labels

# Streamlit app layout
st.title('Sign Language Recognition')

# Function to preprocess and resize images
def preprocess_image(image):
    resized_image = cv2.resize(image, (200, 200)) / 255.0
    return resized_image

# Function to make predictions
def predict_sign(frame):
    if frame.ndim == 3:  # If it's an image
        frame = preprocess_image(frame)
        predictions = model.predict(np.expand_dims(frame, axis=0))
        predicted_class = np.argmax(predictions)
        if predicted_class < len(class_labels):
            return class_labels[predicted_class]
        else:
            return "Unknown Class"  # Handle out-of-range predictions
    elif frame.ndim == 4:  # If it's a video
        predicted_labels = []
        for image in frame:
            preprocessed_image = preprocess_image(image)
            predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
            predicted_class = np.argmax(predictions)
            if predicted_class < len(class_labels):
                predicted_labels.append(class_labels[predicted_class])
            else:
                predicted_labels.append("Unknown Class")  # Handle out-of-range predictions
        return predicted_labels

# Option to use pre-captured sample data
use_sample_data = st.checkbox('Use Pre-captured Sample Data')

if use_sample_data:
    # File uploader for uploading pre-captured sample data
    uploaded_file = st.file_uploader('Upload Pre-captured Sample Data', type=['mp4', 'avi', 'jpg', 'png'])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension in ['mp4', 'avi']:
            # Read video file and process each frame
            video_bytes = uploaded_file.read()
            video_nparray = np.frombuffer(video_bytes, np.uint8)
            video_capture = cv2.VideoCapture()
            video_capture.open(video_nparray)
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                predicted_labels = predict_sign(frame)
                if isinstance(predicted_labels, list):
                    for label in predicted_labels:
                        st.write('Predicted Label:', label)
                else:
                    st.write('Predicted Label:', predicted_labels)
                st.image(frame, caption='Sign Language Gesture')
        elif file_extension in ['jpg', 'png']:
            # Read image file and make prediction
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            predicted_label = predict_sign(image)
            st.image(image, caption='Sign Language Gesture')
            st.write('Predicted Label:', predicted_label)
else:
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        st.error("Error: Unable to access the camera. Please ensure that the camera is connected and accessible.")
        st.stop()  # Stop execution of the app

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to capture frame from the camera.")
            break

        # Make prediction
        predicted_label = predict_sign(frame)
        if isinstance(predicted_label, list):
            for label in predicted_label:
                st.write('Predicted Label:', label)
        else:
            st.write('Predicted Label:', predicted_label)
        st.image(frame, caption='Sign Language Gesture')

        # Break the loop if 'Quit' button is pressed
        if st.button('Quit'):
            break

    # Release the webcam and close the application
    cap.release()
