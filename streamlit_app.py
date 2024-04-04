import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the pre-trained model
model = load_model('sign_language_model.h5')
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

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

# WebRTC video transformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.is_camera_on = False

    def transform(self, frame):
        if self.is_camera_on:
            img = frame.to_ndarray(format="bgr24")
            predicted_label = predict_sign(img)
            st.write('Predicted Label:', predicted_label)
            st.image(img, caption='Sign Language Gesture')
        else:
            st.write('Camera is turned off.')

# Button to toggle the camera on and off
start_stop_button = st.button("Start/Stop Camera")

# WebRTC streamer
webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Turn the camera on or off based on the button click
if start_stop_button:
    if webrtc_ctx.video_transformer.is_camera_on:
        webrtc_ctx.stop()
    else:
        webrtc_ctx.start()
