import cv2
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the pre-trained model
model = load_model('sign_language_model.h5')
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Function to preprocess and resize images
def preprocess_image(image):
    resized_image = cv2.resize(image, (200, 200)) / 255.0
    return resized_image

# Function to make predictions
def predict_sign(frame):
    frame = preprocess_image(frame)
    predictions = model.predict(np.expand_dims(frame, axis=0))
    predicted_class = np.argmax(predictions)
    
    # Print predicted class index for debugging
    print("Predicted Class Index:", predicted_class)
    
    if predicted_class < len(class_labels):
        return class_labels[predicted_class]
    else:
        return "Unknown Class"  # Handle out-of-range predictions

# Custom video transformer class to make predictions on each frame
class SignLanguageTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        predicted_label = predict_sign(image)
        st.image(image, caption='Sign Language Gesture')
        st.write('Predicted Label:', predicted_label)
        return image

# Streamlit app layout
st.title('Sign Language Recognition (WebRTC)')

# Option to start/stop the camera
start_camera = st.button('Start Camera')
stop_camera = st.button('Stop Camera')

if start_camera:
    webrtc_streamer(key="example", video_transformer_factory=SignLanguageTransformer)
elif stop_camera:
    st.stop()  # Stop execution of the app
