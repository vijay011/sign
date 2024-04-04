import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings

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

# Custom video processor class to make predictions on each frame
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.predicted_label = None

    def recv(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:  # If it's an image
            self.predicted_label = predict_sign(frame)
        elif frame.ndim == 4:  # If it's a video
            predicted_labels = []
            for image in frame:
                predicted_labels.append(predict_sign(image))
            self.predicted_label = predicted_labels
        return frame

# Streamlit app layout
st.title('Sign Language Recognition')

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
            st.video(video_nparray)
            predicted_labels = predict_sign(video_nparray)
            st.write('Predicted Labels:', predicted_labels)
        elif file_extension in ['jpg', 'png']:
            # Read image file and make prediction
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            st.image(image, caption='Sign Language Gesture')
            predicted_label = predict_sign(image)
            st.write('Predicted Label:', predicted_label)
else:
    # Display the video stream and predictions using streamlit-webrtc
    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=SignLanguageProcessor, client_settings=ClientSettings(fps=5))

    if webrtc_ctx.video_processor:
        if isinstance(webrtc_ctx.video_processor.predicted_label, list):
            for label in webrtc_ctx.video_processor.predicted_label:
                st.write('Predicted Label:', label)
        else:
            st.write('Predicted Label:', webrtc_ctx.video_processor.predicted_label)
