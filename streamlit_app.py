import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Load the pre-trained model
model = load_model('sign_language_model.h5')
class_labels = ['A', 'B', 'C', ...]  # Your class labels

# Streamlit app layout
st.title('Sign Language Recognition (WebRTC)')

# WebRTC function for capturing frames
def webrtc_app():
    ctx = webrtc_streamer(key="streamer", video_processor_factory=VideoProcessor)
    return ctx.video_processor  # Return the video processor for further processing

# Video processor class for handling frames
class VideoProcessor:
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")

# Function to make predictions (using the frame from WebRTC)
def predict_sign():
    processor = webrtc_app()  # Get the video processor
    while True:
        if processor and processor.frame is not None:  # Check if processor exists and frame is available
            frame = processor.frame
            predicted_label = predict_label(frame)  # Call your prediction function
            st.image(frame, caption='Sign Language Gesture')
            st.write('Predicted Label:', predicted_label)
            # You can also display other information or process the frame further
        else:
            st.write("Waiting for frames...")  # Display a message if no frame is available
            st.stop()  # Stop execution until frame is available

# Main function to run the app
def main():
    # Call the WebRTC function to start capturing frames
    predict_sign()

# Run the main function
if __name__ == "__main__":
    main()
