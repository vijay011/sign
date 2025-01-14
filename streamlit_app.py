import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings

# Load the pre-trained model
model = load_model('sign_language_model.h5')
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Preprocess image for the model
def preprocess_image(image):
    resized_image = cv2.resize(image, (200, 200)) / 255.0
    return resized_image

# Predict sign from an image
def predict_sign(image):
    try:
        preprocessed = preprocess_image(image)
        predictions = model.predict(np.expand_dims(preprocessed, axis=0))
        predicted_class = np.argmax(predictions)
        return class_labels[predicted_class]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error"

# Video processor for real-time predictions
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.predicted_label = "Waiting for input..."

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        self.predicted_label = predict_sign(image)
        return frame

# Streamlit app
st.title("Sign Language Recognition")

# Option to upload pre-captured data
use_sample_data = st.checkbox("Use Pre-captured Sample Data")

if use_sample_data:
    uploaded_file = st.file_uploader("Upload Sample Data (Image or Video)", type=['jpg', 'png', 'mp4', 'avi'])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension in ['jpg', 'png']:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            st.image(image, caption="Uploaded Gesture", use_column_width=True)
            predicted_label = predict_sign(image)
            st.write(f"Predicted Label: {predicted_label}")
        elif file_extension in ['mp4', 'avi']:
            # Decode video file and process frames
            temp_file = f"temp_video.{file_extension}"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp_file)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count > 10:  # Process up to 10 frames for simplicity
                    break
                st.image(frame, caption=f"Frame {frame_count}", use_column_width=True)
                st.write(f"Predicted Label: {predict_sign(frame)}")
                frame_count += 1
            cap.release()
else:
    # Real-time video stream
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=SignLanguageProcessor,
        client_settings=ClientSettings(fps=5),
    )
    if webrtc_ctx.video_processor:
        st.write(f"Predicted Label: {webrtc_ctx.video_processor.predicted_label}")

