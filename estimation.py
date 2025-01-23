import streamlit as st
import mediapipe as mp
from PIL import Image
import numpy as np
import cv2

DEMO_IMAGE = 'pose1.jpg'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.title("Human Pose Estimation with MediaPipe")
st.text('Upload an image to estimate the human pose.')

# File uploader for user input or use a demo image
img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    image = np.array(Image.open(DEMO_IMAGE))

# Display the original image
st.subheader('Original Image')
st.image(image, caption="Original Image", use_column_width=True)

# Slider to set minimum detection confidence
detection_confidence = st.slider('Detection Confidence', min_value=0.1, max_value=1.0, value=0.5, step=0.1)

@st.cache
def pose_estimation(image, detection_confidence):
    """Detect human pose using MediaPipe Pose."""
    # Convert the image from RGB to BGR for OpenCV compatibility
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image using MediaPipe Pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=detection_confidence) as pose:
        results = pose.process(image_rgb)

        # Draw landmarks and connections on the image
        annotated_image = image.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            )

    return annotated_image

# Perform pose estimation
output_image = pose_estimation(image, detection_confidence)

# Display the output image with pose estimation
st.subheader('Pose Estimation Output')
st.image(output_image, caption="Pose Estimation", use_column_width=True)

st.markdown("**This app uses MediaPipe Pose for human pose estimation.**")
