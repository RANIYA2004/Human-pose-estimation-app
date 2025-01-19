import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set the video source: Use 0 for the default webcam or provide a video file path
video_source = 0  # Replace with "dance.mp4" for a video file
cap = cv2.VideoCapture(video_source)

# Set video frame dimensions (optional, applies if the camera supports it)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe Pose
with mp_pose.Pose(static_image_mode=False, 
                  model_complexity=1,  # Model complexity: 0, 1, or 2
                  enable_segmentation=False, 
                  min_detection_confidence=0.5, 
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video source
        if not ret:
            print("Video capture ended.")
            break

        # Convert the frame from BGR to RGB (MediaPipe requires RGB input)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect poses
        results = pose.process(frame_rgb)

        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

        # Display the frame
        cv2.imshow("Pose Detection", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
