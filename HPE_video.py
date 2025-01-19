import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load the video file (replace "dance.mp4" with the name of your video file)
video_source = "dance.mp4"
cap = cv2.VideoCapture(video_source)

# Check if the video file is loaded successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize MediaPipe Pose
with mp_pose.Pose(static_image_mode=False, 
                  model_complexity=1,  # Adjust model complexity for better accuracy (0, 1, or 2)
                  enable_segmentation=False, 
                  min_detection_confidence=0.5, 
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()  # Read frames from the video
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

        # Display the processed frame
        cv2.imshow("Pose Detection on Video", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

