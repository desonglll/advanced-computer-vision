import cv2
import mediapipe as mp
import time

from mediapipe_impl.pose_estimation.PoseEstimationModule import PoseDetector

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Create VideoCapture objects for two different video sources
cap1 = cv2.VideoCapture(0)  # First camera
cap2 = cv2.VideoCapture(2)  # Second camera
# Suppose Cap1 is the front
# Suppose Cap2 is the right sider

detector = PoseDetector()

p_time = 0
while True:
    # Read frames from both cameras
    success1, img1 = cap1.read()
    success2, img2 = cap2.read()

    if not success1 or not success2:
        print("Error: Unable to read from one of the cameras")
        break

    # Process the first camera frame
    img_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    results1 = pose.process(img_rgb1)

    if results1.pose_landmarks:
        mp_draw.draw_landmarks(img1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results1.pose_landmarks.landmark):
            h, w, c = img1.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img1, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    # Process the second camera frame
    img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    results2 = pose.process(img_rgb2)

    if results2.pose_landmarks:
        mp_draw.draw_landmarks(img2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results2.pose_landmarks.landmark):
            h, w, c = img2.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img2, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    # Calculate FPS for both cameras
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    # Display FPS on both images
    cv2.putText(img1, 'FPS: {:.2f}'.format(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img2, 'FPS: {:.2f}'.format(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Show the frames for both cameras
    cv2.imshow('Camera 1', img1)
    cv2.imshow('Camera 2', img2)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video captures and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
