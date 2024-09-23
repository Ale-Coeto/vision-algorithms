import mediapipe as mp
import cv2

'''
Script to get the landmarks of the pose model 
and draw them on the image
'''

pose_model = mp.solutions.pose.Pose(min_detection_confidence=0.6) 
mp_drawing = mp.solutions.drawing_utils

# Load image
image = cv2.imread('../Images/1.jpg')

# Get the results of the pose model
results = pose_model.process(image)

# Get the landmarks
if results.pose_landmarks is not None:
    landmarks = results.pose_landmarks.landmark
    
    # Draw the landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    # Convert the image back to BGR
    

# Show the image
cv2.imshow('image', image)
cv2.waitKey(0)
