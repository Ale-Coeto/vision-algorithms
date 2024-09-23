# Mediapipe
This tool is used for pose estimation using landmarks from the body of a person.

## Guide
### Load model

```python
import mediapipe as mp

pose_model = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
```

### Process image

```python
# Read image
image = cv2.imread("image.jpg")

# Get Landmarks
results = pose_model.process(image)

# Get Key points
if results.pose_landmarks is not None:
    landmarks = results.pose_landmarks.landmark

    # Draw landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    # Left leg points
    hip_left = landmarks[23]
    knee_left = landmarks[25]
    ankle_left = landmarks[27]

    # Right leg points
    hip_right = landmarks[24]
    knee_right = landmarks[26]  
    ankle_right = landmarks[28]

    # Left arm points
    shoulder_left = landmarks[11]
    elbow_left = landmarks[13]
    wrist_left = landmarks[15]
    index_left = landmarks[19]

    # Right arm points
    shoulder_right = landmarks[12]
    elbow_right = landmarks[14]
    wrist_right = landmarks[16]
    index_right = landmarks[20]

```

## Useful functions

### Get angles function 

```python
import numpy as np
from math import acos, degrees

# Get angles
def getAngle(point_close, point_mid, point_far):
    # Convert the points to numpy arrays
    p1 = np.array([point_close.x, point_close.y])
    p2 = np.array([point_mid.x, point_mid.y])
    p3 = np.array([point_far.x, point_far.y])

    # Euclidean distances
    l1 = np.linalg.norm(p2 - p3)    
    l2 = np.linalg.norm(p1 - p3)
    l3 = np.linalg.norm(p1 - p2)

    # Calculate the angle
    return abs(degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3))))
```

### Uses
```python
leg_angle_left = getAngle(hip_left, knee_left, ankle_left)
leg_angle_right = getAngle(hip_right, knee_right, ankle_right)
elbow_angle_left = getAngle(shoulder_left, elbow_left, wrist_left)
shoulder_angle_left = getAngle(shoulder_right, shoulder_left, elbow_left)
elbow_angle_right = getAngle(shoulder_right, elbow_right, wrist_right)
shoulder_angle_right = getAngle(shoulder_left, shoulder_right, elbow_right)
```

## Center of a person (chest)
```python
def getCenterPerson(poseModel, image):
    # Process the image
    results = poseModel.process(image)

    # Get the landmarks
    if results.pose_landmarks is not None:
        landmarks = results.pose_landmarks.landmark
        shoulder_right = landmarks[12]
        shoulder_left = landmarks[11]

        x_center = (shoulder_right.x + shoulder_left.x) / 2
        y_center = (shoulder_right.y + shoulder_left.y) / 2
        
        cv2.circle(image, (int(x_center * image.shape[1]), int(y_center * image.shape[0])), 5, (0, 0, 255), -1)
        
        return x_center*image.shape[1], y_center*image.shape[0]
    
    return None, None
```

## Examples
Check basic example: 
- [Pose estimation](Mediapipe.py)

## References
- [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)