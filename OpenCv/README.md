# OpenCV
OpenCV is a library of programming functions mainly aimed at real-time computer vision. 

## Tools

### Import 
```python
import cv2
```
### Image reading, displaying and writing
```python
# Read image
image = cv2.imread("image.jpg")

# Display image
cv2.imshow("Image", image)

# Write image
cv2.imwrite("output.jpg", image)
```

### Image Modification
```python
# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize image
resized_image = cv2.resize(image, (200, 200))

# Rotate image
rows, cols = image.shape[:2]

# Rotate 90 degrees
rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
```

### Drawing
```python
cv2.rectangle(image, (50, 50), (150, 150), (0, 255, 0), 2)  
cv2.circle(image, (100, 100), 50, (255, 0, 0), 2)          
cv2.putText(image, 'Hello', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
```

### Edge detection
```python
edges = cv2.Canny(image, 100, 200)
```

### Finding contours
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
```

### Video reading
```python
import cv2

# Capture from camera
cap = cv2.VideoCapture(1)

# Capture from video path
video_path = "cp_2.mp4"
cap = cv2.VideoCapture(video_path)

# Analyze frames
while cap.isOpened():

    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        # Do something
        
		    # Display the frame
        cv2.imshow("Frame Name", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # End of video
        break
        
cap.release()
cv2.destroyAllWindows() 
```