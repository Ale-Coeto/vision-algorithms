# YOLO - Object Detection

This tool is used mostly for object detection in images and videos. It is a deep learning algorithm that uses a convolutional neural network to detect objects in images. 

## Guide
1. Import model

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
```

2. Obtain detections

```python   
# Basig usage (use verbose=True to display results or verbose=0 to suppress)
results = model('image.jpg')

# Specify classes (ex: person and chair)
results = model('image.jpg', classes=[0,56])

# Use a tracker (either bytetrack.yaml or botsort.yaml)
results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0)

```

3. Get bboxes and other data from each detection

```python

# Check each detection 
for out in results:
    for box in out.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]

        # Get class name
        class_id = box.cls[0].item()
        label = model.names[class_id]
        
        # Get confidence
        prob = round(box.conf[0].item(), 2)

        # If tracker is used, get id
        track_id = box.id.int().item()
        
        print(f'Class label: {label}, Confidence: {prob}')

        # Draw bounding box
        cv2.rectangle(results_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text
        cv2.putText(results_image, f'{label} {prob}', (x1, y2+22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
```

### Examples
Check basic examples:
- [Image detection](Yolov8-On_Image.py)
- [Video detection](Yolov8-On_Video.py)
- [Tracker (person)](Yolov8-Tracker.py)