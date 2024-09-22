from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# Load video or use webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('path_to_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get results of each frame
    results = model(frame)

    # Get results[0]
    boxes = results[0].boxes

    # Check each detection
    for out in results:
        for box in out.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]

            # Get class name
            class_id = box.cls[0].item()
            label = model.names[class_id]
            
            # Get confidence
            prob = round(box.conf[0].item(), 2)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add text
            cv2.putText(frame, f'{label} {prob}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display image
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('0'):
        break
