from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# Load image
image = '../Images/0.jpg'
results_image = cv2.imread(image)

# Get results of frame detections
results = model(image)


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
        
        print(f'Class label: {label}, Confidence: {prob}')

        # Draw bounding box
        cv2.rectangle(results_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text
        cv2.putText(results_image, f'{label} {prob}', (x1, y2+22), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display image
cv2.imshow('image', results_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
