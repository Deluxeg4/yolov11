from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO('yolo11n.pt')

# Perform object detection on an image
results = model("test/001.png")  # Predict on an image
results[0].show()  # Display results
