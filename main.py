import cv2
import torch
from ultralytics import YOLO
import logging
import os

os.environ["ULTRALYTICS_LOGGING"] = "False"
logging.getLogger("ultralytics").setLevel(logging.WARNING)

model = YOLO('bottle.pt')
model.to(torch.device('cpu'))

source = "001.png"
is_image = source.lower().endswith(('.png', '.jpg', '.jpeg'))

# ตัวแปรสะสมขวดทั้งหมด (global)
total_bottles = 0
last_frame_count = 0  # จำนวนขวดที่เจอในเฟรมก่อนหน้า

def detect_and_draw(frame):
    global total_bottles, last_frame_count
    results = model(frame, imgsz=640, conf=0.5, device='cpu', verbose=False)[0]

    current_frame_count = 0
    for idx, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label.lower() == 'bottle':
            current_frame_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{current_frame_count}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ถ้าขวดในเฟรมนี้มากกว่าครั้งก่อน (อาจมีขวดใหม่) เพิ่มส่วนต่างไป total_bottles
    if current_frame_count > last_frame_count:
        total_bottles += (current_frame_count - last_frame_count)
    last_frame_count = current_frame_count

    # แสดงจำนวนสะสมรวม
    cv2.putText(frame, f'Total Bottles (accumulated): {total_bottles}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
if is_image:
    img = cv2.imread(source)
    img = detect_and_draw(img)
    cv2.imshow("Bottle Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

else:
    cap = cv2.VideoCapture(0 if source == "0" else source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_and_draw(frame)
        cv2.imshow("Bottle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
