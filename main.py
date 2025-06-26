import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments

parser = argparse.ArgumentParser()
# --model is now fixed in the code
parser.add_argument('--source', help='index of USB camera ("usb0"), or index of Picamera ("picamera0")',
                     required=True)
# --thresh is now fixed in the code
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                     otherwise, match source resolution',
                     default=None)
# --record is now fixed in the code

args = parser.parse_args()


# Parse user inputs
# กำหนดค่าคงที่สำหรับ Path ของไฟล์โมเดล YOLO (คุณสามารถเปลี่ยนเป็น Path โมเดลของคุณได้)
model_path = "50epochs.pt" 
img_source = args.source
# กำหนดค่าคงที่สำหรับ Confidence threshold ขั้นต่ำในการแสดงผลวัตถุที่ตรวจจับได้
min_thresh = 0.5 
# กำหนดค่าคงที่ว่าจะบันทึกผลลัพธ์หรือไม่ (ตั้งค่าเป็น True หากต้องการบันทึก)
record = False 

# ตรวจสอบว่าไฟล์โมเดลมีอยู่และถูกต้องหรือไม่
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# โหลดโมเดลเข้าสู่หน่วยความจำและรับชื่อคลาส
model = YOLO(model_path, task='detect')
model.to('cpu') 
labels = model.names # ดึงชื่อคลาสทั้งหมดที่โมเดลสามารถตรวจจับได้


# แยกวิเคราะห์อินพุตเพื่อระบุว่าแหล่งที่มาของภาพเป็นไฟล์, โฟลเดอร์, วิดีโอ หรือกล้อง USB
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# แยกวิเคราะห์ความละเอียดการแสดงผลที่ผู้ใช้ระบุ
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# ตรวจสอบว่าการบันทึกถูกต้องหรือไม่ และตั้งค่าการบันทึก
if record: # บล็อก 'if' นี้จะทำงานเฉพาะเมื่อ 'record' เป็น True เท่านั้น (ปัจจุบันถูกกำหนดเป็น False)
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # ตั้งค่าการบันทึก
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# โหลดหรือเริ่มต้นแหล่งที่มาของภาพ
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # ตั้งค่าความละเอียดกล้องหรือวิดีโอหากผู้ใช้ระบุ
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# กำหนดสีของ Bounding Box (ใช้โทนสี Tableau 10)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# เริ่มต้นตัวแปรควบคุมและสถานะ
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
final_bottle_count = 0 # เพิ่มตัวแปรเพื่อเก็บค่านับขวดสุดท้าย

# เริ่มลูปการอนุมาน
while True:

    t_start = time.perf_counter()

    # โหลดเฟรมจากแหล่งที่มาของภาพ
    if source_type == 'image' or source_type == 'folder': # หากแหล่งที่มาเป็นภาพหรือโฟลเดอร์ภาพ, โหลดภาพจากชื่อไฟล์
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': # หากแหล่งที่มาเป็นวิดีโอ, โหลดเฟรมถัดไปจากไฟล์วิดีโอ
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb': # หากแหล่งที่มาเป็นกล้อง USB, ดึงเฟรมจากกล้อง
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera': # หากแหล่งที่มาเป็น Picamera, ดึงเฟรมโดยใช้อินเทอร์เฟซ picamera
        frame = cap.capture_array()
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # ปรับขนาดเฟรมตามความละเอียดที่ต้องการ
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # รันการอนุมานบนเฟรม
    results = model(frame, verbose=False)

    # ดึงผลลัพธ์
    detections = results[0].boxes

    # เริ่มต้นตัวแปรสำหรับนับจำนวนขวด
    bottle_count = 0

    # วนลูปผ่านแต่ละ Detection เพื่อรับพิกัด bbox, ความเชื่อมั่น, และคลาส
    for i in range(len(detections)):

        # รับพิกัด bounding box
        # Ultralytics จะคืนค่าผลลัพธ์ในรูปแบบ Tensor ซึ่งต้องแปลงเป็นอาร์เรย์ Python ปกติ
        xyxy_tensor = detections[i].xyxy.cpu() # Detections ในรูปแบบ Tensor ในหน่วยความจำ CPU
        xyxy = xyxy_tensor.numpy().squeeze() # แปลง tensors เป็น Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # แยกพิกัดแต่ละส่วนและแปลงเป็น integer

        # รับ ID คลาสและชื่อคลาสของ bounding box
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # รับค่าความเชื่อมั่นของ bounding box
        conf = detections[i].conf.item()

        # วาดกล่อง (box) หากค่าความเชื่อมั่นสูงพอ
        if conf > min_thresh: # ใช้ค่า min_thresh ที่กำหนดไว้
            # ตรวจสอบว่าเป็นคลาส 'bottle' หรือไม่
            # โปรดทราบ: 'bottle' คือชื่อคลาสเริ่มต้นในโมเดล YOLOv8n หากโมเดลของคุณมีชื่อคลาสอื่นสำหรับขวด
            # คุณจะต้องเปลี่ยน 'bottle' ให้ตรงกับชื่อคลาสใน model.names ของคุณ
            if classname == 'bottle': 
                bottle_count = bottle_count + 1 # เพิ่มจำนวนขวด

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # รับขนาดตัวอักษร
            label_ymin = max(ymin, labelSize[1] + 10) # ตรวจสอบให้แน่ใจว่าไม่วาด label ใกล้ขอบบนของหน้าต่างมากเกินไป
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # วาดกล่องสีขาวเพื่อใส่ข้อความ label
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # วาดข้อความ label
            
    # คำนวณและวาดอัตราเฟรม (FPS) (หากใช้แหล่งที่มาเป็นวิดีโอ, USB หรือ Picamera)
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # วาดอัตราเฟรม
    
    # แสดงผลการตรวจจับ - จำนวนขวด
    cv2.putText(frame, f'Number of bottles: {bottle_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # วาดจำนวนขวดที่ตรวจจับได้ทั้งหมด
    cv2.imshow('YOLO detection results',frame) # แสดงภาพ
    if record: recorder.write(frame) # จะทำงานเฉพาะเมื่อ 'record' เป็น True

    # เมื่อสิ้นสุดแต่ละเฟรม ให้เก็บค่านับขวดล่าสุดไว้ใน final_bottle_count
    final_bottle_count = bottle_count 

    # หากอนุมานบนภาพเดี่ยว, ให้รอการกดปุ่มของผู้ใช้ก่อนย้ายไปภาพถัดไป. มิฉะนั้น, ให้รอ 5ms ก่อนย้ายไปเฟรมถัดไป
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # กด 'q' เพื่อออก
        break
    elif key == ord('s') or key == ord('S'): # กด 's' เพื่อหยุดการอนุมานชั่วคราว
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # กด 'p' เพื่อบันทึกภาพผลลัพธ์บนเฟรมนี้
        cv2.imwrite('capture.png',frame)
    
    # คำนวณ FPS สำหรับเฟรมนี้
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # เพิ่มผลลัพธ์ FPS ลงใน frame_rate_buffer (สำหรับหา FPS เฉลี่ยจากหลายเฟรม)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # คำนวณ FPS เฉลี่ยสำหรับเฟรมที่ผ่านมา
    avg_frame_rate = np.mean(frame_rate_buffer)


# ทำความสะอาด
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release() # จะทำงานเฉพาะเมื่อ 'record' เป็น True
cv2.destroyAllWindows()

# บันทึกจำนวนขวดสุดท้ายลงในไฟล์ count.txt
try:
    with open('count.txt', 'w') as f:
        f.write(f'Final bottle count: {final_bottle_count}\n')
    print(f'Final bottle count ({final_bottle_count}) saved to count.txt')
except IOError as e:
    print(f"Error saving count to file: {e}")
