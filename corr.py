import cv2
import numpy as np
import serial
import time
import threading
from ultralytics import YOLO

# ---------------- CONFIG ----------------
IMG_SIZE = 224   
SERIAL_PORT = '/dev/ttyS0' # Use /dev/ttyS0 for Pi 4 GPIO Pins 14/15

# ---------------- SERIAL ----------------
try:
    ser = serial.Serial(SERIAL_PORT, 115200, timeout=0.01)
    time.sleep(2)
except Exception as e:
    print(f"Serial Error: {e}")
    ser = None

def send(cmd):
    if ser:
        ser.write((cmd + '\n').encode())

# ---------------- YOLO ----------------
# Use yolov8n.pt - Ensure you have exported it to .tflite for best Pi 4 speed
model = YOLO("yolov8n.pt") 
TARGET_CLASSES = [0, 2, 3, 5, 7] # person, car, motor, bus, truck

# ---------------- SHARED VARIABLES ----------------
frame = None
decision = "S"
lock = threading.Lock()

# ---------------- PID ----------------
Kp, Kd = 0.6, 0.3
prev_error = 0

def pid_control(error):
    global prev_error
    derivative = error - prev_error
    output = (Kp * error) + (Kd * derivative)
    prev_error = error
    return output

# ---------------- LANE / MAZE DETECTION ----------------
def lane_detection(img):
    if img is None: return 0
    # Resize for speed
    small_img = cv2.resize(img, (160, 120))
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding for maze paths (black lines on light surface)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    
    h, w = thresh.shape
    roi = thresh[int(h*0.5):h, :] # Look at bottom half

    M = cv2.moments(roi)
    if M['m00'] == 0:
        return 0 # No path found

    cx = int(M['m10'] / M['m00'])
    return cx - (w // 2)

# ---------------- THREAD 1: CAMERA ----------------
def camera_thread():
    global frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, img = cap.read()
        if not ret: continue
        with lock:
            frame = img

# ---------------- THREAD 2: VISION & AI ----------------
def vision_thread():
    global frame, decision
    while True:
        if frame is None:
            time.sleep(0.01)
            continue

        with lock:
            img_input = frame.copy()

        # 1. YOLO Detection (Obstacles)
        results = model(img_input, imgsz=IMG_SIZE, conf=0.4, verbose=False)
        
        min_dist = 999
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in TARGET_CLASSES:
                    # Calculate normalized area for distance
                    x1, y1, x2, y2 = box.xyxy[0]
                    area = (x2 - x1) * (y2 - y1)
                    dist = 10000 / float(area) if area > 0 else 999
                    if dist < min_dist: min_dist = dist

        # 2. Maze Steering (Lane Detection)
        error = lane_detection(img_input)
        steering = pid_control(error)

        # 3. Decision Logic
        if min_dist < 1.8:
            new_decision = "S" # EMERGENCY STOP
        elif min_dist < 4.0:
            new_decision = "D" # DANGER - SLOW DOWN (Handle in ESP32)
        else:
            if steering < -15: new_decision = "L"
            elif steering > 15: new_decision = "R"
            else: new_decision = "F"

        with lock:
            decision = new_decision

# ---------------- THREAD 3: CONTROL ----------------
def control_thread():
    global decision
    last_cmd = ""
    while True:
        with lock:
            cmd = decision
        
        if cmd != last_cmd:
            send(cmd)
            last_cmd = cmd
            print(f"Executing: {cmd}")
        
        time.sleep(0.05) # Frequency of command updates

# ---------------- START ----------------
t1 = threading.Thread(target=camera_thread, daemon=True)
t2 = threading.Thread(target=vision_thread, daemon=True)
t3 = threading.Thread(target=control_thread, daemon=True)

t1.start()
t2.start()
t3.start()

print("ADAS + Maze Solver System Online")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    send("S")
    print("System Shutdown")