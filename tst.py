import cv2
import numpy as np
import serial
import time
import threading
from queue import Queue
import tflite_runtime.interpreter as tflite

# ---------------- CONFIG ----------------
IMG_SIZE = 160
YOLO_SKIP = 3   # run YOLO every 3 frames

# ---------------- SERIAL ----------------
try:
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    time.sleep(2)
except:
    ser = None
    print("Serial not connected")

def send(cmd):
    try:
        if ser:
            ser.write((cmd + '\n').encode())
    except:
        print("Serial error")

# ---------------- CAMERA (USB) ----------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ---------------- TFLITE MODEL ----------------
interpreter = tflite.Interpreter(model_path="yolov8n.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

TARGET_CLASSES = [0, 2, 3, 5, 7]

# ---------------- SHARED ----------------
frame_queue = Queue(maxsize=2)
decision = "S"
lock = threading.Lock()

# ---------------- PID ----------------
Kp = 0.5
Kd = 0.2
prev_error = 0

def pid_control(error):
    global prev_error
    derivative = error - prev_error
    output = Kp * error + Kd * derivative
    prev_error = error
    return max(min(output, 100), -100)

# ---------------- DISTANCE ----------------
def estimate_distance(h):
    if h == 0:
        return 999
    return 500 / h

# ---------------- LANE DETECTION ----------------
def lane_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape
    roi = edges[int(h*0.6):h, :]

    M = cv2.moments(roi)
    if M['m00'] == 0:
        return 0

    cx = int(M['m10'] / M['m00'])
    return cx - (w // 2)

# ---------------- YOLO (TFLITE) ----------------
def run_yolo(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    min_dist = 999

    for det in output_data[0]:
        conf = det[4]
        cls = int(det[5])

        if conf > 0.5 and cls in TARGET_CLASSES:
            h = det[3] * IMG_SIZE
            dist = estimate_distance(h)

            if dist < min_dist:
                min_dist = dist

    return min_dist

# ---------------- THREAD 1: CAMERA ----------------
def camera_thread():
    while True:
        # Drop old frames (reduce lag)
        for _ in range(2):
            cap.grab()

        ret, frame = cap.read()

        if not ret:
            time.sleep(0.005)
            continue

        if not frame_queue.full():
            frame_queue.put(frame)

        time.sleep(0.01)

# ---------------- THREAD 2: VISION ----------------
def vision_thread():
    global decision

    frame_count = 0
    min_dist = 999

    while True:
        if frame_queue.empty():
            # time.sleep(0.005)
            time.sleep(0.008)
            continue

        img = frame_queue.get()
        frame_count += 1

        # -------- LANE EVERY FRAME --------
        error = lane_detection(img)
        steering = pid_control(error)

        if steering < -20:
            direction = "L"
        elif steering > 20:
            direction = "R"
        else:
            direction = "F"

        # -------- YOLO EVERY FEW FRAMES --------
        if frame_count % YOLO_SKIP == 0:
            min_dist = run_yolo(img)

        # -------- DECISION --------
        if min_dist < 1.5:
            new_decision = "S"
        elif min_dist < 3:
            new_decision = "D"
        else:
            new_decision = direction

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
            print("CMD:", cmd)

        time.sleep(0.05)

# ---------------- START ----------------
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

t1 = threading.Thread(target=camera_thread, daemon=True)
t2 = threading.Thread(target=vision_thread, daemon=True)
t3 = threading.Thread(target=control_thread, daemon=True)

t1.start()
t2.start()
t3.start()

print("USB CAMERA ADAS RUNNING (OPTIMIZED)")

while True:
    time.sleep(1)