import cv2
import time
from ultralytics import YOLO

# ---------------- SERIAL (COMMENTED OUT) ----------------
# import serial
# try:
#     ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
#     time.sleep(2)
# except Exception as e:
#     ser = None
#     print(f"Serial not connected: {e}")
#
# def send(cmd):
#     try:
#         if ser:
#             ser.write((cmd + '\n').encode())
#     except:
#         print("Serial error")
# --------------------------------------------------------

# ---------------- CONFIG ----------------
TARGET_CLASSES = [0, 2, 3, 5, 7] # person, car, motorcycle, bus, truck
IMG_SIZE = 160

def estimate_distance(h):
    if h == 0:
        return 999
    # The constant '500' might need tuning depending on the camera's FOV 
    # and the new pixel heights from Ultralytics
    return 500 / h 

# ---------------- YOLO MODEL ----------------
print("Loading YOLO model...")
# Ultralytics can run your existing .tflite model directly. 
# If it fails, you can swap this to "yolov8n.pt" to use the standard PyTorch model.
# model = YOLO("yolov8n.tflite", task='detect') 
model = YOLO("yolov8n.pt")

# ---------------- CAMERA ----------------
print("Starting USB Camera...")
cap = cv2.VideoCapture(0)

# Set resolution and FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_FPS, 30)

# Warm up camera
time.sleep(2)

print("Starting Inference Loop. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
            time.sleep(0.1)
            continue

        # -------- RUN YOLO --------
        results = model(frame, imgsz=IMG_SIZE, classes=TARGET_CLASSES, conf=0.5, verbose=False)
        
        min_dist = 999
        found_targets = False

        # Parse results and draw them
        for result in results:
            boxes = result.boxes
            for box in boxes:
                found_targets = True
                
                # Extract data for logic
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                h = float(box.xywh[0][3]) 
                
                # Extract Top-Left / Bottom-Right coordinates to draw the box (xyxy format)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Estimate distance
                dist = estimate_distance(h)
                if dist < min_dist:
                    min_dist = dist

                # --- DRAW THE VISUALS ---
                label = f"Class {cls} | Dist: {dist:.1f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                print(f"Found Class: {cls} | Conf: {conf:.2f} | Box Height: {h:.1f}px | Dist: {dist:.2f}")

        if found_targets:
            print(f"---> MINIMUM DISTANCE THIS FRAME: {min_dist:.2f}\n")

        # --- SHOW THE FRAME ---
        cv2.imshow("YOLO Camera View", frame)

        # OpenCV needs waitKey(1) to refresh the window. Press 'q' to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping test...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released.")