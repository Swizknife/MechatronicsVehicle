
import cv2
import time
from ultralytics import YOLO

# ---------------- CONFIG ----------------
TARGET_CLASSES = [0, 2, 3, 5, 7] # person, car, motorcycle, bus, truck
IMG_SIZE = 160

def estimate_distance(h):
    if h == 0:
        return 999
    return 500 / h 

# ---------------- YOLO MODEL ----------------
print("Loading YOLO model...")
# Using .pt so Ultralytics automatically downloads it if missing!
model = YOLO("yolov8n.pt") 

# ---------------- CAMERA ----------------
print("Starting USB Camera...")
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
cap.set(cv2.CAP_PROP_FPS, 30)

time.sleep(2) # Camera warm-up

print("\n--- Starting Inference Loop (Terminal Only) ---")
print("Press Ctrl+C to stop the script.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # -------- RUN YOLO --------
        # verbose=False prevents Ultralytics from printing its own debug spam
        results = model(frame, imgsz=IMG_SIZE, classes=TARGET_CLASSES, conf=0.5, verbose=False)
        
        min_dist = 999
        found_targets = False

        # -------- PARSE & PRINT --------
        for result in results:
            for box in result.boxes:
                found_targets = True
                
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                h = float(box.xywh[0][3]) 
                
                dist = estimate_distance(h)
                if dist < min_dist:
                    min_dist = dist

                # Print individual target data
                print(f"[DETECTED] Class {cls} | Conf: {conf:.2f} | Height: {h:.1f}px | Dist: {dist:.1f}")

        # Print the final decision variable for this frame
        if found_targets:
            print(f"---> [SYSTEM UPDATE] Closest Target Distance: {min_dist:.1f}\n")
        
        # Simulate frame pacing
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nKeyboard interrupt received...")
finally:
    cap.release()
    print("Camera safely released. Exiting.")
####################
# import cv2
# import time
# from ultralytics import YOLO

# # ---------------- SERIAL (COMMENTED OUT) ----------------
# # import serial
# # try:
# #     ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
# #     time.sleep(2)
# # except Exception as e:
# #     ser = None
# #     print(f"Serial not connected: {e}")
# #
# # def send(cmd):
# #     try:
# #         if ser:
# #             ser.write((cmd + '\n').encode())
# #     except:
# #         print("Serial error")
# # --------------------------------------------------------

# # ---------------- CONFIG ----------------
# TARGET_CLASSES = [0, 2, 3, 5, 7] # person, car, motorcycle, bus, truck
# IMG_SIZE = 160

# def estimate_distance(h):
#     if h == 0:
#         return 999
#     # The constant '500' might need tuning depending on the camera's FOV 
#     # and the new pixel heights from Ultralytics
#     return 500 / h 

# # ---------------- YOLO MODEL ----------------
# print("Loading YOLO model...")
# # Ultralytics can run your existing .tflite model directly. 
# # If it fails, you can swap this to "yolov8n.pt" to use the standard PyTorch model.
# # model = YOLO("yolov8n.tflite", task='detect') 
# model = YOLO("yolov8n.pt")

# # ---------------- CAMERA ----------------
# print("Starting USB Camera...")
# cap = cv2.VideoCapture(0)

# # Set resolution and FPS
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
# cap.set(cv2.CAP_PROP_FPS, 30)

# # Warm up camera
# time.sleep(2)

# print("Starting Inference Loop. Press Ctrl+C to stop.")

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame. Retrying...")
#             time.sleep(0.1)
#             continue

#         # -------- RUN YOLO --------
#         # Ultralytics natively handles filtering by class and confidence threshold.
#         # Setting verbose=False stops it from printing spam on every frame.
#         results = model(frame, imgsz=IMG_SIZE, classes=TARGET_CLASSES, conf=0.5, verbose=False)
        
#         min_dist = 999
#         found_targets = False

#         # Parse results
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 found_targets = True
                
#                 # Extract data from the Ultralytics box object
#                 cls = int(box.cls[0])
#                 conf = float(box.conf[0])
                
#                 # xywh gets [x_center, y_center, width, height] in pixels
#                 h = float(box.xywh[0][3]) 
                
#                 # Estimate distance
#                 dist = estimate_distance(h)
#                 if dist < min_dist:
#                     min_dist = dist

#                 print(f"Found Class: {cls} | Conf: {conf:.2f} | Box Height: {h:.1f}px | Dist: {dist:.2f}")

#         if found_targets:
#             print(f"---> MINIMUM DISTANCE THIS FRAME: {min_dist:.2f}\n")

#         # Sleep briefly to simulate the frame skipping from your original code
#         time.sleep(0.05)

# except KeyboardInterrupt:
#     print("\f test...")
# finally:
#     cap.release()
#     print("Camera released.")