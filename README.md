# MechatronicsVehicle 🚗🛠️

A Python‑based vehicle perception and mechatronics project that uses YOLO‑based object detection for real‑time vehicle detection. The repository includes various scripts for experimentation and building an intelligent system for robotics/vehicle analysis.

---

## 🔍 Overview

This project integrates **computer vision** with **mechatronics principles** to detect and process vehicle information from video or images using Python. The core object detection is implemented using **YOLO (You Only Look Once)** models, enabling fast and accurate recognition of vehicles such as cars, bikes, and trucks.

If you’re integrating this into a robotic system or autonomous agent, these scripts form the vision foundation for perception tasks (detection, tracking, classification).

---

## 🧠 Features

✔ Real‑time vehicle detection using YOLO models  
✔ Multiple Python scripts for different experiment setups  
✔ Lightweight project structure ideal for mechatronics and robotics integration  
✔ Works with pre‑trained YOLO weights or custom datasets (optional)

---

## 📁 Repository Structure

| File / Folder | Description |
|--------------|-------------|
| `yolo.py` | Main YOLO detection logic (uses a YOLO model) |
| `startpy.py` | Entry point script (example runner) |
| `corr.py` | Correlation or data processing utility |
| `span.py` | Additional helper module |
| `done.py`, `old.py`, `improved1.py` | Example/legacy scripts |
| `requirements.txt` | Python dependencies |
| `yolov8n.pt` | Pre‑trained YOLOv8‑n weights used for detection |

---

## 🚀 Getting Started

### 🧰 Prerequisites

You need Python 3.7+ installed. Then install dependencies:

```bash
git clone https://github.com/Swizknife/MechatronicsVehicle.git
cd MechatronicsVehicle
pip install -r requirements.txt
