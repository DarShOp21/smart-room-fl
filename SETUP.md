# 🧠 Smart Room Federated Trust Evaluation System

## 📌 Project Setup Guide

This document provides a **complete step-by-step setup** for running the project, including:

* Federated Learning Server
* Raspberry Pi Clients
* Cloud Monitoring Server
* Web Sensor Simulator
* Dataset Generation

---

# 📁 Project Structure

```
smart-room-fl/
│
├── fl_client/           # Raspberry Pi client (local training)
├── fl_server/           # Federated server (Jetson/Laptop)
├── cloud/               # Cloud monitoring server
├── web_api/             # Prediction API
├── web_app/             # Frontend UI
├── data/                # Generated datasets
├── requirements.txt
```

---

# ⚙️ 1. Prerequisites

## System Requirements

* Python 3.10–3.11 (⚠ avoid 3.12+ on Pi)
* pip
* Git

## Hardware (Optional)

* Raspberry Pi (clients)
* Jetson Nano / Laptop (server)

---

# 🧪 2. Clone Repository

```bash
git clone <your-repo-url>
cd smart-room-fl
```

---

# 🐍 3. Create Virtual Environment

### Windows

```bash
python -m venv fl_env
fl_env\Scripts\activate
```

### Linux / Raspberry Pi

```bash
python3 -m venv fl_env
source fl_env/bin/activate
```

---

# 📦 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If missing, install manually:

```bash
pip install flwr torch numpy fastapi uvicorn requests
```

for pi
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

for fog node(jetson nano)
```
pip install tokenizers==0.20.3
pip install transformers==4.35.0
```

---

# 📊 5. Dataset Generation

Run:

```bash
python fl_client/synthetic/generator.py
```

This will:

* Generate balanced dataset
* Save CSV in:

```
data/smart_room_balanced_2000.csv
```

---

# 🧠 6. Start Federated Learning System

## 🔹 Step 1: Start Server

```bash
python -m fl_server.server
```

---

## 🔹 Step 2: Start Client(s)

```bash
python -m fl_client.client
```

You can run multiple clients (simulate multiple rooms).

---

# ☁️ 7. Start Cloud Server

```bash
python -m uvicorn cloud.cloud_server:app --host 0.0.0.0 --port 9000
```

Open:

```
http://localhost:9000/docs
```

Endpoints:

* `/models`
* `/trust`

---

# 🔮 8. Start Prediction API

```bash
python -m uvicorn web_api.api:app --port 8500
```

Test:

```
http://localhost:8500/docs
```

---

# 🌐 9. Run Web App (Sensor Simulator)

Start a local server:

```bash
cd web_app
python -m http.server 8000
```

Open:

```
http://localhost:8000/index.html
```

---

# 🔁 10. Full System Flow

```
Web App (Simulated Sensors)
        ↓
Prediction API
        ↓
AutoEncoder Model
        ↓
Trust Score
        ↓
Federated Server
        ↓
Cloud Monitoring
```

---

# 📈 11. Trust Score Meaning

| Trust Score | Status         |
| ----------- | -------------- |
| 0.8 – 1.0   | Normal         |
| 0.5 – 0.8   | Slight Anomaly |
| 0.2 – 0.5   | Suspicious     |
| < 0.2       | High Risk      |

---

# 🧪 12. Testing the System

Click:

```
Generate Sensor Data
```

Expected output:

```
Trust Score: 0.72
Status: NORMAL
```

---

# ⚠️ Common Issues & Fixes

## ❌ CORS Error

Fix in `web_api/api.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## ❌ Trust Score Always 0

Fix:

* Use correct trust function
* Use realistic input data (not pure random)

---

## ❌ flwr not found

```bash
pip install flwr
```

---

## ❌ NumPy JSON error

Fix in `cloud_client.py`:

```python
parameters = [p.tolist() for p in parameters]
```

---

# 🚀 13. Key Features Implemented

* Federated Learning (FLWR)
* Trust-Weighted Aggregation
* AutoEncoder-based anomaly detection
* Cloud monitoring system
* Web-based sensor simulation
* CSV dataset generation
* Edge–Fog–Cloud architecture

---

# 🏁 Final Notes

This project demonstrates:

* Privacy-preserving ML
* Smart-room anomaly detection
* Federated trust evaluation
* Real-time monitoring system

---

# 📌 Optional Improvements

* Save/load trained model (`.pt`)
* Real sensor integration (PIR, door sensor)
* Live dashboard with charts
* Multi-client simulation

---

# 👨‍💻 Author

Smart Room Federated Trust Evaluation Project
