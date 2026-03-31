# Sensathon 2.0: Cross-Layer Physics-Based IDS for Software-Defined Vehicles 🛡️🚙

A real-time **Cross-Layer Intrusion Detection System** that uses physics-based consistency checking across GPS, IMU, CAN bus, and V2X networks to detect coordinated cyberattacks on modern vehicles.

Edge inference runs on an ESP32 microcontroller (~$5, 0.33ms latency).

---

## 🏗️ Architecture

```text
Python Simulator (Vehicle Physics + Attack Injection)
         |
         | USB Serial (115200 baud)
         v
ESP32 Microcontroller (29-feature RF inference + 3 LEDs)
         |
         | USB Serial (JSON responses)
         v
Streamlit Dashboard (Real-time visualization)
```

## 📂 Repository Structure

```
sensathon/
├── generate_dataset.py      # Synthetic physics-based training data (Kinematic Bicycle Model)
├── train_model.py           # RF training pipeline + ablation study + baselines
├── export_to_c.py           # Model → C header conversion for ESP32
├── quick_demo.py            # Interactive real-time simulation (keyboard + dashboard control)
├── stream_to_esp32.py       # Serial communication layer (scripted demo mode)
├── dashboard.py             # Streamlit live dashboard with attack injection buttons
├── baselines.py             # Baseline model comparisons (GBM, SVM, MLP, LogReg)
├── config.py                # Centralized configuration (env-var overridable)
├── requirements.txt         # Dependencies (flexible bounds)
├── requirements-lock.txt    # Pinned versions (reproducible builds)
├── esp32_firmware/
│   ├── platformio.ini       # PlatformIO build configuration
│   ├── src/main.cpp         # ESP32 firmware (inference + LEDs)
│   └── include/rf_model.h   # Exported Random Forest model (auto-generated)
├── tests/                   # Unit tests (pytest/unittest)
├── data/                    # Generated datasets (gitignored)
├── models/                  # Trained models (gitignored)
└── results/                 # Evaluation results (gitignored)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/shroff45/sensathon-2.0.git
cd sensathon-2.0
pip install -r requirements-lock.txt
```

### 2. Generate Data & Train Model
```bash
python generate_dataset.py    # ~290K train + ~58K test samples
python train_model.py         # RF training + ablation + baselines
python export_to_c.py         # Export model → esp32_firmware/rf_model.h
```

### 3. Flash ESP32 (Optional — software-only mode available)

**Hardware wiring:**
- GPIO 25 → 220Ω → Green LED (Normal)
- GPIO 26 → 220Ω → Yellow LED (Single-Layer Attack)
- GPIO 27 → 220Ω → Red LED (Coordinated Attack)

Open `esp32_firmware/` in PlatformIO and click **Build → Upload**.

### 4. Run the Demo

**Terminal 1 — Interactive simulation:**
```bash
python quick_demo.py
```

**Terminal 2 — Live dashboard:**
```bash
streamlit run dashboard.py
```

Open `http://localhost:8501` in your browser. Use sidebar buttons or keyboard (0-9) to inject attacks.

### 5. Windows Quick Launch
```bash
run_demo.bat
```

## 🔬 Classification

| Class | Label | LED | Description |
|-------|-------|-----|-------------|
| Normal | 0 | 🟢 Green | All sensor layers physically consistent |
| Single-Layer Attack | 1 | 🟡 Yellow | One sensor layer spoofed |
| Coordinated Attack | 2 | 🔴 Red | Multiple layers simultaneously compromised |

## 📊 Key Metrics

| Parameter | Value |
|-----------|-------|
| Features | 29 (19 raw + 6 cross-layer + 4 temporal) |
| Model | Random Forest (15 trees, depth 10) |
| Weighted F1 | ~94% |
| Coordinated Attack F1 | ~92% |
| Inference Latency | ~245 µs on ESP32 |
| Hardware Cost | ~$5 |

## 🧪 Running Tests
```bash
python -m pytest tests/
```

## 🔧 Configuration

Set environment variables to override defaults (see `config.py`):
```bash
IDS_PORT=COM4        # Serial port (default: auto-detect)
IDS_BAUD=115200      # Baud rate
IDS_MODEL_PATH=models/full_29_model.pkl
```

## 💡 Core Insight

> **Physics cannot be hacked.** An attacker can inject false data into any digital channel, but they cannot change the laws of physics. Our 6 cross-layer features encode kinematic constraints that serve as unforgeable lie detectors.

---

*Developed for the SENSEathon Hackathon Challenge.*
