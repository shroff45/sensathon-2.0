# System Architecture

The overarching system is designed as a localized node representing an advanced Software-Defined Vehicle (SDV) security system. It explicitly employs "Cross-Layer Physics Validation" to combat targeted multi-phase spoofing and GPS displacement attacks.

## Core Philosophical Pattern

*   **Physics First:** Validated against absolute kinematic truths like the *Bicycle Model* (calculating yaw dynamically from steering inputs) rather than solely relying upon isolated machine learning black boxes.
*   **Offline First / Edge Focused:** The most vulnerable component is the CAN bus; offloading the intelligence into a constrained environment (an ESP32 Edge Node) mimics the restrictions inherent within real Automotive ECU deployments.

## Major Layers & Modules

1.  **Event / Scenario Generator (`generate_dataset.py`)** 
    Responsible for generating completely synthetically accurate drive sessions mapping steering geometries directly to IMU responses. It injects classes of attacks (`GNSS Spoofing`, `IMU injection`, `CAN manipulation`).
2.  **Machine Learning Training Script (`train_model.py`)** 
    A modular script supporting robust cross-layer validation models with ablation pipelines (Sensor vs V2X vs CAN) exported efficiently via `Joblib`.
3.  **Real-Time Data Broker (`quick_demo.py` & `stream_to_esp32.py`)** 
    Serves as the interactive middleware that replaces the strict scripted demo logic. It loops continuously in `run_interactive_demo()` checking for user interaction signals via IPC state loops, then compiles packet strings formatted for the micro-controller.
4.  **Hardware Defense Node (`esp32_firmware.ino`)** 
    Executes native `RandomForest` binary decision trees dynamically against incoming bytes, validating that the underlying physics model remains intact (`dY_diff`, `LatAcc_diff`), firing the status back or lighting LEDs based on the classification probability.
5.  **Analytics Frontend (`dashboard.py`)** 
    React-driven Python architecture polling status every ~100ms drawing localized charts displaying exactly why an attack was classified (highlighting Cross-Layer consistency scores visually).

## Data Flow Pipeline

1.  *User Command* → Click Button via `dashboard.py` (e.g. "Inject GPS Spoofing").
2.  *State Update* → Writes to `demo_results/interactive_state.json`.
3.  *Simulator Read* → `quick_demo.py` loads State, applies GPS noise modifier inside realistic engine (`RealisticIMU.update(...)`).
4.  *Packet Translation* → `stream_to_esp32.py` structures a string like `V,... S,... C,...`.
5.  *Node Execution* → UART receives packet, ESP32 `setup()` / `loop()` decodes string, runs math, passes features into inference header.
6.  *Event Triggers* → ESP32 writes `R, [Classification] [Time]` to Serial buffer, lighting visual pins on board.
7.  *UI Loop* → Simulator intercepts `R,`, logs to `latest.json`, `dashboard.py` visualizes the detection anomaly.
