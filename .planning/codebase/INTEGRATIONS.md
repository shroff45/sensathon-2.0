# Integrations Map

This codebase does not heavily map external HTTP APIs since it's designed to run offline on an embedded ESP32 Node combined with a standard Python UI simulation environment targeting a Software-Defined Vehicle (SDV).

## Core Interfaces

### 1. Serial Port / UART Pipeline

**Where:** `stream_to_esp32.py`, `dashboard.py`, `esp32_firmware.ino`
**Role:**
The single most critical integration is the UART Serial transmission linking the robust Python backend (acting as the SDV vehicle sensors) and the ESP32 (acting as the localized Cross-Layer Intrusion Detection System).

**Protocol Implementation Details:**
*   Strings format delimited parameters via commas: `V,...`, `S,...`, `C,...` for Vehicle dynamics, Sensors, and CAN layers respectively.
*   Data parsing checks are tightly controlled inside C++ string manipulators (e.g. tracking `NaN` and `Inf`).
*   Output string `R,...` delivers the Random Forest prediction score and classification back to the computer.

### 2. Streamlit Web UI Sandbox

**Where:** `dashboard.py` via `streamlit`
**Role:**
Serves as an HTML5 integration binding localhost to the interactive hackathon controls and data visualizers. It parses telemetry state continuously.

**Mechanism:**
It integrates with the data layer strictly via file JSON buffers, circumventing multi-processing locks via customized polling implementations (`demo_results/latest.json`).

### 3. V2X & Real-Time GNSS Simulation

**Role:**
Simulated GPS coordinates injected into the physics calculations via noise modules inside `generate_dataset.py` representing standard real-world sensor drift vs explicit spoofing attacks.

### No Cloud Database integrations are present. State is intentionally transient or locked within `.json` text logs and `.csv` local data stores specifically scaled to match hackathon demonstration latency limits.
