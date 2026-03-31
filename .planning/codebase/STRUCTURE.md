# Codebase Structure

The project directory breaks cleanly into domain-specific clusters targeting Dataset Engineering, Machine Learning Training, Live Demonstration Tooling, and Embedded Firmware Execution.

## Root Level Topology

### 1. The Core Simulator Scripts
The heart of the Python infrastructure exists here.
*   `generate_dataset.py` – Holds mathematical logic and physical bounds testing (Kinematic rules). The heavy-lifter.
*   `train_model.py` – Orchestrates hyperparameter assignments.
*   `export_to_c.py` – Utility script wrapping ML predictions natively.
*   `baselines.py` – Scripts used for extracting non-cross-layer comparison data against the unified model.

### 2. The Interactive Demonstration Module
*   `quick_demo.py` – Contains the `LiveSimulator` and controls state management.
*   `dashboard.py` – The Streamlit GUI, structured around the `st.set_page_config` and containing localized polling layouts.
*   `stream_to_esp32.py` – Wrapper handling serial protocols exclusively.

### 3. Generated Components (Dynamically Modified Directories)
*   `/data/` – `*.csv` files holding compiled simulation runs across 1500+ trials.
*   `/models/` – The generated `Joblib` `.pkl` pipeline objects holding decision boundaries.
*   `/results/` and `/demo_results/` – The hot-loading `JSON` and `NPY` logs buffering rapid analytics state.
*   `/.planning/` – The GSD autonomous workflow documentation and system context maps (where this file resides).

### 4. Edge Environment Configuration
*   `/esp32_firmware/`
    *   `esp32_firmware.ino` – The single Arduino-format translation layer applying physical validation in C++.
    *   `rf_model.h` – C header files produced by the pipeline carrying static memory definitions mapping the 20 Random Forest trees directly onto the IC board.

## Naming Conventions & Organization
*   *Snake Case* primarily utilized across Python scripts emphasizing data-science-style procedural steps (`generate_dataset`, `train_model`).
*   Object-Oriented approaches are utilized internally to track complex states across the SDV lifecycle (`class LiveSimulator`, `class RealisticIMU`).
*   Configuration limits (Baud Rates, Thresholds, Clamping limits) are usually placed explicitly at the top of loops to make iterative testing faster.
