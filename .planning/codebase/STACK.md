# Technology Stack

This document outlines the languages, runtimes, frameworks, and core dependencies used across the project to power both the software simulation and the embedded hardware inference node.

## Core Languages & Runtimes

*   **Python (3.x):** Used for the entirely of the Software-Defined Vehicle simulation, machine learning pipeline, data generation, and dashboard UI.
*   **C++ (Arduino/ESP32):** Used for the embedded edge node firmware, responsible for native execution of the Machine Learning model via C header map.
*   **FreeRTOS:** Multi-threading and task management embedded within the ESP32 firmware loop.

## AI & Data Pipeline (Python)

*   **Scikit-Learn (`sklearn`):** Core engine for training the Random Forest models, cross-validation, and ablation studies.
*   **Numpy / Pandas:** Used extensively for manipulating vectorized sensor physics, calculating vehicle physics logic (like the Kinematic Bicycle Model), and slicing datasets.
*   **Joblib:** Checkpointing and loading trained `.pkl` models for the interactive Python simulation layer (`quick_demo.py`).

## Visualization & UI (Python)

*   **Streamlit:** Drives the `dashboard.py` real-time demonstration UI mapping attack sequences visually.
*   **Plotly (`plotly.graph_objects`):** Responsible for building complex, live-updating radar charts representing Cross-Layer Physics consistencies. 

## Hardware & Embedded (C++)

*   **Arduino Core for ESP32:** Hardware abstraction layer for setting up the Serial connection, pins, and LEDs.
*   **In-house C++ Physics Engine:** A manual C-port of the Python physics formulas (normalized differences, velocity checks) baked directly into the `loop()` to perform cross-layer comparisons at the edge.

## File-Based IPC (Inter-Process Communication)

*   **JSON:** Both `dashboard.py` and the simulation payload scripts transmit state across threads securely using protected, retry-wrapped JSON `dumps/loads` mechanisms targeting `demo_results/`.
