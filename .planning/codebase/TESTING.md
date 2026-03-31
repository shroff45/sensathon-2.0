# Testing & Verification

The testing approach relies fundamentally on simulated attack vectors across the complete dataset spectrum, mapping Random Forest metrics via Stratified Cross-Validation techniques evaluated against a Ground-Truth system running in isolated Python threads.

## Frameworks & Processes

1.  **Synthetic Dataset Baseline Tests (`generate_dataset.py`)** 
    Validates Kinematic consistency over physical limits by clamping bounds (Steering at `±0.6 rads`, Velocity thresholding). Creates 1500 regular, multi-modal, and edge-case samples.
2.  **Machine Learning Metrics (`train_model.py`)** 
    Logs ablation studies generating output payloads across:
    *   Confusion Matrices.
    *   Feature Importance outputs (identifying strictly how important the Physics/Cross-Layer metrics are vs Raw limits).
    *   Weighted F1 classification summaries per-attack vector.
3.  **Real-Time Simulation Testing (`quick_demo.py`)** 
    Continuously loops over a dynamic "Interactive Modality", forcing system developers to manually verify visual response timings against hard-coded anomalies injected interactively.
4.  **Hardware In The Loop Verification (`esp32_firmware.ino`)** 
    Checks the serialized string payload against native physical constraint bounds (`dY`, `Ay`, `aY`). The hardware is functionally verifying Python implementations of formulas by actively executing matching C logic on limited compute.
