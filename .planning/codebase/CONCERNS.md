# Codebase Concerns & Known Issues

This document highlights areas of technical debt, potential fragility, and security/performance considerations within the Sensathon IDS Implementation.

## Hardware & Edge Computing Boundaries

1.  **RAM Contamination (`esp32_firmware.ino`):**
    *   The `rf_model.h` binary structure heavily relies upon static `PROGMEM` constants for parsing nodes. If the dataset significantly expands (`n_estimators` > 20, or `max_depth` > 12), the firmware map will overwhelm the ESP32 dynamic heap tracking (currently integrated using `ESP.getFreeHeap()`).
    *   If memory leaks arise in `parseFloat`, the device loop `delay()` must be evaluated alongside explicitly clearing the `Serial.available()` buffer faster.
2.  **Physics Model Scaling Boundaries (`generate_dataset.py`):**
    *   Currently, mathematically bounding kinematic attributes explicitly uses wheelbases assumed via standard measurements (`L = 2.7m`). The physics logic starts to break down or generate false positives specifically at extremely low vehicle speeds (e.g. `< 0.5m/s`).
    *   This has been patched via explicit speed thresholds, suppressing logic outputs when parked. Modifying this logic without thorough test validation will destabilize the Random Forest.

## Simulation Pipeline Fraglity

1.  **File-Based IPC Conflicts (`dashboard.py` vs `quick_demo.py`):**
    *   Both scripts continuously perform asynchronous writes/reads against `latest.json` and `interactive_state.json`.
    *   Although the race condition is mitigated via `safe_load_json()` holding retry logic mapped to a `0.05s` delay block, high CPU congestion could result in Dashboard stutter mapping JSON parsing failures. A move to Websockets or Thread-Queues would resolve this permanently but adds complexity.
2.  **Serial Port Conformance:**
    *   Currently defaults explicitly to `COM3` / `/dev/ttyUSB0`. If the embedded node port changes dynamically across different operating systems (Windows vs Linux vs macOS), the user must explicitly alter configuration loops within the code.

## Long-Term Security

1.  **Spoofing Input Payload Safety:**
    *   While we hardened `isinf()` / `isnan()` payload values against causing hard arithmetic errors tracking string inputs, we do not employ packet cryptography. A malicious system could spoof the actual serial payload string (`V,...`) directly over UART if physically connected.
