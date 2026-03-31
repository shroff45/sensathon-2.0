# Development Conventions

These conventions reflect the priority of hardening the Sensathon IDS Implementation.

## Code Design Strategy

*   **Defensive Embedded C++ (`esp32_firmware.ino`):**
    *   Must employ rigorous validation for `Serial.parseFloat()` inputs utilizing `isinf()` and `isnan()` flags.
    *   Always utilize robust clamping (`max(abs(a), epsilon)`) before division to protect against `ZeroDivisionError` lockups across constrained devices.
    *   Enforce hardware watchdog yields (`vTaskDelay(1)`) explicitly within infinite processing or packet string parsing `while()` structures.
*   **Thread-Safe IPC Access (`dashboard.py` / `quick_demo.py`):**
    *   Always implement explicit file-locking simulations using robust iteration logic like `safe_load_json()` protecting shared configuration files from race-conditions between Streamlit Readers and Simulator Writers.
*   **Predictable Machine Learning Structure (`train_model.py`):**
    *   Random Forest configurations explicitly capped (`max_depth=12`, `n_estimators=20`) to map memory consumption dynamically targeting edge limits instead of optimizing purely for Python-based F1 performance models.

## Structure and Naming

*   **Constants over Magic Numbers:** Hard thresholds (like Velocity checks at `< 0.5m/s` or the Wheelbase coefficient `L = 2.7m`) are documented clearly inline to clarify why specific anomaly features are masked out during standstill tests.
*   **File Isolation:** Output streams are segregated. Models inside `/models/`, Data inside `/data/`, Realtime analytics inside `/demo_results/`.
*   **Error Handling:** Use aggressive error tracking loops printing JSON failures visually into the console explicitly without masking the unparsed stack trace.
