# Sensathon IDS Validation & Reporting

## What This Is
A hardened, edge-deployed Intrusion Detection System validating anomalous cross-layer telemetry via physical kinematics for Software-Defined Vehicles. The core build is completely locked and operational. The current critical milestone is to definitively test the system end-to-end and construct an exhaustive, presentation-ready technical report validating the detection rates, system stability, and edge-case boundaries.

## Core Value
**Confidence and Clarity:** Proving the system works exactly as intended under duress and cleanly communicating those validations to the hackathon judges through structured documentation.

## Requirements

### Validated
- ✓ Synthetic dataset generation via Kinematic Bicycle Model constraints
- ✓ Random Forest model training pipeline exporting to static `.pkl` and C-header constants
- ✓ Robust Real-Time Simulation and Serial Interfacing (`quick_demo.py` + `stream_to_esp32.py`)
- ✓ Active Edge Inference parsing memory-safe telemetry on the ESP32 microcontroller
- ✓ Live Dashboard visualization mapping attack timelines without IPC lockups

### Active
- [ ] Execute end-to-end systematic testing of the complete software-to-hardware pipeline
- [ ] Generate a highly detailed technical validation report capturing constraints, accuracy metrics, and architectural decisions

### Out of Scope
- Building new architectures — The system is feature-complete; focus is exclusively on stress testing and reporting.

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| End-to-End Test Priority | The user explicitly directed the focus strictly onto report generation and system confirmation rather than adding new UI elements | — Pending |

---
*Last updated: 2026-03-31 after initialization*

## Evolution
This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state
