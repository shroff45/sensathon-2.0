"""E2E Smoke Test — Verify the 4 previously-broken attack buttons"""
import numpy as np
import sys
sys.path.insert(0, '.')

from quick_demo import LiveSimulator, apply_attack, model, FEATURE_NAMES, PredictionSmoother
from generate_dataset import compute_cross_layer_features, compute_temporal_features

sim = LiveSimulator()

broken_attacks = ['v2x_fake_curv', 'can_dos', 'coord_gps_v2x', 'coord_speed_all']
results = {}

for attack_type in broken_attacks:
    detections = 0
    total = 30
    for step in range(total):
        raw = sim.generate_timestep()
        raw, true_label = apply_attack(raw, attack_type)
        xl = compute_cross_layer_features(raw)
        raw.update(xl)
        temporal = compute_temporal_features(sim.history, window=10)
        raw.update(temporal)
        sim.history.append(raw.copy())
        if len(sim.history) > 20:
            sim.history = sim.history[-15:]
        fv = np.array([raw.get(n, 0.0) for n in FEATURE_NAMES]).reshape(1, -1)
        pred = int(model.predict(fv)[0])
        if pred > 0:
            detections += 1
    det_rate = detections / total * 100
    status = 'PASS' if det_rate > 30 else 'FAIL'
    results[attack_type] = (det_rate, status)
    print(f"  {attack_type:20s} | Det: {det_rate:5.1f}% | {status}")

failures = [k for k, v in results.items() if v[1] == 'FAIL']
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
else:
    print("ALL 4 PREVIOUSLY-BROKEN BUTTONS VERIFIED")
