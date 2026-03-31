"""
REAL-TIME INTERACTIVE CROSS-LAYER IDS
No scripted sequence. Runs continuously.
Attack control via keyboard or dashboard buttons.

Controls:
  0 = Normal driving
  1 = GPS Speed Spoofing
  2 = CAN Steering Injection
  3 = V2X Fake Curvature
  4 = CAN DoS Flooding
  5 = Coordinated CAN + V2X (Phantom Curve)
  6 = Coordinated All Three Layers
  7 = Coordinated GPS + CAN Speed
  8 = Coordinated GPS + V2X Heading
  9 = UNSEEN: Coordinated Speed All
  Q = Quit
  H = Help

Run dashboard alongside: streamlit run dashboard.py
"""

import numpy as np
import joblib
import time
import json
import os
import sys
import threading
import math
from generate_dataset import (
    FEATURE_NAMES, VehicleState, update_vehicle,
    RealisticIMU, RealisticGPS, compute_cross_layer_features,
    compute_temporal_features, compute_can_entropy, NOISE,
    WHEELBASE, STEERING_RATIO, DT
)

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

RESULTS_DIR = 'demo_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature clipping bounds (training range protection)
FEATURE_CLIP_MIN = np.array([
    0, -1.5, -15, -1.5, -15, 0, -50,       # sensor F0-F6
    0, -1.2, 0, 0, 0, 0, 0,                 # CAN F7-F13
    0, 0, 0, 0, 0,                           # V2X F14-F18
    0, 0, 0, 0, 0, 0,                        # cross-layer F19-F24
    0, 0, 0, 0                               # temporal F25-F28
])

FEATURE_CLIP_MAX = np.array([
    50, 1.5, 15, 1.5, 15, 100, 50,          # sensor
    50, 1.2, 200, 100, 300, 6, 2,            # CAN
    0.15, 40, 200, 1.0, 50,                  # V2X
    2, 2, 2, 2, 2, 2,                        # cross-layer
    2, 2, 2, 2                               # temporal
])

# ═══════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════

print("Loading model...")
model = joblib.load('models/full_29_model.pkl')
model.n_jobs = 1  # Prevent threading crash on Python 3.13
print(f"✅ Model loaded: {model.n_estimators} trees, {model.n_features_in_} features")

# ═══════════════════════════════════════════════════════════
# PREDICTION SMOOTHER
# ═══════════════════════════════════════════════════════════

class PredictionSmoother:
    """Require N consecutive same-class predictions to prevent flicker"""
    def __init__(self, window=3):
        self.buffer = [0] * window
        self.idx = 0
        self.window = window

    def smooth(self, prediction):
        self.buffer[self.idx] = prediction
        self.idx = (self.idx + 1) % self.window
        votes = [0, 0, 0]
        for p in self.buffer:
            if 0 <= p <= 2:
                votes[p] += 1
        return votes.index(max(votes))

# ═══════════════════════════════════════════════════════════
# ATTACK STATE (thread-safe)
# ═══════════════════════════════════════════════════════════

class AttackState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current = 'none'
        self.label = 0
        self.description = '🟢 Normal driving'

    def set(self, attack_type, label, description):
        with self.lock:
            self.current = attack_type
            self.label = label
            self.description = description
            try:
                state_file = os.path.join(RESULTS_DIR, 'interactive_state.json')
                with open(state_file, 'w') as f:
                    json.dump({
                        'attack_type': attack_type,
                        'label': label,
                        'description': description,
                        'timestamp': time.time()
                    }, f)
            except OSError:
                pass

    def get(self):
        with self.lock:
            # Check if dashboard changed the attack
            try:
                state_file = os.path.join(RESULTS_DIR, 'interactive_state.json')
                if os.path.exists(state_file):
                    mtime = os.path.getmtime(state_file)
                    if time.time() - mtime < 1.0:
                        with open(state_file) as f:
                            data = json.load(f)
                        if data.get('attack_type') != self.current:
                            self.current = data['attack_type']
                            self.label = data.get('label', 0)
                            self.description = data.get('description', '')
            except (json.JSONDecodeError, FileNotFoundError, OSError):
                pass
            return self.current, self.label, self.description


attack_state = AttackState()

ATTACK_MAP = {
    '0': ('none',            0, '🟢 Normal driving'),
    '1': ('gps_spoof',       1, '🟡 GPS Speed Spoofing'),
    '2': ('can_inject',      1, '🟡 CAN Steering Injection'),
    '3': ('v2x_fake_curv',   1, '🟡 V2X Fake Curvature'),
    '4': ('can_dos',         1, '🟡 CAN DoS Flooding'),
    '5': ('coord_can_v2x',   2, '🔴 COORDINATED: CAN + V2X (Phantom Curve)'),
    '6': ('coord_all',       2, '🔴 COORDINATED: All Three Layers'),
    '7': ('coord_gps_can',   2, '🔴 COORDINATED: GPS + CAN Speed'),
    '8': ('coord_gps_v2x',   2, '🔴 COORDINATED: GPS + V2X Heading'),
    '9': ('coord_speed_all', 2, '🔴 UNSEEN: Speed Spoofing All Layers'),
}

# ═══════════════════════════════════════════════════════════
# KEYBOARD INPUT THREAD
# ═══════════════════════════════════════════════════════════

def keyboard_listener():
    while True:
        try:
            key = input().strip().lower()
            if key == 'q':
                print("\n🛑 Shutting down...")
                os._exit(0)
            elif key in ATTACK_MAP:
                atype, label, desc = ATTACK_MAP[key]
                attack_state.set(atype, label, desc)
                print(f"\n  >>> ATTACK CHANGED: {desc}")
                print(f"  >>> Press 0 to return to normal\n")
            elif key in ('h', 'help'):
                print_controls()
        except EOFError:
            break
        except Exception:
            pass


def print_controls():
    print("""
  ╔═══════════════════════════════════════════════════════╗
  ║  ATTACK CONTROL PANEL                                ║
  ╠═══════════════════════════════════════════════════════╣
  ║  0 = Normal driving (clear attack)                   ║
  ║  1 = GPS Speed Spoofing                              ║
  ║  2 = CAN Steering Injection                          ║
  ║  3 = V2X Fake Road Curvature                         ║
  ║  4 = CAN DoS Flooding                                ║
  ║  5 = Coordinated CAN + V2X (Phantom Curve)           ║
  ║  6 = Coordinated All Three Layers                    ║
  ║  7 = Coordinated GPS + CAN Speed                     ║
  ║  8 = Coordinated GPS + V2X Heading                   ║
  ║  9 = UNSEEN: Speed Spoofing All Layers               ║
  ║  Q = Quit          H = Show this help                ║
  ╠═══════════════════════════════════════════════════════╣
  ║  Or use Streamlit dashboard sidebar buttons           ║
  ╚═══════════════════════════════════════════════════════╝
    """)


# ═══════════════════════════════════════════════════════════
# ATTACK APPLICATION
# ═══════════════════════════════════════════════════════════

def apply_attack(raw, attack_type):
    if attack_type == 'none':
        return raw, 0

    elif attack_type == 'gps_spoof':
        raw['gps_speed'] += np.random.uniform(8, 15)
        return raw, 1

    elif attack_type == 'can_inject':
        fake_steer = np.random.uniform(0.6, 1.0) * np.random.choice([-1, 1])
        raw['can_steering_angle'] = fake_steer
        raw['can_payload_anomaly'] = abs(np.random.normal(0.3, 0.1))
        return raw, 1

    elif attack_type == 'v2x_fake_curv':
        raw['v2x_road_curvature'] = np.random.uniform(0.03, 0.06)
        return raw, 1

    elif attack_type == 'can_dos':
        raw['can_id_entropy'] = np.random.uniform(0.3, 1.0)
        raw['can_msg_freq_dev'] = np.random.uniform(80, 200)
        raw['can_payload_anomaly'] = np.random.uniform(0.5, 1.0)
        return raw, 1

    elif attack_type == 'coord_can_v2x':
        fake_steer = np.random.uniform(0.5, 0.9) * np.random.choice([-1, 1])
        fake_wheel = max(-0.6, min(0.6, fake_steer / STEERING_RATIO))
        fake_curv = abs(math.tan(fake_wheel) / WHEELBASE)
        raw['can_steering_angle'] = fake_steer
        raw['v2x_road_curvature'] = fake_curv
        return raw, 2

    elif attack_type == 'coord_all':
        fake_steer = np.random.uniform(0.5, 0.9) * np.random.choice([-1, 1])
        fake_wheel = max(-0.6, min(0.6, fake_steer / STEERING_RATIO))
        fake_yaw = raw['can_wheel_speed'] * math.tan(fake_wheel) / WHEELBASE
        fake_curv = abs(math.tan(fake_wheel) / WHEELBASE)
        raw['can_steering_angle'] = fake_steer
        raw['v2x_road_curvature'] = fake_curv
        raw['gps_heading_rate'] = fake_yaw
        return raw, 2

    elif attack_type == 'coord_gps_can':
        speed_boost = np.random.uniform(8, 15)
        fake_speed = raw['gps_speed'] + speed_boost
        raw['gps_speed'] = fake_speed
        raw['can_wheel_speed'] = fake_speed
        return raw, 2

    elif attack_type == 'coord_gps_v2x':
        fake_hr = np.random.uniform(0.15, 0.4) * np.random.choice([-1, 1])
        fake_curv = abs(fake_hr / max(abs(raw['gps_speed']), 1.0))
        raw['gps_heading_rate'] = fake_hr
        raw['v2x_road_curvature'] = fake_curv
        return raw, 2

    elif attack_type == 'coord_speed_all':
        speed_boost = np.random.uniform(10, 20)
        fake_speed = raw['gps_speed'] + speed_boost
        raw['gps_speed'] = fake_speed
        raw['can_wheel_speed'] = fake_speed
        raw['v2x_speed_limit'] = fake_speed + 5
        return raw, 2

    return raw, 0


# ═══════════════════════════════════════════════════════════
# DRIVING SIMULATOR
# ═══════════════════════════════════════════════════════════

class LiveSimulator:
    def __init__(self):
        self.state = VehicleState(speed=np.random.uniform(18, 25))
        self.imu = RealisticIMU()
        self.gps = RealisticGPS()
        self.prev_ultra = 50.0
        self.history = []
        self.step = 0
        self.drive_mode = 'highway'
        self.mode_timer = 0
        self.mode_duration = np.random.randint(200, 500)
        self.turn_direction = np.random.choice([-1, 1])
        self.target_speed_base = np.random.uniform(18, 30)

    def _update_driving_mode(self):
        self.mode_timer += 1
        if self.mode_timer >= self.mode_duration:
            self.mode_timer = 0
            self.mode_duration = np.random.randint(200, 500)
            self.turn_direction = np.random.choice([-1, 1])
            modes = ['highway', 'gentle_curve', 'urban', 'lane_change']
            weights = [0.4, 0.25, 0.2, 0.15]
            self.drive_mode = np.random.choice(modes, p=weights)
            self.target_speed_base = {
                'highway': np.random.uniform(22, 32),
                'gentle_curve': np.random.uniform(15, 22),
                'urban': np.random.uniform(8, 15),
                'lane_change': np.random.uniform(20, 28),
            }[self.drive_mode]

    def generate_timestep(self):
        self._update_driving_mode()
        self.step += 1

        if self.drive_mode == 'highway':
            target_steer = np.random.normal(0, 0.008)
            target_speed = self.target_speed_base + np.random.normal(0, 0.3)
        elif self.drive_mode == 'gentle_curve':
            target_steer = self.turn_direction * np.random.uniform(0.08, 0.15) + np.random.normal(0, 0.01)
            target_speed = self.target_speed_base + np.random.normal(0, 0.2)
        elif self.drive_mode == 'urban':
            phase = self.mode_timer % 80
            if 30 < phase < 45:
                target_steer = self.turn_direction * np.random.uniform(0.15, 0.3)
            else:
                target_steer = np.random.normal(0, 0.02)
            target_speed = max(3, self.target_speed_base + np.random.normal(0, 1.0))
        elif self.drive_mode == 'lane_change':
            phase = self.mode_timer % 60
            if 10 < phase < 20:
                target_steer = self.turn_direction * 0.12
            elif 25 < phase < 35:
                target_steer = -self.turn_direction * 0.12
            else:
                target_steer = np.random.normal(0, 0.01)
            target_speed = self.target_speed_base + np.random.normal(0, 0.2)
        else:
            target_steer = np.random.normal(0, 0.01)
            target_speed = 20.0

        self.state = update_vehicle(self.state, target_steer, target_speed)

        raw = {
            'gps_speed': self.gps.read_speed(self.state.speed),
            'gps_heading_rate': self.gps.read_heading_rate(self.state.yaw_rate),
            'imu_lat_accel': self.imu.read_lat_accel(self.state.lat_accel),
            'imu_yaw_rate': self.imu.read_yaw_rate(self.state.yaw_rate),
            'imu_lon_accel': self.imu.read_lon_accel(self.state.lon_accel),
            'ultrasonic_min': max(0.02, self.state.obstacle_dist + np.random.normal(0, NOISE['ultrasonic'])),
            'ultrasonic_rate': 0.0,
            'can_wheel_speed': self.state.speed + np.random.normal(0, NOISE['can_wheel_speed']),
            'can_steering_angle': self.state.steering_wheel_angle + np.random.normal(0, NOISE['can_steering']),
            'can_brake_pressure': max(0, -self.state.lon_accel * 15 + np.random.normal(0, 2)),
            'can_throttle_pos': max(0, min(100, self.state.lon_accel * 20 + 30 + np.random.normal(0, 3))),
            'can_msg_freq_dev': abs(np.random.normal(0, 3)),
            'can_id_entropy': compute_can_entropy(),
            'can_payload_anomaly': abs(np.random.normal(0, 0.05)),
            'v2x_road_curvature': abs(self.state.yaw_rate / max(abs(self.state.speed), 1.0)) + np.random.normal(0, 0.002),
            'v2x_speed_limit': np.random.choice([13.89, 16.67, 22.22, 27.78, 33.33]),
            'v2x_obstacle_dist': self.state.obstacle_dist + np.random.normal(0, 1.0),
            'v2x_auth_score': np.random.choice([0.9, 0.95, 1.0], p=[0.1, 0.2, 0.7]),
            'v2x_msg_frequency': 10.0 + np.random.normal(0, 0.5),
        }

        raw['ultrasonic_rate'] = (raw['ultrasonic_min'] - self.prev_ultra) / DT
        self.prev_ultra = raw['ultrasonic_min']

        return raw

    def compute_all_features(self, raw):
        xl = compute_cross_layer_features(raw)
        raw.update(xl)
        temporal = compute_temporal_features(self.history, window=10)
        raw.update(temporal)
        self.history.append(raw.copy())
        if len(self.history) > 20:
            self.history = self.history[-15:]
        return raw, xl, temporal


# ═══════════════════════════════════════════════════════════
# MAIN REAL-TIME LOOP
# ═══════════════════════════════════════════════════════════

def main():
    sim = LiveSimulator()
    smoother = PredictionSmoother(window=3)
    log_data = []
    stats = {'correct': 0, 'total': 0, 'attacks_detected': 0, 'attacks_total': 0}

    print("\n" + "="*70)
    print("  🛡️  CROSS-LAYER PHYSICS-BASED IDS — REAL-TIME MODE")
    print("="*70)
    print(f"  {model.n_estimators} trees | {model.n_features_in_} features | 7 physics equations")
    print("  Vehicle simulation running continuously")
    print("  Type a number + Enter to trigger attacks")
    print("="*70)
    print_controls()
    print("  System running... Type attack number + Enter at any time\n")

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    start_time = time.time()

    try:
        while True:
            loop_start = time.time()
            step = sim.step
            elapsed = time.time() - start_time

            attack_type, expected_label, description = attack_state.get()
            raw = sim.generate_timestep()
            raw, true_label = apply_attack(raw, attack_type)
            raw, xl_features, temporal_features = sim.compute_all_features(raw)

            # Build feature vector with validation
            feature_vector = []
            for name in FEATURE_NAMES:
                val = raw.get(name, 0.0)
                if not np.isfinite(val):
                    val = 0.0
                feature_vector.append(val)

            fv = np.array(feature_vector).reshape(1, -1)

            # Clip to training range
            fv = np.clip(fv, FEATURE_CLIP_MIN.reshape(1, -1),
                         FEATURE_CLIP_MAX.reshape(1, -1))

            # Classify
            raw_prediction = int(model.predict(fv)[0])
            prediction = smoother.smooth(raw_prediction)

            # Confidence
            proba = model.predict_proba(fv)[0]
            confidence = float(max(proba))

            # Response mode
            mode_names = {0: 'NORMAL', 1: 'DEGRADED', 2: 'SAFE_STOP'}
            mode = mode_names.get(prediction, 'UNKNOWN')

            # Track accuracy
            is_correct = (prediction == 0 and true_label == 0) or \
                         (prediction > 0 and true_label > 0)
            stats['total'] += 1
            if is_correct:
                stats['correct'] += 1
            if true_label > 0:
                stats['attacks_total'] += 1
                if prediction > 0:
                    stats['attacks_detected'] += 1

            # Build log entry
            entry = {
                'step': step,
                'time': round(elapsed, 1),
                'prediction': prediction,
                'true_label': true_label,
                'raw_prediction': raw_prediction,
                'confidence': round(confidence, 3),
                'vehicle_mode': prediction,
                'mode_name': mode,
                'feature_time_us': 0,
                'inference_time_us': 0,
                'attack_type': attack_type,
                'attack_name': attack_type,
                'description': description,
                'drive_mode': sim.drive_mode,
                'speed': round(float(sim.state.speed), 1),
                'steering': round(float(sim.state.steering_wheel_angle), 4),
                'xl_speed_consistency': round(float(xl_features.get('xl_speed_consistency', 0)), 4),
                'xl_yaw_can_vs_gps': round(float(xl_features.get('xl_yaw_can_vs_gps', 0)), 4),
                'xl_yaw_can_vs_imu': round(float(xl_features.get('xl_yaw_can_vs_imu', 0)), 4),
                'xl_lataccel_gps_vs_imu': round(float(xl_features.get('xl_lataccel_gps_vs_imu', 0)), 4),
                'xl_obstacle_ultra_vs_v2x': round(float(xl_features.get('xl_obstacle_ultra_vs_v2x', 0)), 4),
                'xl_curvature_3way': round(float(xl_features.get('xl_curvature_3way', 0)), 4),
                'xl_accel_consistency': round(float(temporal_features.get('xl_accel_consistency', 0)), 4),
                'xl_score_variance': round(float(temporal_features.get('xl_score_variance', 0)), 4),
                'xl_steering_jerk': round(float(temporal_features.get('xl_steering_jerk', 0)), 4),
                'xl_heading_integral_diff': round(float(temporal_features.get('xl_heading_integral_diff', 0)), 4),
            }

            log_data.append(entry)
            if len(log_data) > 5000:
                log_data = log_data[-3000:]

            # Save latest for dashboard
            try:
                with open(os.path.join(RESULTS_DIR, 'latest.json'), 'w') as f:
                    json.dump(entry, f)
            except OSError:
                pass

            # Save full log every 10 seconds
            if step % 100 == 0 and log_data:
                try:
                    with open(os.path.join(RESULTS_DIR, 'full_demo_log.json'), 'w') as f:
                        json.dump(log_data[-2000:], f)
                except OSError:
                    pass

            # Console output every second
            if step % 10 == 0:
                pred_str = ['🟢 NORMAL ', '🟡 SINGLE ', '🔴 COORD  '][prediction]
                check = "✅" if is_correct else "❌"

                f21 = xl_features.get('xl_yaw_can_vs_imu', 0)
                spd_c = xl_features.get('xl_speed_consistency', 0)

                accuracy = stats['correct'] / max(stats['total'], 1) * 100
                det_rate = stats['attacks_detected'] / max(stats['attacks_total'], 1) * 100 \
                           if stats['attacks_total'] > 0 else 0

                drive_short = sim.drive_mode[:3].upper()

                sys.stdout.write(
                    f"\r  {elapsed:6.1f}s | {pred_str} | {check} | "
                    f"F21={f21:.3f} SpC={spd_c:.3f} | "
                    f"Conf={confidence:.0%} | "
                    f"{drive_short} {sim.state.speed:.0f}m/s | "
                    f"Acc={accuracy:.0f}% Det={det_rate:.0f}% | "
                    f"{description}"
                    f"          "
                )
                sys.stdout.flush()

            # Maintain real-time pace
            loop_elapsed = time.time() - loop_start
            sleep_time = max(0, DT - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    finally:
        print(f"\n\n{'='*70}")
        print(f"  SESSION COMPLETE")
        print(f"  Duration: {time.time() - start_time:.0f}s")
        print(f"  Total samples: {stats['total']}")
        print(f"  Overall accuracy: {stats['correct']}/{stats['total']} = "
              f"{stats['correct']/max(stats['total'],1)*100:.1f}%")
        if stats['attacks_total'] > 0:
            print(f"  Attack detection: {stats['attacks_detected']}/{stats['attacks_total']} = "
                  f"{stats['attacks_detected']/stats['attacks_total']*100:.1f}%")
        print(f"{'='*70}")

        try:
            with open(os.path.join(RESULTS_DIR, 'full_demo_log.json'), 'w') as f:
                json.dump(log_data, f)
        except OSError:
            pass


if __name__ == '__main__':
    main()
