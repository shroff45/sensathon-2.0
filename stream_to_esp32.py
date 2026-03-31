"""
DEMO ORCHESTRATOR - Sends simulated vehicle data to ESP32 over serial.
Supports both scripted demo mode and interactive judge mode.
"""

import serial
import serial.tools.list_ports
import time
import json
import os
import sys
import shutil
import tempfile
import numpy as np
import math
import threading
import logging
from generate_dataset import (
    FEATURE_NAMES, WHEELBASE, STEERING_RATIO, DT,
    VehicleState, update_vehicle, RealisticIMU, RealisticGPS,
    compute_cross_layer_features, compute_temporal_features,
    compute_can_entropy, AttackGenerator, safe_normalized_diff, NOISE
)

BAUD_RATE = 115200
RESULTS_DIR = 'demo_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
current_attack = {"type": "none", "lock": threading.Lock()}

def find_esp32():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = port.description.lower()
        if 'cp210' in desc or 'ch340' in desc or 'ftdi' in desc or 'usb' in desc:
            print(f"Found potential ESP32: {port.device} ({port.description})")
            return port.device
    if ports:
        print(f"No ESP32 auto-detected. Available ports:")
        for i, port in enumerate(ports):
            print(f"  [{i}] {port.device}: {port.description}")
        choice = input("Select port number (or enter port path): ").strip()
        try: return ports[int(choice)].device
        except (ValueError, IndexError): return choice
    return None

def connect_esp32(port=None):
    if port is None: port = find_esp32()
    if port is None:
        print("No serial port found. Running in SOFTWARE-ONLY mode.")
        return None
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
        for _ in range(10):
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"ESP32: {line}")
                if 'READY' in line:
                    print("ESP32 connected and ready!")
                    return ser
            time.sleep(0.3)
        print("ESP32 connected but no READY signal. Proceeding anyway.")
        return ser
    except Exception as e:
        print(f"Serial connection failed: {e}")
        print("Running in SOFTWARE-ONLY mode.")
        return None

def format_packet(raw_features):
    """Format 19 raw features into ESP32 packet format with validation"""
    s_keys = [
        'gps_speed', 'gps_heading_rate', 'imu_lat_accel', 'imu_yaw_rate',
        'imu_lon_accel', 'ultrasonic_min', 'ultrasonic_rate'
    ]
    c_keys = [
        'can_wheel_speed', 'can_steering_angle', 'can_brake_pressure',
        'can_throttle_pos', 'can_msg_freq_dev', 'can_id_entropy', 'can_payload_anomaly'
    ]
    v_keys = [
        'v2x_road_curvature', 'v2x_speed_limit', 'v2x_obstacle_dist',
        'v2x_auth_score', 'v2x_msg_frequency'
    ]
    
    # Validate all values are finite
    for key in s_keys + c_keys + v_keys:
        val = raw_features.get(key, 0.0)
        if not np.isfinite(val):
            raw_features[key] = 0.0
    
    s_vals = ','.join(f"{raw_features[k]:.4f}" for k in s_keys)
    c_vals = ','.join(f"{raw_features[k]:.4f}" for k in c_keys)
    v_vals = ','.join(f"{raw_features[k]:.4f}" for k in v_keys)
    
    packet = f"S,{s_vals}|C,{c_vals}|V,{v_vals}\n"
    
    if len(packet) > 500:
        raise ValueError(f"Packet too long: {len(packet)} chars")
    
    return packet

def parse_response(line):
    parts = line.strip().split(',')
    if len(parts) >= 6 and parts[0] == 'R':
        result = {
            'prediction': int(parts[1]), 'raw_prediction': int(parts[2]),
            'vehicle_mode': int(parts[3]), 'feature_time_us': int(parts[4]),
            'inference_time_us': int(parts[5]),
        }
        xl_names = ['xl_speed_consistency', 'xl_yaw_can_vs_gps', 'xl_yaw_can_vs_imu',
                     'xl_lataccel_gps_vs_imu', 'xl_obstacle_ultra_vs_v2x', 'xl_curvature_3way',
                     'xl_accel_consistency', 'xl_score_variance', 'xl_steering_jerk', 'xl_heading_integral_diff']
        for i, name in enumerate(xl_names):
            if 6 + i < len(parts): result[name] = float(parts[6 + i])
        return result
    return None

class DemoSimulator:
    def __init__(self):
        self.state = VehicleState(speed=20.0)
        self.imu = RealisticIMU()
        self.gps = RealisticGPS()
        self.prev_ultrasonic = 50.0
        self.feature_history = []
        self.t = 0

    def generate_raw_features(self, attack_type='none'):
        target_steer = np.random.normal(0, 0.03)
        target_speed = self.state.speed + np.random.normal(0, 0.3)
        if self.t % 100 < 20: target_steer = 0.2 * np.sin(self.t * 0.1)
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
            'v2x_speed_limit': 22.22,
            'v2x_obstacle_dist': self.state.obstacle_dist + np.random.normal(0, 1.0),
            'v2x_auth_score': 1.0,
            'v2x_msg_frequency': 10.0 + np.random.normal(0, 0.5),
        }
        raw['ultrasonic_rate'] = (raw['ultrasonic_min'] - self.prev_ultrasonic) / DT
        self.prev_ultrasonic = raw['ultrasonic_min']
        label = 0
        attack_name = 'normal'
        if attack_type == 'gps_spoof':
            raw['gps_speed'] += 10.0; label = 1; attack_name = 'gps_speed_spoof'
        elif attack_type == 'can_inject':
            raw['can_steering_angle'] = 0.5 * np.random.choice([-1, 1]); label = 1; attack_name = 'can_steering_inject'
        elif attack_type == 'v2x_fake' or attack_type == 'v2x_fake_curv':
            raw['v2x_road_curvature'] = 0.04; label = 1; attack_name = 'v2x_curvature_fake'
        elif attack_type == 'can_dos':
            raw['can_id_entropy'] = np.random.uniform(0.3, 1.0)
            raw['can_msg_freq_dev'] = np.random.uniform(80, 200)
            raw['can_payload_anomaly'] = np.random.uniform(0.5, 1.0)
            label = 1; attack_name = 'can_dos_flooding'
        elif attack_type == 'coord_can_v2x':
            fake_steer = 0.6 * np.random.choice([-1, 1])
            fake_wheel = fake_steer / STEERING_RATIO
            fake_curv = abs(math.tan(fake_wheel) / WHEELBASE)
            raw['can_steering_angle'] = fake_steer; raw['v2x_road_curvature'] = fake_curv
            label = 2; attack_name = 'coord_can_v2x'
        elif attack_type == 'coord_all':
            fake_steer = 0.6 * np.random.choice([-1, 1])
            fake_wheel = fake_steer / STEERING_RATIO
            fake_yaw = raw['can_wheel_speed'] * math.tan(fake_wheel) / WHEELBASE
            fake_curv = abs(math.tan(fake_wheel) / WHEELBASE)
            raw['can_steering_angle'] = fake_steer; raw['v2x_road_curvature'] = fake_curv
            raw['gps_heading_rate'] = fake_yaw; label = 2; attack_name = 'coord_all_three'
        elif attack_type == 'coord_gps_can':
            speed_boost = 12.0; fake_speed = raw['gps_speed'] + speed_boost
            raw['gps_speed'] = fake_speed; raw['can_wheel_speed'] = fake_speed
            label = 2; attack_name = 'coord_gps_can'
        elif attack_type == 'coord_gps_v2x':
            fake_hr = np.random.uniform(0.15, 0.4) * np.random.choice([-1, 1])
            fake_curv = abs(fake_hr / max(abs(raw['gps_speed']), 1.0))
            raw['gps_heading_rate'] = fake_hr; raw['v2x_road_curvature'] = fake_curv
            label = 2; attack_name = 'coord_gps_v2x'
        elif attack_type == 'coord_speed_all':
            speed_boost = np.random.uniform(10, 20)
            fake_speed = raw['gps_speed'] + speed_boost
            raw['gps_speed'] = fake_speed; raw['can_wheel_speed'] = fake_speed
            raw['v2x_speed_limit'] = fake_speed + 5
            label = 2; attack_name = 'coord_speed_all'
        self.t += 1
        return raw, label, attack_name

DEMO_SEQUENCE = [
    (0, 300, 'none', 'Normal highway driving'),
    (300, 500, 'gps_spoof', 'GPS Speed Spoofing (single-layer)'),
    (500, 700, 'none', 'Normal - recovery'),
    (700, 900, 'can_inject', 'CAN Steering Injection (single-layer)'),
    (900, 1100, 'none', 'Normal - recovery'),
    (1100, 1350, 'coord_can_v2x', 'COORDINATED: CAN + V2X (Phantom Curve)'),
    (1350, 1500, 'none', 'Normal - recovery'),
    (1500, 1700, 'coord_all', 'COORDINATED: CAN + V2X + GPS (all three)'),
    (1700, 1800, 'none', 'Normal - final recovery'),
]

def get_scripted_attack(step):
    for start, end, attack_type, desc in DEMO_SEQUENCE:
        if start <= step < end: return attack_type, desc
    return 'none', 'Normal'

def set_attack(attack_type):
    with current_attack["lock"]: current_attack["type"] = attack_type

def get_attack():
    with current_attack["lock"]: return current_attack["type"]

def run_demo(mode='scripted', port=None, total_steps=1800):
    ser = connect_esp32(port)
    use_hardware = ser is not None
    if not use_hardware:
        print("\nRunning in SOFTWARE-ONLY mode (no ESP32)")
        try:
            import joblib
            sw_model = joblib.load('models/full_29_model.pkl')
            print("Software model loaded as fallback")
        except Exception:
            print("No software model found. Run train_model.py first!")
            return
    sim = DemoSimulator()
    results_log = []
    print("\n" + "=" * 60)
    print(f"DEMO MODE: {mode.upper()}")
    print("=" * 60 + "\n")
    for step in range(total_steps):
        if mode == 'scripted': attack_type, description = get_scripted_attack(step)
        else: attack_type = get_attack(); description = f"Interactive: {attack_type}"
        raw, true_label, attack_name = sim.generate_raw_features(attack_type)
        packet = format_packet(raw)
        result = None
        if use_hardware:
            try:
                ser.write(packet.encode())
                response_line = ser.readline().decode('utf-8', errors='ignore').strip()
                if response_line: result = parse_response(response_line)
            except Exception as e: print(f"Serial error: {e}")
        if result is None and not use_hardware:
            # Compute cross-layer features
            xl = compute_cross_layer_features(raw)
            raw.update(xl)
            
            # Temporal features need history
            temporal = compute_temporal_features(sim.feature_history, window=10)
            raw.update(temporal)
            
            # Keep history bounded
            sim.feature_history.append(raw.copy())
            if len(sim.feature_history) > 20:
                sim.feature_history = sim.feature_history[-15:]
            
            # Build feature vector in correct order
            feature_vector = []
            for name in FEATURE_NAMES:
                feature_vector.append(raw.get(name, 0.0))
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            prediction = int(sw_model.predict(feature_vector)[0])
            
            result = {
                'prediction': prediction,
                'raw_prediction': prediction,
                'vehicle_mode': 0 if prediction == 0 else (1 if prediction == 1 else 2),
                'feature_time_us': 0,
                'inference_time_us': 0,
            }
            result.update(xl)
            result.update(temporal)
        if result:
            entry = {'step': step, 'time': round(step * DT, 1), 'true_label': true_label,
                     'attack_name': attack_name, 'attack_type': attack_type, 'description': description, **result}
            results_log.append(entry)
            latest_file = os.path.join(RESULTS_DIR, 'latest.json')
            tmp_fd, tmp_path = tempfile.mkstemp(dir=RESULTS_DIR, suffix='.json')
            try:
                with os.fdopen(tmp_fd, 'w') as f:
                    json.dump(entry, f)
                shutil.move(tmp_path, latest_file)
            except OSError:
                try: os.unlink(tmp_path)
                except OSError: pass
            if step % 10 == 0:
                pred = result['prediction']
                pred_str = ['NORMAL', 'SINGLE', 'COORD'][pred]
                correct = "OK" if (pred == 0 and true_label == 0) or (pred > 0 and true_label > 0) else "MISS"
                timing = ""
                if result.get('feature_time_us', 0) > 0:
                    timing = f" [{result['feature_time_us']}+{result['inference_time_us']}us]"
                print(f"  t={step * DT:6.1f}s | {pred_str} | True: {true_label} | {correct} | {description}{timing}")
        time.sleep(DT * 0.8)
    results_file = os.path.join(RESULTS_DIR, 'full_demo_log.json')
    with open(results_file, 'w') as f: json.dump(results_log, f, indent=2)
    print(f"\nDemo complete! {len(results_log)} timesteps logged.")
    if results_log:
        correct = sum(1 for r in results_log if (r['prediction'] == 0 and r['true_label'] == 0) or (r['prediction'] > 0 and r['true_label'] > 0))
        total = len(results_log)
        print(f"   Detection accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    if ser: ser.close()

if __name__ == '__main__':
    mode = 'scripted'; port = None
    for arg in sys.argv[1:]:
        if arg == '--interactive': mode = 'interactive'
        elif arg.startswith('--port='): port = arg.split('=')[1]
        elif arg == '--help': print("Usage: python stream_to_esp32.py [--interactive] [--port=COM3]"); sys.exit(0)
    run_demo(mode=mode, port=port)
