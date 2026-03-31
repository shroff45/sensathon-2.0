"""
COMPLETE SYNTHETIC DATASET GENERATOR
Cross-Layer Physics-Based IDS for Software-Defined Vehicles
Author: SENSEathon Team
"""

import numpy as np
import pandas as pd
import math
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

WHEELBASE = 2.7
STEERING_RATIO = 16.0
DT = 0.1
GRAVITY = 9.81

NOISE = {
    'gps_speed': 0.3, 'gps_heading_rate': 0.02, 'imu_lat_accel': 0.15,
    'imu_yaw_rate': 0.03, 'imu_lon_accel': 0.15, 'ultrasonic': 0.08,
    'can_wheel_speed': 0.2, 'can_steering': 0.008,
}

FEATURE_NAMES = [
    'gps_speed', 'gps_heading_rate', 'imu_lat_accel', 'imu_yaw_rate',
    'imu_lon_accel', 'ultrasonic_min', 'ultrasonic_rate',
    'can_wheel_speed', 'can_steering_angle', 'can_brake_pressure',
    'can_throttle_pos', 'can_msg_freq_dev', 'can_id_entropy', 'can_payload_anomaly',
    'v2x_road_curvature', 'v2x_speed_limit', 'v2x_obstacle_dist',
    'v2x_auth_score', 'v2x_msg_frequency',
    'xl_speed_consistency', 'xl_yaw_can_vs_gps', 'xl_yaw_can_vs_imu',
    'xl_lataccel_gps_vs_imu', 'xl_obstacle_ultra_vs_v2x', 'xl_curvature_3way',
    'xl_accel_consistency', 'xl_score_variance', 'xl_steering_jerk',
    'xl_heading_integral_diff'
]

class RealisticIMU:
    def __init__(self):
        self.gyro_bias = np.random.normal(0, 0.01)
        self.gyro_drift_rate = np.random.normal(0, 0.0001)
        self.accel_bias_x = np.random.normal(0, 0.05)
        self.accel_bias_y = np.random.normal(0, 0.05)
        self.time = 0
        self.prev_noise_gyro = 0
        self.prev_noise_accel_x = 0
        self.prev_noise_accel_y = 0
        self.correlation = 0.3

    def read_yaw_rate(self, true_val):
        self.time += DT
        current_bias = self.gyro_bias + self.gyro_drift_rate * self.time
        white_noise = np.random.normal(0, NOISE['imu_yaw_rate'])
        correlated_noise = self.correlation * self.prev_noise_gyro + (1 - self.correlation) * white_noise
        self.prev_noise_gyro = correlated_noise
        return true_val + current_bias + correlated_noise

    def read_lat_accel(self, true_val):
        white_noise = np.random.normal(0, NOISE['imu_lat_accel'])
        correlated = self.correlation * self.prev_noise_accel_y + (1 - self.correlation) * white_noise
        self.prev_noise_accel_y = correlated
        measured = true_val + self.accel_bias_y + correlated
        return np.clip(measured, -16 * GRAVITY, 16 * GRAVITY)

    def read_lon_accel(self, true_val):
        white_noise = np.random.normal(0, NOISE['imu_lon_accel'])
        correlated = self.correlation * self.prev_noise_accel_x + (1 - self.correlation) * white_noise
        self.prev_noise_accel_x = correlated
        measured = true_val + self.accel_bias_x + correlated
        return np.clip(measured, -16 * GRAVITY, 16 * GRAVITY)

class RealisticGPS:
    def __init__(self):
        self.multipath_active = False
        self.multipath_bias_speed = 0
        self.multipath_bias_heading = 0
        self.multipath_probability = 0.02
        self.multipath_clear_probability = 0.3

    def read_speed(self, true_val):
        self._update_multipath()
        noise = np.random.normal(0, NOISE['gps_speed'])
        measured = true_val + noise + self.multipath_bias_speed
        return max(0, measured)

    def read_heading_rate(self, true_val):
        noise = np.random.normal(0, NOISE['gps_heading_rate'])
        return true_val + noise + self.multipath_bias_heading * 0.1

    def _update_multipath(self):
        if not self.multipath_active:
            if np.random.random() < self.multipath_probability:
                self.multipath_active = True
                self.multipath_bias_speed = np.random.uniform(0.5, 2.5)
                self.multipath_bias_heading = np.random.uniform(-0.05, 0.05)
        else:
            if np.random.random() < self.multipath_clear_probability:
                self.multipath_active = False
                self.multipath_bias_speed = 0
                self.multipath_bias_heading = 0

@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 20.0
    steering_wheel_angle: float = 0.0
    acceleration: float = 0.0
    yaw_rate: float = 0.0
    lat_accel: float = 0.0
    lon_accel: float = 0.0
    obstacle_dist: float = 50.0

def bicycle_model_yaw(speed, steering_wheel_angle, wheelbase=WHEELBASE, steering_ratio=STEERING_RATIO):
    wheel_angle = steering_wheel_angle / steering_ratio
    if abs(speed) < 0.1:
        return 0.0
    return speed * math.tan(wheel_angle) / wheelbase

def update_vehicle(state, target_steering, target_speed, dt=DT):
    steering_rate = 0.3
    state.steering_wheel_angle += steering_rate * (target_steering - state.steering_wheel_angle)
    prev_speed = state.speed
    speed_rate = 0.2
    state.speed += speed_rate * (target_speed - state.speed)
    state.speed = max(0, state.speed)
    state.lon_accel = (state.speed - prev_speed) / dt
    state.yaw_rate = bicycle_model_yaw(state.speed, state.steering_wheel_angle)
    state.lat_accel = state.speed * state.yaw_rate
    state.heading += state.yaw_rate * dt
    state.x += state.speed * math.cos(state.heading) * dt
    state.y += state.speed * math.sin(state.heading) * dt
    if np.random.random() < 0.01:
        state.obstacle_dist = np.random.uniform(2.0, 50.0)
    elif state.obstacle_dist < 50:
        state.obstacle_dist += state.speed * dt * 0.1
        state.obstacle_dist = min(state.obstacle_dist, 50.0)
    return state

def scenario_highway_cruise(duration=200):
    state = VehicleState(speed=np.random.uniform(25, 35))
    steps = []
    for t in range(duration):
        state = update_vehicle(state, np.random.normal(0, 0.02), state.speed + np.random.normal(0, 0.3))
        steps.append(state.__dict__.copy())
    return steps

def scenario_urban(duration=200):
    state = VehicleState(speed=np.random.uniform(5, 15))
    steps = []
    for t in range(duration):
        if t % 50 < 10:
            target_steer = np.random.choice([-1, 1]) * np.random.uniform(0.3, 0.8)
        else:
            target_steer = np.random.normal(0, 0.05)
        state = update_vehicle(state, target_steer, np.random.uniform(3, 15))
        steps.append(state.__dict__.copy())
    return steps

def scenario_gentle_curve(duration=200):
    state = VehicleState(speed=np.random.uniform(15, 25))
    curve_dir = np.random.choice([-1, 1])
    curve_mag = np.random.uniform(0.2, 0.5)
    steps = []
    for t in range(duration):
        state = update_vehicle(state, curve_dir * curve_mag, state.speed + np.random.normal(0, 0.2))
        steps.append(state.__dict__.copy())
    return steps

def scenario_acceleration_braking(duration=200):
    state = VehicleState(speed=np.random.uniform(10, 20))
    steps = []
    for t in range(duration):
        if (t // 30) % 2 == 0:
            target_speed = state.speed + 3
        else:
            target_speed = max(2, state.speed - 3)
        state = update_vehicle(state, np.random.normal(0, 0.03), target_speed)
        steps.append(state.__dict__.copy())
    return steps

def scenario_lane_change(duration=200):
    state = VehicleState(speed=np.random.uniform(20, 30))
    steps = []
    for t in range(duration):
        if 40 < t < 55: target_steer = 0.3
        elif 55 < t < 70: target_steer = -0.3
        elif 70 < t < 80: target_steer = 0.1
        else: target_steer = np.random.normal(0, 0.02)
        state = update_vehicle(state, target_steer, state.speed + np.random.normal(0, 0.2))
        steps.append(state.__dict__.copy())
    return steps

def scenario_stop_and_go(duration=200):
    state = VehicleState(speed=np.random.uniform(5, 10))
    steps = []
    for t in range(duration):
        cycle = t % 40
        if cycle < 15: target_speed = np.random.uniform(8, 15)
        elif cycle < 25: target_speed = 1.0
        else: target_speed = np.random.uniform(3, 8)
        state = update_vehicle(state, np.random.normal(0, 0.05), target_speed)
        steps.append(state.__dict__.copy())
    return steps

def scenario_emergency_brake(duration=150):
    state = VehicleState(speed=np.random.uniform(25, 35))
    steps = []
    for t in range(duration):
        if 50 < t < 70:
            target_speed = max(0, state.speed - 8)
            state.obstacle_dist = max(2, 30 - (t - 50) * 1.5)
        else:
            target_speed = state.speed + np.random.normal(0, 0.3)
        state = update_vehicle(state, np.random.normal(0, 0.02), target_speed)
        steps.append(state.__dict__.copy())
    return steps

def scenario_sharp_turn(duration=200):
    state = VehicleState(speed=np.random.uniform(8, 15))
    direction = np.random.choice([-1, 1])
    steps = []
    for t in range(duration):
        if 60 < t < 100:
            target_steer = direction * np.random.uniform(0.6, 1.0)
            target_speed = max(5, state.speed - 1)
        else:
            target_steer = np.random.normal(0, 0.03)
            target_speed = np.random.uniform(10, 15)
        state = update_vehicle(state, target_steer, target_speed)
        steps.append(state.__dict__.copy())
    return steps

SCENARIO_GENERATORS = [
    scenario_highway_cruise, scenario_urban, scenario_gentle_curve,
    scenario_acceleration_braking, scenario_lane_change, scenario_stop_and_go,
    scenario_emergency_brake, scenario_sharp_turn,
]

def apply_ramp(true_val, fake_val, t, attack_start, ramp_duration):
    if t < attack_start: return true_val
    elapsed = t - attack_start
    if ramp_duration <= 0 or elapsed >= ramp_duration: blend = 1.0
    else: blend = elapsed / ramp_duration
    return true_val * (1 - blend) + fake_val * blend

def safe_normalized_diff(a, b, epsilon=0.001):
    """
    Compute normalized absolute difference.
    
    Returns value in [0, 2.0] range.
    Returns 0.0 when both values are near zero (below epsilon).
    Returns ~1.0 when one value is zero and other is not.
    Returns ~0.0 when values are close relative to their magnitude.
    
    Capped at 2.0 to prevent extreme outliers from dominating
    Random Forest splits.
    """
    if math.isnan(a) or math.isnan(b):
        return 0.0
        
    numerator = abs(a - b)
    denominator = max(abs(a), abs(b), epsilon)
    result = numerator / denominator
    return min(result, 2.0)


def compute_cross_layer_features(raw_features):
    """
    Compute 6 cross-layer physics consistency features.
    
    Handles edge cases:
    - Vehicle stopped (speed < 0.5 m/s): yaw features suppressed
    - Extreme steering angles: clamped to ±0.6 rad wheel angle
    - Division by zero: protected by safe_normalized_diff epsilon
    """
    gps_speed = raw_features['gps_speed']
    gps_heading_rate = raw_features['gps_heading_rate']
    imu_lat_accel = raw_features['imu_lat_accel']
    imu_yaw_rate = raw_features['imu_yaw_rate']
    ultrasonic_min = raw_features['ultrasonic_min']
    can_wheel_speed = raw_features['can_wheel_speed']
    can_steering = raw_features['can_steering_angle']
    v2x_curvature = raw_features['v2x_road_curvature']
    v2x_obstacle = raw_features['v2x_obstacle_dist']
    
    SPEED_THRESHOLD = 0.5  # m/s — below this, yaw features unreliable
    
    # Compute wheel angle with clamp to physical limits
    wheel_angle = can_steering / STEERING_RATIO
    wheel_angle = max(-0.6, min(0.6, wheel_angle))  # ±34° physical limit
    
    # Bicycle model yaw prediction — only valid above walking speed
    if abs(can_wheel_speed) > SPEED_THRESHOLD:
        yaw_from_can = can_wheel_speed * math.tan(wheel_angle) / WHEELBASE
    else:
        yaw_from_can = 0.0
    
    # F19: Speed consistency — always valid
    xl_speed = safe_normalized_diff(gps_speed, can_wheel_speed)
    
    # F20, F21: Yaw features — suppress when stopped
    if abs(can_wheel_speed) > SPEED_THRESHOLD and abs(gps_speed) > SPEED_THRESHOLD:
        xl_yaw_gps = safe_normalized_diff(yaw_from_can, gps_heading_rate)
        xl_yaw_imu = safe_normalized_diff(yaw_from_can, imu_yaw_rate)
    else:
        xl_yaw_gps = 0.0
        xl_yaw_imu = 0.0
    
    # F22: Lateral acceleration — suppress when stopped
    if abs(gps_speed) > SPEED_THRESHOLD:
        lat_accel_from_gps = gps_speed * gps_heading_rate
        xl_lat_accel = safe_normalized_diff(lat_accel_from_gps, imu_lat_accel)
    else:
        xl_lat_accel = 0.0
    
    # F23: Obstacle — always valid
    xl_obstacle = safe_normalized_diff(ultrasonic_min, v2x_obstacle)
    
    # F24: Curvature 3-way — suppress when stopped
    if abs(can_wheel_speed) > SPEED_THRESHOLD and abs(gps_speed) > SPEED_THRESHOLD:
        kappa_v2x = v2x_curvature
        kappa_can = math.tan(wheel_angle) / WHEELBASE
        kappa_gps = gps_heading_rate / gps_speed
        
        d1 = safe_normalized_diff(kappa_v2x, abs(kappa_can))
        d2 = safe_normalized_diff(kappa_v2x, abs(kappa_gps))
        xl_curvature = (d1 + d2) / 2
    else:
        xl_curvature = 0.0
    
    return {
        'xl_speed_consistency': xl_speed,
        'xl_yaw_can_vs_gps': xl_yaw_gps,
        'xl_yaw_can_vs_imu': xl_yaw_imu,
        'xl_lataccel_gps_vs_imu': xl_lat_accel,
        'xl_obstacle_ultra_vs_v2x': xl_obstacle,
        'xl_curvature_3way': xl_curvature,
    }


def compute_temporal_features(history, window=10):
    """
    Compute 4 temporal features from sliding window of past feature vectors.
    
    Edge cases handled:
    - Insufficient history (< window): returns zeros
    - Missing keys in history dicts: uses .get() with defaults
    - NaN/Inf values: caught by safe_normalized_diff cap
    """
    if len(history) < window:
        return {
            'xl_accel_consistency': 0.0,
            'xl_score_variance': 0.0,
            'xl_steering_jerk': 0.0,
            'xl_heading_integral_diff': 0.0,
        }
    
    recent = history[-window:]
    
    # F25: Acceleration consistency
    speeds_gps = [h.get('gps_speed', 0) for h in recent]
    speed_delta = abs(speeds_gps[-1] - speeds_gps[0]) / (window * DT)
    imu_lon_vals = [h.get('imu_lon_accel', 0) for h in recent]
    avg_imu_accel = sum(imu_lon_vals) / len(imu_lon_vals)
    f25 = safe_normalized_diff(speed_delta, abs(avg_imu_accel), 0.1)
    
    # F26: Cross-layer score variance (sudden spike detection)
    yaw_imu_scores = [h.get('xl_yaw_can_vs_imu', 0) for h in recent]
    f26 = max(yaw_imu_scores) - min(yaw_imu_scores)
    f26 = min(f26, 2.0)  # Cap
    
    # F27: Steering jerk (smoothness)
    steerings = [h.get('can_steering_angle', 0) for h in recent]
    diffs = [abs(steerings[i+1] - steerings[i]) for i in range(len(steerings)-1)]
    f27 = max(diffs) if diffs else 0.0
    f27 = min(f27, 2.0)  # Cap
    
    # F28: Heading integral consistency
    hr_vals = [h.get('gps_heading_rate', 0) for h in recent]
    imu_yr_vals = [h.get('imu_yaw_rate', 0) for h in recent]
    integrated_gps = sum(v * DT for v in hr_vals)
    integrated_imu = sum(v * DT for v in imu_yr_vals)
    f28 = safe_normalized_diff(integrated_gps, integrated_imu, 0.001)
    
    return {
        'xl_accel_consistency': f25,
        'xl_score_variance': f26,
        'xl_steering_jerk': f27,
        'xl_heading_integral_diff': f28,
    }

def compute_can_entropy(msg_ids=None):
    if msg_ids is None:
        n_ids = np.random.randint(15, 25)
        probs = np.random.dirichlet(np.ones(n_ids))
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    return 0.0

class AttackGenerator:
    @staticmethod
    def get_ramp_duration(onset_type='random'):
        if onset_type == 'instant': return 0
        elif onset_type == 'fast': return np.random.randint(3, 8)
        elif onset_type == 'slow': return np.random.randint(15, 30)
        else: return np.random.choice([0, 0, 5, 10, 20])

    @staticmethod
    def attack_single_gps_speed(features, t, attack_start, ramp):
        fake_speed = features['gps_speed'] + np.random.uniform(5, 15)
        features['gps_speed'] = apply_ramp(features['gps_speed'], fake_speed, t, attack_start, ramp)
        return features, 'attack_single_gps_speed'

    @staticmethod
    def attack_single_gps_heading(features, t, attack_start, ramp):
        fake_hr = features['gps_heading_rate'] + np.random.uniform(0.1, 0.5)
        features['gps_heading_rate'] = apply_ramp(features['gps_heading_rate'], fake_hr, t, attack_start, ramp)
        return features, 'attack_single_gps_heading'

    @staticmethod
    def attack_single_can_steering(features, t, attack_start, ramp):
        fake_steer = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])
        features['can_steering_angle'] = apply_ramp(features['can_steering_angle'], fake_steer, t, attack_start, ramp)
        return features, 'attack_single_can_steering'

    @staticmethod
    def attack_single_can_speed(features, t, attack_start, ramp):
        fake_speed = features['can_wheel_speed'] + np.random.uniform(5, 15)
        features['can_wheel_speed'] = apply_ramp(features['can_wheel_speed'], fake_speed, t, attack_start, ramp)
        return features, 'attack_single_can_speed'

    @staticmethod
    def attack_single_v2x_curvature(features, t, attack_start, ramp):
        fake_curv = np.random.uniform(0.02, 0.06)
        features['v2x_road_curvature'] = apply_ramp(features['v2x_road_curvature'], fake_curv, t, attack_start, ramp)
        return features, 'attack_single_v2x_curvature'

    @staticmethod
    def attack_single_v2x_obstacle(features, t, attack_start, ramp):
        fake_dist = np.random.uniform(2.0, 10.0)
        features['v2x_obstacle_dist'] = apply_ramp(features['v2x_obstacle_dist'], fake_dist, t, attack_start, ramp)
        return features, 'attack_single_v2x_obstacle'

    @staticmethod
    def attack_single_can_dos(features, t, attack_start, ramp):
        if t >= attack_start:
            features['can_id_entropy'] = np.random.uniform(0.3, 1.2)
            features['can_msg_freq_dev'] = np.random.uniform(50, 200)
            features['can_payload_anomaly'] = np.random.uniform(0.5, 1.0)
        return features, 'attack_single_can_dos'

    @staticmethod
    def attack_coord_can_v2x(features, t, attack_start, ramp):
        fake_steer = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])
        fake_wheel = fake_steer / STEERING_RATIO
        fake_curvature = math.tan(fake_wheel) / WHEELBASE
        features['can_steering_angle'] = apply_ramp(features['can_steering_angle'], fake_steer, t, attack_start, ramp)
        features['v2x_road_curvature'] = apply_ramp(features['v2x_road_curvature'], abs(fake_curvature), t, attack_start, ramp)
        return features, 'attack_coord_can_v2x'

    @staticmethod
    def attack_coord_gps_can(features, t, attack_start, ramp):
        speed_offset = np.random.uniform(5, 15)
        fake_speed = features['gps_speed'] + speed_offset
        features['gps_speed'] = apply_ramp(features['gps_speed'], fake_speed, t, attack_start, ramp)
        features['can_wheel_speed'] = apply_ramp(features['can_wheel_speed'], fake_speed, t, attack_start, ramp)
        return features, 'attack_coord_gps_can'

    @staticmethod
    def attack_coord_gps_v2x(features, t, attack_start, ramp):
        fake_heading_rate = np.random.uniform(0.1, 0.4) * np.random.choice([-1, 1])
        fake_curvature = abs(fake_heading_rate / max(abs(features['gps_speed']), 1.0))
        features['gps_heading_rate'] = apply_ramp(features['gps_heading_rate'], fake_heading_rate, t, attack_start, ramp)
        features['v2x_road_curvature'] = apply_ramp(features['v2x_road_curvature'], fake_curvature, t, attack_start, ramp)
        return features, 'attack_coord_gps_v2x'

    @staticmethod
    def attack_coord_all_three(features, t, attack_start, ramp):
        fake_steer = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])
        fake_wheel = fake_steer / STEERING_RATIO
        fake_yaw = features['can_wheel_speed'] * math.tan(fake_wheel) / WHEELBASE
        fake_curvature = abs(math.tan(fake_wheel) / WHEELBASE)
        features['can_steering_angle'] = apply_ramp(features['can_steering_angle'], fake_steer, t, attack_start, ramp)
        features['v2x_road_curvature'] = apply_ramp(features['v2x_road_curvature'], fake_curvature, t, attack_start, ramp)
        features['gps_heading_rate'] = apply_ramp(features['gps_heading_rate'], fake_yaw, t, attack_start, ramp)
        return features, 'attack_coord_all_three'

    @staticmethod
    def attack_coord_v2x_imu(features, t, attack_start, ramp):
        fake_curvature = np.random.uniform(0.02, 0.06)
        features['v2x_road_curvature'] = apply_ramp(features['v2x_road_curvature'], fake_curvature, t, attack_start, ramp)
        if t >= attack_start:
            features['imu_yaw_rate'] += np.random.normal(0, 0.15)
            features['imu_lat_accel'] += np.random.normal(0, 0.5)
        return features, 'attack_coord_v2x_imu'

    @staticmethod
    def attack_coord_can_imu(features, t, attack_start, ramp):
        fake_steer = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])
        fake_wheel = fake_steer / STEERING_RATIO
        fake_yaw = features['can_wheel_speed'] * math.tan(fake_wheel) / WHEELBASE
        features['can_steering_angle'] = apply_ramp(features['can_steering_angle'], fake_steer, t, attack_start, ramp)
        features['imu_yaw_rate'] = apply_ramp(features['imu_yaw_rate'], fake_yaw, t, attack_start, ramp)
        features['imu_lat_accel'] = apply_ramp(features['imu_lat_accel'], features['gps_speed'] * fake_yaw, t, attack_start, ramp)
        return features, 'attack_coord_can_imu'

    @staticmethod
    def attack_coord_speed_all(features, t, attack_start, ramp):
        speed_boost = np.random.uniform(8, 20)
        fake_speed = features['gps_speed'] + speed_boost
        features['gps_speed'] = apply_ramp(features['gps_speed'], fake_speed, t, attack_start, ramp)
        features['can_wheel_speed'] = apply_ramp(features['can_wheel_speed'], fake_speed, t, attack_start, ramp)
        features['v2x_speed_limit'] = apply_ramp(features['v2x_speed_limit'], fake_speed + 5, t, attack_start, ramp)
        return features, 'attack_coord_speed_all'

SINGLE_ATTACKS = [
    AttackGenerator.attack_single_gps_speed, AttackGenerator.attack_single_gps_heading,
    AttackGenerator.attack_single_can_steering, AttackGenerator.attack_single_can_speed,
    AttackGenerator.attack_single_v2x_curvature, AttackGenerator.attack_single_v2x_obstacle,
    AttackGenerator.attack_single_can_dos,
]
COORD_ATTACKS_TRAIN = [
    AttackGenerator.attack_coord_can_v2x, AttackGenerator.attack_coord_gps_can,
    AttackGenerator.attack_coord_gps_v2x, AttackGenerator.attack_coord_all_three,
    AttackGenerator.attack_coord_v2x_imu,
]
COORD_ATTACKS_UNSEEN = [AttackGenerator.attack_coord_can_imu, AttackGenerator.attack_coord_speed_all]

def generate_scenario_data(scenario_fn, attack_fn=None, attack_label=0, onset_type='random', scenario_id=0):
    imu = RealisticIMU()
    gps = RealisticGPS()
    steps = scenario_fn()
    duration = len(steps)
    if attack_fn is not None:
        attack_start = np.random.randint(duration // 4, duration // 2)
        ramp = AttackGenerator.get_ramp_duration(onset_type)
    else:
        attack_start = duration + 1
        ramp = 0
    prev_ultrasonic = 50.0
    feature_history = []
    rows = []
    for t, state in enumerate(steps):
        true_speed = state['speed']
        true_yaw = state['yaw_rate']
        true_lat_accel = state['lat_accel']
        true_lon_accel = state['lon_accel']
        true_steering = state['steering_wheel_angle']
        true_obstacle = state['obstacle_dist']
        raw = {
            'gps_speed': gps.read_speed(true_speed),
            'gps_heading_rate': gps.read_heading_rate(true_yaw),
            'imu_lat_accel': imu.read_lat_accel(true_lat_accel),
            'imu_yaw_rate': imu.read_yaw_rate(true_yaw),
            'imu_lon_accel': imu.read_lon_accel(true_lon_accel),
            'ultrasonic_min': max(0.02, true_obstacle + np.random.normal(0, NOISE['ultrasonic'])),
            'ultrasonic_rate': 0.0,
            'can_wheel_speed': true_speed + np.random.normal(0, NOISE['can_wheel_speed']),
            'can_steering_angle': true_steering + np.random.normal(0, NOISE['can_steering']),
            'can_brake_pressure': max(0, -true_lon_accel * 15 + np.random.normal(0, 2)),
            'can_throttle_pos': max(0, min(100, true_lon_accel * 20 + 30 + np.random.normal(0, 3))),
            'can_msg_freq_dev': abs(np.random.normal(0, 3)),
            'can_id_entropy': compute_can_entropy(),
            'can_payload_anomaly': abs(np.random.normal(0, 0.05)),
            'v2x_road_curvature': abs(true_yaw / max(abs(true_speed), 1.0)) + np.random.normal(0, 0.002),
            'v2x_speed_limit': np.random.choice([8.33, 11.11, 13.89, 16.67, 22.22, 27.78, 33.33]),
            'v2x_obstacle_dist': true_obstacle + np.random.normal(0, 1.0),
            'v2x_auth_score': np.random.choice([0.9, 0.95, 1.0], p=[0.1, 0.2, 0.7]),
            'v2x_msg_frequency': 10.0 + np.random.normal(0, 1.0),
        }
        raw['ultrasonic_rate'] = (raw['ultrasonic_min'] - prev_ultrasonic) / DT
        prev_ultrasonic = raw['ultrasonic_min']
        attack_name = 'normal'
        if attack_fn is not None and t >= attack_start:
            raw, attack_name = attack_fn(raw, t, attack_start, ramp)
        if t < attack_start or attack_fn is None: label = 0
        else: label = attack_label
        xl_features = compute_cross_layer_features(raw)
        raw.update(xl_features)
        temporal = compute_temporal_features(feature_history, window=10)
        raw.update(temporal)
        feature_history.append(raw.copy())
        row = {name: raw[name] for name in FEATURE_NAMES}
        row['label'] = label
        row['attack_name'] = attack_name if label > 0 else 'normal'
        row['scenario_id'] = scenario_id
        row['timestep'] = t
        rows.append(row)
    return rows

def generate_full_dataset(n_scenarios=1500, seed=42, include_unseen=False):
    np.random.seed(seed)
    all_rows = []
    scenario_id = 0
    n_normal = int(n_scenarios * 0.35)
    n_single = int(n_scenarios * 0.30)
    n_coord = n_scenarios - n_normal - n_single
    onset_types = ['instant', 'fast', 'slow', 'random']
    print(f"Generating {n_normal} normal scenarios...")
    for i in range(n_normal):
        scenario_fn = np.random.choice(SCENARIO_GENERATORS)
        rows = generate_scenario_data(scenario_fn, attack_fn=None, attack_label=0, scenario_id=scenario_id)
        all_rows.extend(rows)
        scenario_id += 1
        if (i + 1) % 100 == 0: print(f"  Normal: {i+1}/{n_normal}")
    print(f"Generating {n_single} single-attack scenarios...")
    for i in range(n_single):
        scenario_fn = np.random.choice(SCENARIO_GENERATORS)
        attack_fn = np.random.choice(SINGLE_ATTACKS)
        onset = np.random.choice(onset_types)
        rows = generate_scenario_data(scenario_fn, attack_fn=attack_fn, attack_label=1, onset_type=onset, scenario_id=scenario_id)
        all_rows.extend(rows)
        scenario_id += 1
        if (i + 1) % 100 == 0: print(f"  Single: {i+1}/{n_single}")
    print(f"Generating {n_coord} coordinated-attack scenarios...")
    for i in range(n_coord):
        scenario_fn = np.random.choice(SCENARIO_GENERATORS)
        if include_unseen: attack_fn = np.random.choice(COORD_ATTACKS_TRAIN + COORD_ATTACKS_UNSEEN)
        else: attack_fn = np.random.choice(COORD_ATTACKS_TRAIN)
        onset = np.random.choice(onset_types)
        rows = generate_scenario_data(scenario_fn, attack_fn=attack_fn, attack_label=2, onset_type=onset, scenario_id=scenario_id)
        all_rows.extend(rows)
        scenario_id += 1
        if (i + 1) % 100 == 0: print(f"  Coordinated: {i+1}/{n_coord}")
    df = pd.DataFrame(all_rows)
    feature_cols = FEATURE_NAMES + ['label', 'attack_name', 'scenario_id', 'timestep']
    df = df[feature_cols]
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    print("=" * 60)
    print("GENERATING TRAINING DATASET")
    print("=" * 60)
    train_df = generate_full_dataset(n_scenarios=1500, seed=42, include_unseen=False)
    train_df.to_csv('data/train_dataset.csv', index=False)
    print(f"\nTraining set: {len(train_df)} rows")
    print(f"Class distribution:\n{train_df['label'].value_counts().sort_index()}")
    print(f"Attack types:\n{train_df['attack_name'].value_counts()}")
    print("\n" + "=" * 60)
    print("GENERATING TEST DATASET (includes unseen attacks)")
    print("=" * 60)
    test_df = generate_full_dataset(n_scenarios=300, seed=99, include_unseen=True)
    test_df.to_csv('data/test_dataset.csv', index=False)
    print(f"\nTest set: {len(test_df)} rows")
    print(f"Class distribution:\n{test_df['label'].value_counts().sort_index()}")
    print(f"Attack types:\n{test_df['attack_name'].value_counts()}")
    print("\n✅ Dataset generation complete!")
    print(f"Features per sample: {len(FEATURE_NAMES)}")
    print(f"Feature names: {FEATURE_NAMES}")
