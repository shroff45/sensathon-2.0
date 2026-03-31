import unittest
import numpy as np
import copy
from stream_to_esp32 import format_packet

class TestStreamToESP32TDD(unittest.TestCase):
    
    def get_base_features(self):
        return {
            'gps_speed': 10.0, 'gps_heading_rate': 0.1, 'imu_lat_accel': 1.0, 
            'imu_yaw_rate': 0.1, 'imu_lon_accel': 0.5, 'ultrasonic_min': 5.0, 
            'ultrasonic_rate': 0.0,
            
            'can_wheel_speed': 10.0, 'can_steering_angle': 0.1, 'can_brake_pressure': 0.0,
            'can_throttle_pos': 20.0, 'can_msg_freq_dev': 1.0, 'can_id_entropy': 2.0, 
            'can_payload_anomaly': 0.01,
            
            'v2x_road_curvature': 0.01, 'v2x_speed_limit': 13.89, 'v2x_obstacle_dist': 5.0,
            'v2x_auth_score': 1.0, 'v2x_msg_frequency': 10.0
        }

    def test_format_packet_valid(self):
        features = self.get_base_features()
        packet = format_packet(features)
        
        # Format expects: S,x,x...|C,x,x...|V,x,x...\n
        self.assertTrue(packet.startswith("S,"))
        self.assertIn("|C,", packet)
        self.assertIn("|V,", packet)
        self.assertTrue(packet.endswith("\n"))
        self.assertLess(len(packet), 500)

    def test_format_packet_inf_replaced(self):
        features = self.get_base_features()
        features['gps_speed'] = float('inf')
        
        packet = format_packet(features)
        # Should replace inf with 0.0000
        # First element after S, is gps_speed
        self.assertTrue(packet.startswith("S,0.0000,"))

    def test_format_packet_nan_replaced(self):
        features = self.get_base_features()
        features['can_wheel_speed'] = float('nan')
        
        packet = format_packet(features)
        # can_wheel_speed is the 1st element exactly after |C,
        parts = packet.split("|C,")
        can_parts = parts[1].split(",")
        self.assertEqual(can_parts[0], "0.0000")

    def test_format_packet_oversized_raises_error(self):
        features = self.get_base_features()
        # Trigger an oversized string by inputting extraordinarily large floats 
        # But wait, np.isfinite allows extremely large floats up to 1e308. Stop before python formatting truncates to scientific notation if any.
        # Python's '{:.4f}' formatting of a very large float produces a huge string.
        # E.g., 1e100 formatted to .4f is 100 digits long.
        features['gps_speed'] = 1e200
        features['can_wheel_speed'] = 1e200
        features['v2x_speed_limit'] = 1e200
        # Each gives a 200+ character string, easily pushing packet past 500 chars

        with self.assertRaises(ValueError):
            format_packet(features)

if __name__ == '__main__':
    unittest.main()
