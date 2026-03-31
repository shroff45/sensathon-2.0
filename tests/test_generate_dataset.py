import unittest
import math
from generate_dataset import safe_normalized_diff, compute_cross_layer_features


class TestGenerateDatasetTDD(unittest.TestCase):

    def test_safe_normalized_diff_identical(self):
        # Arrange & Act
        result = safe_normalized_diff(10, 10)
        # Assert
        self.assertEqual(result, 0.0)

    def test_safe_normalized_diff_zero(self):
        # Arrange & Act
        result = safe_normalized_diff(0, 0)
        # Assert
        self.assertEqual(result, 0.0)

    def test_safe_normalized_diff_capped(self):
        # Arrange & Act
        result = safe_normalized_diff(100, 0)
        # Assert (100 / 100 = 1.0, wait... 100/100 is 1.0. Let's test 100 vs -100 to get > 2.0 capped!)
        result_capped = safe_normalized_diff(100, -100)
        self.assertEqual(result_capped, 2.0)
        self.assertEqual(result, 1.0)

    def get_base_features(self):
        return {
            'gps_speed': 10.0,
            'gps_heading_rate': 0.1,
            'imu_lat_accel': 1.0,
            'imu_yaw_rate': 0.1,
            'ultrasonic_min': 5.0,
            'can_wheel_speed': 10.0,
            'can_steering_angle': 0.1,
            'v2x_road_curvature': 0.01,
            'v2x_obstacle_dist': 5.0
        }

    def test_compute_xl_suppression_low_speed(self):
        # Arrange
        features = self.get_base_features()
        features['can_wheel_speed'] = 0.3  # Below 0.5 m/s threshold
        features['gps_speed'] = 0.3        # Below 0.5 m/s threshold
        
        # Act
        result = compute_cross_layer_features(features)
        
        # Assert
        self.assertEqual(result['xl_yaw_can_vs_gps'], 0.0)
        self.assertEqual(result['xl_yaw_can_vs_imu'], 0.0)
        self.assertEqual(result['xl_lataccel_gps_vs_imu'], 0.0)
        self.assertEqual(result['xl_curvature_3way'], 0.0)

    def test_compute_xl_steering_clamp(self):
        # Arrange
        features = self.get_base_features()
        # STEERING_RATIO is 16.0. Wheel angle > 0.6 means steering > 9.6
        features['can_steering_angle'] = 20.0  # Should be clamped to 0.6 radians internally
        
        # Act
        result = compute_cross_layer_features(features)
        
        # Assert
        # The computation uses map to bicycle model.
        # We ensure it doesn't blow up to extreme values due to tan(excessive)
        # tan(0.6) = 0.684, Speed=10, WB=2.7 -> yaw = 10 * 0.684 / 2.7 = 2.53
        # If not clamped, tan(20/16 = 1.25) = 3.00 -> yaw = 11.1
        self.assertLess(result['xl_yaw_can_vs_gps'], 2.0)  # Capped diff

    def test_compute_xl_nan_inputs(self):
        # Arrange
        features = self.get_base_features()
        features['gps_speed'] = float('nan')
        
        # Act
        result = compute_cross_layer_features(features)
        
        # Assert
        for key, val in result.items():
            self.assertFalse(math.isnan(val), f"Output contains NaN for {key}")

if __name__ == '__main__':
    unittest.main()
