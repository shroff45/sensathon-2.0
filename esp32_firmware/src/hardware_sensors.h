#ifndef HARDWARE_SENSORS_H
#define HARDWARE_SENSORS_H

#include <Arduino.h>

// Initialize all hardware sensors (MPU6050, NEO6M, HCSR04)
void initHardwareSensors();

// Read all required features from hardware
// Returns true if sensors read successfully
bool readHardwareSensors(
    float* imu_lat_accel,
    float* imu_lon_accel,
    float* imu_yaw_rate,
    float* gps_speed,
    float* gps_heading_rate,
    float* ultrasonic_min
);

#endif // HARDWARE_SENSORS_H
