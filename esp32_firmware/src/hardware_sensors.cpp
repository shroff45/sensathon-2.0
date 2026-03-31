#include "hardware_sensors.h"
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <TinyGPSPlus.h>
#include <HardwareSerial.h>

#define TRIG_PIN 18
#define ECHO_PIN 19

Adafruit_MPU6050 mpu;
TinyGPSPlus gps;
HardwareSerial SerialGPS(2);

static float current_heading_rate = 0.0;
static double last_heading = -1;
static unsigned long last_heading_time = 0;

void initHardwareSensors() {
    Serial.println("Initializing sensors...");
    
    // MPU6050
    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip - will use 0.0 fallback");
    } else {
        mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
        mpu.setGyroRange(MPU6050_RANGE_500_DEG);
        mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
        Serial.println("MPU6050 Initialized.");
    }

    // NEO-6M GPS (RX=16, TX=17)
    SerialGPS.begin(9600, SERIAL_8N1, 16, 17);
    Serial.println("NEO-6M GPS UART2 Initialized.");

    // HC-SR04
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    Serial.println("HC-SR04 Initialized.");
}

bool readHardwareSensors(float* lat_accel, float* lon_accel, float* yaw_rate, float* speed, float* hdg_rate, float* us_dist) {
    // 1. MPU6050 (Assume 0 if not connected)
    sensors_event_t a, g, temp;
    if (mpu.getEvent(&a, &g, &temp)) {
        *lat_accel = a.acceleration.y;  // Lateral depends on board mount
        *lon_accel = a.acceleration.x; 
        *yaw_rate  = g.gyro.z;         // Rad/s
    } else {
        *lat_accel = 0.0; *lon_accel = 0.0; *yaw_rate = 0.0;
    }

    // 2. NEO-6M GPS - Parse incoming chars
    while (SerialGPS.available() > 0) {
        gps.encode(SerialGPS.read());
    }

    // Speed in m/s
    if (gps.speed.isValid()) {
        *speed = gps.speed.mps();
    } else {
        *speed = 0.0; // Fallback indoors
    }

    // Heading Rate (rad/s)
    if (gps.course.isUpdated()) {
        double new_hdg = gps.course.deg();
        if (last_heading >= 0) {
            double diff = new_hdg - last_heading;
            if (diff > 180) diff -= 360;
            if (diff < -180) diff += 360;
            
            unsigned long dt = millis() - last_heading_time;
            if (dt > 0) {
                current_heading_rate = (diff * M_PI / 180.0) / (dt / 1000.0);
            }
        }
        last_heading = new_hdg;
        last_heading_time = millis();
    }
    *hdg_rate = current_heading_rate;

    // 3. HC-SR04 (Ultrasonic) 
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    
    // Timeout 30ms (~5 meters) to prevent blocking main loop
    long duration = pulseIn(ECHO_PIN, HIGH, 30000); 
    if (duration == 0) {
        *us_dist = 50.0; // Default max distance (no obstacle in 5m range)
    } else {
        *us_dist = (duration / 2.0) * 0.0343; // cm
        *us_dist = *us_dist / 100.0; // meters
    }
    
    return true;
}
