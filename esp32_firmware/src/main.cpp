#include <Arduino.h>

/*
 * CROSS-LAYER PHYSICS-BASED IDS - ESP32 FIRMWARE
 * Receives 19 raw sensor features via serial
 * Computes 6 cross-layer physics features + 4 temporal features
 * Runs 20-tree Random Forest (29 features)
 * Controls 3 LEDs (Green/Yellow/Red)
 * Returns classification + timing + feature values
 *
 * Pin connections:
 *   GPIO 25 -> Green LED (+ 220 ohm resistor -> GND)
 *   GPIO 26 -> Yellow LED (+ 220 ohm resistor -> GND)
 *   GPIO 27 -> Red LED (+ 220 ohm resistor -> GND)
 */

#include "rf_model.h"
#include "hardware_sensors.h"
#include <math.h>

#define LED_GREEN  25
#define LED_YELLOW 26
#define LED_RED    27

#define SERIAL_BAUD 115200
#define SERIAL_TIMEOUT_MS 500
#define WHEELBASE 2.7f
#define STEERING_RATIO 16.0f
#define DT 0.1f
#define EPSILON 0.001f

#define WINDOW_SIZE 10
#define NUM_RAW_FEATURES 19
#define NUM_TOTAL_FEATURES 29
#define CONFIRM_COUNT 3

const float FEATURE_MIN[NUM_RAW_FEATURES] = {
    -1.0f, -2.0f, -160.0f, -3.0f, -160.0f, 0.0f, -50.0f,
    -1.0f, -1.5f, -5.0f, -5.0f, -10.0f, 0.0f, -0.5f,
    -0.01f, 0.0f, 0.0f, 0.0f, 0.0f,
};

const float FEATURE_MAX[NUM_RAW_FEATURES] = {
    80.0f, 2.0f, 160.0f, 3.0f, 160.0f, 100.0f, 50.0f,
    80.0f, 1.5f, 200.0f, 105.0f, 300.0f, 8.0f, 5.0f,
    0.2f, 50.0f, 500.0f, 1.1f, 100.0f,
};

float features[NUM_TOTAL_FEATURES];
float feature_history[WINDOW_SIZE][NUM_TOTAL_FEATURES];
int history_count = 0;
int history_idx = 0;

int prediction_buffer[CONFIRM_COUNT];
int pred_buf_idx = 0;

unsigned long lastReceiveTime = 0;
char serialBuffer[600];
int bufferPos = 0;

enum VehicleMode { MODE_NORMAL, MODE_DEGRADED, MODE_SAFE_STOP };
int consecutive_alerts = 0;

void setup() {
    initHardwareSensors();
    Serial.begin(SERIAL_BAUD);
    
    // Memory check
    Serial.print("FREE_HEAP,");
    Serial.println(ESP.getFreeHeap());
    
    if (ESP.getFreeHeap() < 50000) {
        Serial.println("WARNING,LOW_MEMORY");
    }
    
    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_YELLOW, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    
    // Startup animation
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_GREEN, HIGH); delay(100);
        digitalWrite(LED_GREEN, LOW);
        digitalWrite(LED_YELLOW, HIGH); delay(100);
        digitalWrite(LED_YELLOW, LOW);
        digitalWrite(LED_RED, HIGH); delay(100);
        digitalWrite(LED_RED, LOW);
    }
    
    for (int i = 0; i < CONFIRM_COUNT; i++) {
        prediction_buffer[i] = 0;
    }
    
    memset(feature_history, 0, sizeof(feature_history));
    
    digitalWrite(LED_GREEN, HIGH);
    
    Serial.println("READY,CrossLayer_IDS_v2.0");
}

float safe_norm_diff(float a, float b) {
    float diff = fabsf(a - b);
    float maxval = fmaxf(fabsf(a), fabsf(b));
    maxval = fmaxf(maxval, EPSILON);
    float result = diff / maxval;
    // Cap at 2.0 to match training data range
    if (result > 2.0f) result = 2.0f;
    return result;
}

void setLEDs(int prediction) {
    digitalWrite(LED_GREEN, prediction == 0 ? HIGH : LOW);
    digitalWrite(LED_YELLOW, prediction == 1 ? HIGH : LOW);
    digitalWrite(LED_RED, prediction == 2 ? HIGH : LOW);
}

int get_confirmed_prediction(int new_pred) {
    prediction_buffer[pred_buf_idx] = new_pred;
    pred_buf_idx = (pred_buf_idx + 1) % CONFIRM_COUNT;
    int votes[3] = {0, 0, 0};
    for (int i = 0; i < CONFIRM_COUNT; i++) {
        if (prediction_buffer[i] >= 0 && prediction_buffer[i] <= 2)
            votes[prediction_buffer[i]]++;
    }
    int best = 0;
    for (int c = 1; c < 3; c++) {
        if (votes[c] > votes[best]) best = c;
    }
    return best;
}

VehicleMode determine_response(int prediction, int consecutive) {
    if (prediction == 0) return MODE_NORMAL;
    if (prediction == 1) return MODE_DEGRADED;
    if (prediction == 2 && consecutive >= 3) return MODE_SAFE_STOP;
    return MODE_DEGRADED;
}

void compute_cross_layer_features() {
    float gps_speed = features[0];
    float gps_hr = features[1];
    float imu_lat = features[2];
    float imu_yaw = features[3];
    float ultra_min = features[5];
    float can_speed = features[7];
    float can_steer = features[8];
    float v2x_curv = features[14];
    float v2x_obs = features[16];
    
    const float SPEED_THRESH = 0.5f;
    
    // Wheel angle with physical clamp
    float wheel_angle = can_steer / STEERING_RATIO;
    if (wheel_angle > 0.6f) wheel_angle = 0.6f;
    if (wheel_angle < -0.6f) wheel_angle = -0.6f;
    
    // Bicycle model — only above walking speed
    float yaw_from_can = 0.0f;
    if (fabsf(can_speed) > SPEED_THRESH) {
        yaw_from_can = can_speed * tanf(wheel_angle) / WHEELBASE;
    }
    
    // F19: Speed consistency — always valid
    features[19] = safe_norm_diff(gps_speed, can_speed);
    
    // F20, F21: Yaw features — suppress when stopped
    if (fabsf(can_speed) > SPEED_THRESH && fabsf(gps_speed) > SPEED_THRESH) {
        features[20] = safe_norm_diff(yaw_from_can, gps_hr);
        features[21] = safe_norm_diff(yaw_from_can, imu_yaw);
    } else {
        features[20] = 0.0f;
        features[21] = 0.0f;
    }
    
    // F22: Lateral acceleration — suppress when stopped
    if (fabsf(gps_speed) > SPEED_THRESH) {
        float lat_accel_gps = gps_speed * gps_hr;
        features[22] = safe_norm_diff(lat_accel_gps, imu_lat);
    } else {
        features[22] = 0.0f;
    }
    
    // F23: Obstacle — always valid
    features[23] = safe_norm_diff(ultra_min, v2x_obs);
    
    // F24: Curvature 3-way — suppress when stopped
    if (fabsf(can_speed) > SPEED_THRESH && fabsf(gps_speed) > SPEED_THRESH) {
        float kappa_can = tanf(wheel_angle) / WHEELBASE;
        float kappa_gps = gps_hr / gps_speed;
        float d1 = safe_norm_diff(v2x_curv, fabsf(kappa_can));
        float d2 = safe_norm_diff(v2x_curv, fabsf(kappa_gps));
        features[24] = (d1 + d2) / 2.0f;
    } else {
        features[24] = 0.0f;
    }
}

void compute_temporal_features() {
    if (history_count < WINDOW_SIZE) {
        features[25] = 0.0f; features[26] = 0.0f;
        features[27] = 0.0f; features[28] = 0.0f;
        return;
    }
    int oldest = (history_idx + 1) % WINDOW_SIZE;
    int newest = history_idx;
    float speed_old = feature_history[oldest][0];
    float speed_new = feature_history[newest][0];
    float speed_delta = fabsf(speed_new - speed_old) / (WINDOW_SIZE * DT);
    float avg_imu_lon = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) avg_imu_lon += feature_history[i][4];
    avg_imu_lon /= WINDOW_SIZE;
    features[25] = safe_norm_diff(speed_delta, fabsf(avg_imu_lon));
    float min_xl = 999.0f, max_xl = -999.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        float val = feature_history[i][21];
        if (val < min_xl) min_xl = val;
        if (val > max_xl) max_xl = val;
    }
    features[26] = max_xl - min_xl;
    float max_jerk = 0.0f;
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        int curr = (oldest + i) % WINDOW_SIZE;
        int next = (oldest + i + 1) % WINDOW_SIZE;
        float jerk = fabsf(feature_history[next][8] - feature_history[curr][8]);
        if (jerk > max_jerk) max_jerk = jerk;
    }
    features[27] = max_jerk;
    float sum_gps_hr = 0.0f, sum_imu_yr = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        sum_gps_hr += feature_history[i][1] * DT;
        sum_imu_yr += feature_history[i][3] * DT;
    }
    features[28] = safe_norm_diff(sum_gps_hr, sum_imu_yr);
}

void store_in_history() {
    history_idx = (history_idx + 1) % WINDOW_SIZE;
    for (int i = 0; i < NUM_TOTAL_FEATURES; i++)
        feature_history[history_idx][i] = features[i];
    if (history_count < WINDOW_SIZE) history_count++;
}

int predict_single_tree(int tree_idx) {
    int node = 0;
    while (tree_left[tree_idx][node] != -1) {
        int feat_idx = tree_features[tree_idx][node];
        float threshold = tree_thresholds[tree_idx][node];
        if (features[feat_idx] <= threshold)
            node = tree_left[tree_idx][node];
        else
            node = tree_right[tree_idx][node];
    }
    return tree_classes[tree_idx][node];
}

int predict_forest() {
    int votes[NUM_CLASSES] = {0};
    for (int t = 0; t < NUM_TREES; t++) {
        int pred = predict_single_tree(t);
        if (pred >= 0 && pred < NUM_CLASSES) votes[pred]++;
    }
    int best_class = 0;
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (votes[c] > votes[best_class]) best_class = c;
    }
    return best_class;
}

bool parse_packet(char* packet) {
    char* ptr = packet;
    int feat_idx = 0;
    
    if (*ptr != 'S') return false;
    ptr += 2;
    
    while (feat_idx < NUM_RAW_FEATURES && *ptr != '\0') {
        // Skip section markers
        while (*ptr == '|' || *ptr == 'C' || *ptr == 'V' || *ptr == ',') {
            ptr++;
        }
        if (*ptr == '\0') break;
        
        // Check for empty field (consecutive delimiters)
        if (*ptr == ',' || *ptr == '|') {
            features[feat_idx] = 0.0f;
            feat_idx++;
            continue;
        }
        
        char* end;
        float val = strtof(ptr, &end);
        
        if (end == ptr) {
            ptr++;
            continue;
        }
        
        // Check for inf/nan
        if (isinf(val) || isnan(val)) {
            val = 0.0f;
        }
        
        // Clamp to valid range
        if (val < FEATURE_MIN[feat_idx]) {
            val = FEATURE_MIN[feat_idx];
        } else if (val > FEATURE_MAX[feat_idx]) {
            val = FEATURE_MAX[feat_idx];
        }
        
        features[feat_idx] = val;
        feat_idx++;
        ptr = end;
        
        if (*ptr == ',') ptr++;
    }
    
    if (feat_idx != NUM_RAW_FEATURES) {
        Serial.print("R,ERR,INCOMPLETE,");
        Serial.println(feat_idx);
        return false;
    }
    
    return true;
}

void loop() {
    static unsigned long lastSensorSendTime = 0;
    if (millis() - lastSensorSendTime >= 100) {
        lastSensorSendTime = millis();
        float lat, lon, yaw, spd, hdg, dist;
        readHardwareSensors(&lat, &lon, &yaw, &spd, &hdg, &dist);
        Serial.print("SENSOR,");
        Serial.print(lat, 4); Serial.print(",");
        Serial.print(lon, 4); Serial.print(",");
        Serial.print(yaw, 4); Serial.print(",");
        Serial.print(spd, 4); Serial.print(",");
        Serial.print(hdg, 4); Serial.print(",");
        Serial.println(dist, 4);
    }
    while (Serial.available()) {
        char c = Serial.read();
        lastReceiveTime = millis();
        
        if (c == '\n' || c == '\r') {
            if (bufferPos > 0) {
                serialBuffer[bufferPos] = '\0';
                
                if (parse_packet(serialBuffer)) {
                    unsigned long t0 = micros();
                    
                    compute_cross_layer_features();
                    compute_temporal_features();
                    store_in_history();
                    
                    unsigned long t1 = micros();
                    unsigned long feature_time = t1 - t0;
                    
                    int raw_prediction = predict_forest();
                    
                    unsigned long t2 = micros();
                    unsigned long inference_time = t2 - t1;
                    
                    int confirmed = get_confirmed_prediction(raw_prediction);
                    
                    if (confirmed > 0) {
                        consecutive_alerts++;
                    } else {
                        consecutive_alerts = 0;
                    }
                    VehicleMode mode = determine_response(confirmed, consecutive_alerts);
                    
                    setLEDs(confirmed);
                    
                    Serial.print("R,");
                    Serial.print(confirmed);
                    Serial.print(",");
                    Serial.print(raw_prediction);
                    Serial.print(",");
                    Serial.print((int)mode);
                    Serial.print(",");
                    Serial.print(feature_time);
                    Serial.print(",");
                    Serial.print(inference_time);
                    for (int i = 19; i < 29; i++) {
                        Serial.print(",");
                        Serial.print(features[i], 4);
                    }
                    Serial.println();
                }
                
                bufferPos = 0;  // Always reset after processing
            }
        } else {
            if (bufferPos < 597) {  // Leave room for null terminator
                serialBuffer[bufferPos++] = c;
            } else {
                // Buffer overflow — discard entire packet
                bufferPos = 0;
                Serial.println("R,ERR,OVERFLOW");
            }
        }
    }
    
    // Timeout check
    if (lastReceiveTime > 0 && millis() - lastReceiveTime > SERIAL_TIMEOUT_MS) {
        setLEDs(-1);
        digitalWrite(LED_YELLOW, HIGH);
        
        // Reset state on communication loss
        for (int i = 0; i < CONFIRM_COUNT; i++) {
            prediction_buffer[i] = 0;
        }
        consecutive_alerts = 0;
    }
    
    yield();  // Feed FreeRTOS watchdog
}
