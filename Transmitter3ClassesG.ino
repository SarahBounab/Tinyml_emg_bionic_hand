
#include <Arduino.h>
#include <math.h>
#include <WiFi.h>
#include <esp_now.h>

const int   SENSOR_PIN = 35;          
const float SAMPLE_MS = 1.0f;        

//HP IIR: y[n] = α ( y[n-1] + x[n] − x[n-1] )
const float dt    = SAMPLE_MS * 1e-3f;                
const float RC    = 1.0f / (2.0f * PI * 50.0f);       
const float α     = RC / (RC + dt);                  

float prevRaw      = 0.0f;
float prevFiltered = 0.0f;

//Segmentation paramms 
const float FS               = 1000.0f;              
const float WINDOW_SEC       = 0.3f;                 
const float STEP_RATIO       = 0.75f;               
const float STEP_SEC         = WINDOW_SEC * STEP_RATIO; 

const int WINDOW_SIZE = int(WINDOW_SEC * FS);        
const int STEP_SIZE   = int(STEP_SEC   * FS);       


const int   NUM_FEATURES   = 5;
const float W[NUM_FEATURES] = {
  -0.50699216, 4.88517627, 0.03267370, 4.75082771, 4.62333909
};
const float BIAS            = -7.31570857;
const float X_MIN[NUM_FEATURES] = {
  -0.92080000, 0.00270000, -0.02842333, 0.00087568, 0.00068900
};
const float X_MAX[NUM_FEATURES] = {
  -0.00280000, 1.52140000,  0.01851867, 0.29823671, 0.21019033
};
const int   CLASS_LABELS[2]    = { 0, 1 };

// Replace with the MAC address of your receiver ESP32
uint8_t receiverMac[] = {0xf0, 0x9e, 0x9e, 0x22, 0x6c, 0xac};

typedef struct struct_message {
  int classLabel;
} struct_message;

struct_message outgoingMsg;



float buffer[WINDOW_SIZE];
int   bufIndex      = 0;
int   samplesFilled = 0;
int   stepCounter   = 0;
int   lastClass     = 0;


float computeMin(const float *d, size_t N) {
  float m = d[0];
  for (size_t i = 1; i < N; ++i) if (d[i] < m) m = d[i];
  return m;
}
float computeMax(const float *d, size_t N) {
  float M = d[0];
  for (size_t i = 1; i < N; ++i) if (d[i] > M) M = d[i];
  return M;
}
float computeMean(const float *d, size_t N) {
  double s = 0.0;
  for (size_t i = 0; i < N; ++i) s += d[i];
  return float(s / N);
}
float computeStd(const float *d, size_t N, float mean) {
  double acc = 0.0;
  for (size_t i = 0; i < N; ++i) {
    double diff = d[i] - mean;
    acc += diff * diff;
  }
  return float(sqrt(acc / N));
}
float computeMeanAbs(const float *d, size_t N) {
  double s = 0.0;
  for (size_t i = 0; i < N; ++i) s += fabs(d[i]);
  return float(s / N);
}

// Feature Extraction & Scaling 
void extractFeatures(const float *seg, size_t N, float feats[NUM_FEATURES]) {
  feats[0] = computeMin(seg, N);
  feats[1] = computeMax(seg, N);
  feats[2] = computeMean(seg, N);
  feats[3] = computeStd(seg, N, feats[2]);
  feats[4] = computeMeanAbs(seg, N);
}
void scaleFeatures(const float in[NUM_FEATURES], float out[NUM_FEATURES]) {
  for (int i = 0; i < NUM_FEATURES; ++i) {
    out[i] = (in[i] - X_MIN[i]) / (X_MAX[i] - X_MIN[i]);
  }
}

//SVM Prediction
int predictLabel(const float scaled[NUM_FEATURES]) {
  double score = BIAS;
  for (int i = 0; i < NUM_FEATURES; ++i) {
    score += scaled[i] * W[i];
  }
  return (score >= 0) ? CLASS_LABELS[1] : CLASS_LABELS[0];
}

// segmentation & update lastclass —————
void maybeClassify() {
  if (samplesFilled < WINDOW_SIZE) return;   
  if (++stepCounter < STEP_SIZE) return;     
  stepCounter = 0;


  static float window[WINDOW_SIZE];
  for (int i = 0; i < WINDOW_SIZE; ++i) {
    int idx = (bufIndex + i) % WINDOW_SIZE;
    window[i] = buffer[idx];
  }


  float feats[NUM_FEATURES], scaled[NUM_FEATURES];
  extractFeatures(window, WINDOW_SIZE, feats);
  scaleFeatures(feats, scaled);
 lastClass = predictLabel(scaled);

// Send the class via ESPNOW
outgoingMsg.classLabel = lastClass;
esp_err_t result = esp_now_send(receiverMac, (uint8_t *)&outgoingMsg, sizeof(outgoingMsg));

if (result == ESP_OK) {
  Serial.println("Sent class via ESP-NOW");
} else {
  Serial.println("Error sending class via ESP-NOW");
}

}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("Filtered_Signal  Class_Label");
    WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (!esp_now_is_peer_exist(receiverMac)) {
    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
      Serial.println("Failed to add peer");
      return;
    }
  }

}

void loop() {
  static unsigned long lastMs = 0;
  unsigned long now = millis();
  if (now - lastMs < unsigned(SAMPLE_MS)) return;
  lastMs = now;
  int   rawADC = analogRead(SENSOR_PIN);
  float raw    = rawADC * (3.3f / 4095.0f);

//High-pass filter
  float filtered = α * (prevFiltered + raw - prevRaw);
  prevRaw      = raw;
  prevFiltered = filtered;

  buffer[bufIndex++] = filtered;
  if (bufIndex >= WINDOW_SIZE) bufIndex = 0;
  if (samplesFilled < WINDOW_SIZE) ++samplesFilled;
  
  maybeClassify();
  Serial.print(filtered, 4);
  Serial.print(' ');
  Serial.println(lastClass);
}
