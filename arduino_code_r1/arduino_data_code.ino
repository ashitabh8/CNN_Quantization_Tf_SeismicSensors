#include <WiFi.h>
#include <WiFiUdp.h>
#include "queue_arduino.h"

const char* SSID = "iobt-2.4";      // <-- your Wi-Fi name (in quotes)
const char* PASS = "73FOQUTORTQx";      // <-- your Wi-Fi password (in quotes)

const int SENSOR_PIN = A0;
WiFiUDP Udp;

IPAddress destIP(10, 7, 0, 186);
// 10.7.0.186
const int destPort = 5005;

// Queue for collecting 200 samples
Queue sensorQueue;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12); 

  // Initialize the sensor queue
  queue_init(&sensorQueue);

  Serial.print("Connecting to WiFi");
  WiFi.begin(SSID, PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
  }
  Serial.println("\nConnected!");

  // Print the board's IP so you know it joined the right network
  Serial.print("Board IP: ");
  Serial.println(WiFi.localIP());

  Udp.begin(5005); // local UDP port (ephemeral)
}

void loop() {
  // Read sensor value
  int sensorValue = analogRead(SENSOR_PIN);
  
  // Convert to float and push to queue
  float floatValue = (float)sensorValue;
  queue_push(&sensorQueue, floatValue);
  
  // Check if queue is full (200 samples collected)
  if (queue_is_full(&sensorQueue)) {
    // Get the array of 200 samples
    float* samples = queue_get_array(&sensorQueue);
    
    if (samples != NULL) {
      // Create packet with all 200 samples
      sendSensorDataPacket(samples);
      
      // Clear queue for next batch
      queue_clear(&sensorQueue);
    }
  }
  
  delay(10); // ~100 Hz sampling rate
}

void sendSensorDataPacket(float* samples) {
  // Create a packet with all 200 samples
  // Format: "timestamp,sample1,sample2,...,sample200"
  char packet[2048]; // Large enough for 200 float values
  int pos = 0;
  
  // Add timestamp
  unsigned long timestamp = millis();
  pos += snprintf(packet + pos, sizeof(packet) - pos, "%lu", timestamp);
  
  // Add all 200 samples
  for (int i = 0; i < QUEUE_SIZE; i++) {
    pos += snprintf(packet + pos, sizeof(packet) - pos, ",%.2f", samples[i]);
  }
  
  // Add newline
  packet[pos] = '\n';
  pos++;
  
  // Debug print to Serial
  Serial.print("Sending packet with 200 samples, size: ");
  Serial.println(pos);
  
  // Send via UDP
  Udp.beginPacket(destIP, destPort);
  Udp.write((uint8_t*)packet, pos);
  Udp.endPacket();
  
  Serial.println("Packet sent successfully!");
}
