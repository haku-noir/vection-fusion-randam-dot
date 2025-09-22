#include <Arduino.h>

// pins
const int inputPin = 18;
const int dacPin25 = 25;
const int dacPin26 = 26;

// sine params
const float sineFreq = 0.1;   // Hz
const int duration = 180;     // seconds (audiovisual_experimentに合わせて180秒)

// amplitude control
// amplitudeScaled: 0..255 used for dacWrite
volatile uint8_t amplitudeScaled = 127; // default: full scale (0-255)

// Potentiometer settings (optional)
const bool USE_POT = false;    // true にするとポテンショで振幅制御
const int potPin = 34;         // ADC1_CH6 (GPIO34) - 入力専用ピン（例）
const int adcMax = 4095;       // 12-bit ADC reading (ESP32 default)

// serial parsing
String serialBuf = "";

unsigned long startMillis = 0;
bool active = false;

void setup() {
  Serial.begin(115200);
  pinMode(inputPin, INPUT);
  dacWrite(dacPin25, 0);
  dacWrite(dacPin26, 0);

  Serial.println("Ready. Commands:");
  Serial.println(" START_GVS = start GVS stimulation");
  Serial.println(" STOP_GVS = stop GVS stimulation");
  Serial.println(" V<0-255> = set amplitude (e.g. V128)");
  Serial.print(" Use potentiometer: "); Serial.println(USE_POT ? "ENABLED" : "disabled");
}

void loop() {
  // --- optional: read pot and map to amplitude ---
  if (USE_POT) {
    int adc = analogRead(potPin); // 0..4095
    // map to 0..255
    uint8_t amp = (uint8_t)(( (long)adc * 255 ) / adcMax);
    amplitudeScaled = amp;
    // print occasionally (optional)
    static unsigned long lastPrint = 0;
    if (millis() - lastPrint > 1000) {
      Serial.print("POT amplitude: "); Serial.println(amplitudeScaled);
      lastPrint = millis();
    }
  }

  // --- Serial input handling (line-based) ---
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r' || c == '\n') {
      if (serialBuf.length() > 0) {
        // process command
        if (serialBuf == "START_GVS") {
          if (!active) {
            active = true;
            startMillis = millis();
            Serial.println(">>> GVS START (180s) <<<");
          } else {
            Serial.println("GVS already active.");
          }
        } else if (serialBuf == "STOP_GVS") {
          active = false;
          dacWrite(dacPin25, 0);
          dacWrite(dacPin26, 0);
          Serial.println(">>> GVS STOP <<<");
        } else if (serialBuf == "s") {
          // 後方互換性のため古いコマンドも維持
          if (!active) {
            active = true;
            startMillis = millis();
            Serial.println(">>> GVS START (180s) <<<");
          } else {
            Serial.println("GVS already active.");
          }
        } else if (serialBuf == "x") {
          // 後方互換性のため古いコマンドも維持
          active = false;
          dacWrite(dacPin25, 0);
          dacWrite(dacPin26, 0);
          Serial.println(">>> GVS STOP <<<");;
        } else if (serialBuf[0] == 'V') {
          // parse amplitude value after 'V'
          String num = serialBuf.substring(1);
          int v = num.toInt();
          if (v < 0) v = 0;
          if (v > 255) v = 255;
          amplitudeScaled = (uint8_t)v;
          Serial.print("Amplitude set to: ");
          Serial.println(amplitudeScaled);
        } else {
          Serial.print("Unknown command: ");
          Serial.println(serialBuf);
        }
      }
      serialBuf = "";
    } else {
      serialBuf += c;
      // safety: limit buffer length
      if (serialBuf.length() > 20) serialBuf = serialBuf.substring(serialBuf.length()-20);
    }
  }

  // --- generate sine while active ---
  if (active) {
    unsigned long elapsed = millis() - startMillis;
    if (elapsed > duration * 1000UL) {
      // auto-stop
      dacWrite(dacPin25, 0);
      dacWrite(dacPin26, 0);
      active = false;
      Serial.println(">>> Auto GVS STOP after 180s <<<");
      return;
    }

    float t = elapsed / 1000.0;
    float sineValue = sin(2.0 * PI * sineFreq * t); // -1..1

    // scale by amplitudeScaled (0..255)
    int outVal = (int)(fabs(sineValue) * (float)amplitudeScaled + 0.5);

    if (sineValue >= 0) {
      dacWrite(dacPin25, outVal);
      dacWrite(dacPin26, 0);
    } else {
      dacWrite(dacPin25, 0);
      dacWrite(dacPin26, outVal);
    }
  }
}
