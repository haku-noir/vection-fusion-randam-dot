#include <M5Core2.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <vector>
#include "wifi_config.h"

// UDP通信設定
WiFiUDP udp;
IPAddress pc_ip;

// データ保存用構造体
struct AccelData {
  unsigned long timestamp;  // マイクロ秒
  float x, y, z;           // 加速度データ (m/s^2)
};

// データ保存用バッファ
std::vector<AccelData> accel_buffer;
const size_t MAX_BUFFER_SIZE = 100000;  // 最大保存数（100秒分の余裕）

// 測定状態
bool measuring = false;
unsigned long measurement_start_time = 0;

// 単位変換のための定数（1G = 9.80665 m/s^2）
const float G_TO_MS2 = 9.80665;

// 関数の前方宣言
void checkUDPCommands();
void collectAccelData();
void updateDisplay();
void startMeasurement();
void stopMeasurement();
void sendDataToPC();

void setup() {
  // M5Stackの初期化（警告対応：パラメータを明示的に指定）
  M5.begin(true, true, true, true);

  // IMU（加速度センサ）の初期化
  M5.IMU.Init();

  // シリアル通信の開始（デバッグ用）
  Serial.begin(115200);

  // LCD初期化
  M5.Lcd.setTextSize(2);
  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setCursor(0, 0);
  M5.Lcd.println("Wireless Accel");
  M5.Lcd.println("Connecting WiFi...");

  // PCのIPアドレスを設定
  pc_ip.fromString(PC_IP_STR);

  // WiFi接続
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    M5.Lcd.print(".");
  }

  M5.Lcd.fillScreen(BLACK);
  M5.Lcd.setCursor(0, 0);
  M5.Lcd.println("WiFi Connected!");
  M5.Lcd.printf("IP: %s\n", WiFi.localIP().toString().c_str());
  M5.Lcd.printf("Port: %d\n", M5_PORT);

  // UDP開始
  udp.begin(M5_PORT);

  // バッファを予約
  accel_buffer.reserve(MAX_BUFFER_SIZE);

  M5.Lcd.println("Ready for commands");
  Serial.println("M5Stack ready for wireless communication");
}

void loop() {
  M5.update();

  // UDP パケットの受信チェック
  checkUDPCommands();

  // 測定中の場合、加速度データを収集
  if (measuring) {
    collectAccelData();
  }

  // 画面更新（頻度を下げて処理負荷軽減）
  updateDisplay();

  // delayを削除して最高速度でループ実行
  // yield(); // WiFiスタック処理のため
}

void checkUDPCommands() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char packet[256];
    int len = udp.read(packet, 255);
    packet[len] = '\0';

    String command = String(packet);
    command.trim();

    Serial.printf("Received command: %s\n", command.c_str());

    if (command == "START_MEASUREMENT") {
      startMeasurement();
    } else if (command == "STOP_MEASUREMENT") {
      stopMeasurement();
    } else if (command == "SEND_DATA") {
      sendDataToPC();
    }
  }
}

void startMeasurement() {
  measuring = true;
  measurement_start_time = micros();
  accel_buffer.clear();

  // PCに開始確認を送信
  udp.beginPacket(pc_ip, PC_PORT);
  udp.print("MEASUREMENT_STARTED");
  udp.endPacket();

  Serial.printf("Measurement started. Buffer capacity: %d samples\n", MAX_BUFFER_SIZE);
  Serial.printf("Expected capacity for 60s at 1000Hz: 60,000 samples\n");
}

void stopMeasurement() {
  measuring = false;

  // PCに停止確認を送信
  udp.beginPacket(pc_ip, PC_PORT);
  udp.print("MEASUREMENT_STOPPED");
  udp.endPacket();

  Serial.printf("Measurement stopped. Collected %d samples\n", accel_buffer.size());
}

void collectAccelData() {
  // バッファがフルになった場合の警告表示（ただし収集は継続しない）
  if (accel_buffer.size() >= MAX_BUFFER_SIZE) {
    static bool warning_shown = false;
    if (!warning_shown) {
      Serial.printf("Warning: Buffer full at %d samples. Collection stopped.\n", accel_buffer.size());
      warning_shown = true;
    }
    return;  // バッファが満杯の場合は収集停止
  }

  // 高頻度サンプリングのため、static変数でタイミング制御
  static unsigned long last_sample = 0;
  unsigned long now = micros();

  // 120Hz (約8.33ms間隔) でサンプリング
  // 1/120秒 = 8333マイクロ秒
  if (now - last_sample < 8333) {
    return;
  }
  last_sample = now;

  float accX_g, accY_g, accZ_g;
  M5.IMU.getAccelData(&accX_g, &accY_g, &accZ_g);

  AccelData data;
  data.timestamp = now - measurement_start_time;  // 測定開始からの相対時間
  data.x = accX_g * G_TO_MS2;
  data.y = accY_g * G_TO_MS2;
  data.z = accZ_g * G_TO_MS2;

  accel_buffer.push_back(data);
}

void sendDataToPC() {
  Serial.printf("Sending %d data points to PC\n", accel_buffer.size());

  // データをJSON形式で送信（遅延を最小化）
  udp.beginPacket(pc_ip, PC_PORT);
  udp.print("DATA_START");
  udp.endPacket();
  delay(2);  // 遅延を短縮

  // データを分割して送信（チャンクサイズを増加）
  const size_t CHUNK_SIZE = 20;  // 一度に送信するデータ数を増加
  for (size_t i = 0; i < accel_buffer.size(); i += CHUNK_SIZE) {
    String dataChunk = "";

    for (size_t j = i; j < i + CHUNK_SIZE && j < accel_buffer.size(); j++) {
      if (j > i) dataChunk += "|";
      dataChunk += String(accel_buffer[j].timestamp) + "," + 
                   String(accel_buffer[j].x, 6) + "," +
                   String(accel_buffer[j].y, 6) + "," +
                   String(accel_buffer[j].z, 6);
    }

    udp.beginPacket(pc_ip, PC_PORT);
    udp.print(dataChunk);
    udp.endPacket();
    delay(1);  // パケット間の遅延を短縮
  }

  // 送信完了を通知
  udp.beginPacket(pc_ip, PC_PORT);
  udp.print("DATA_END");
  udp.endPacket();

  Serial.println("Data transmission completed");
}

void updateDisplay() {
  static unsigned long last_update = 0;
  if (millis() - last_update < 200) return;  // 200ms間隔で更新

  last_update = millis();

  M5.Lcd.fillRect(0, 120, 320, 120, BLACK);
  M5.Lcd.setCursor(0, 120);

  if (measuring) {
    M5.Lcd.println("MEASURING...");
    M5.Lcd.printf("Samples: %d/%d\n", accel_buffer.size(), MAX_BUFFER_SIZE);

    // バッファ使用率を表示
    float usage_percent = (float)accel_buffer.size() / MAX_BUFFER_SIZE * 100.0;
    M5.Lcd.printf("Usage: %.1f%%\n", usage_percent);

    // 現在の加速度を表示
    if (!accel_buffer.empty()) {
      auto& latest = accel_buffer.back();
      M5.Lcd.printf("X: %+.3f\n", latest.x);
      M5.Lcd.printf("Y: %+.3f\n", latest.y);
      M5.Lcd.printf("Z: %+.3f\n", latest.z);
    }
  } else {
    M5.Lcd.println("STANDBY");
    M5.Lcd.printf("Buffer: %d/%d\n", accel_buffer.size(), MAX_BUFFER_SIZE);
    M5.Lcd.println("Ready for commands");
  }
}
