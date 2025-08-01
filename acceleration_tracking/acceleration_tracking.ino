#include <M5Core2.h>
#include "BluetoothSerial.h"

// BluetoothSerialオブジェクトを作成
BluetoothSerial SerialBT;

// Bluetoothデバイス名
String deviceName = "M5Stack-Accel";

// 単位変換のための定数（1G = 9.80665 m/s^2）
const float G_TO_MS2 = 9.80665;

void setup() {
  // M5Stackの初期化
  M5.begin();
  
  // IMU（加速度センサ）の初期化
  M5.IMU.Init();

  // Bluetoothシリアル通信を開始
  SerialBT.begin(deviceName);

  M5.Lcd.setTextSize(2);
  M5.Lcd.println("Bluetooth Accel Sender");
  M5.Lcd.printf("Device: %s\n", deviceName.c_str());
  M5.Lcd.println("Waiting for connection...");
}

void loop() {
  // PCからの接続がある場合のみデータを送信
  if (SerialBT.connected()) {
    float accX_g, accY_g, accZ_g;

    // 加速度センサのデータを取得 (単位: G)
    M5.IMU.getAccelData(&accX_g, &accY_g, &accZ_g);

    // Gからm/s^2に単位を変換
    float accX_ms2 = accX_g * G_TO_MS2;
    float accY_ms2 = accY_g * G_TO_MS2;
    float accZ_ms2 = accZ_g * G_TO_MS2;

    // m/s^2に変換したデータをカンマ区切りで送信
    SerialBT.printf("%.6f,%.6f,%.6f\n", accX_ms2, accY_ms2, accZ_ms2);

    // 画面にも値を表示（デバッグ用）
    M5.Lcd.setCursor(0, 80);
    M5.Lcd.printf("Acc X: %+.3f m/s^2\nAcc Y: %+.3f m/s^2\nAcc Z: %+.3f m/s^2", accX_ms2, accY_ms2, accZ_ms2);
  } else {
    // 接続待機中のメッセージ
    M5.Lcd.setCursor(0, 80);
    M5.Lcd.println("Connection waiting...   ");
    M5.Lcd.println("                        ");
    M5.Lcd.println("                        ");
  }

  // データの送信間隔（10ミリ秒）
  delay(10);
}
