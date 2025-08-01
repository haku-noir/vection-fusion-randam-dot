#include <M5Core2.h>

// 単位変換のための定数（1G = 9.80665 m/s^2）
const float G_TO_MS2 = 9.80665;

// M5StackのIMU（慣性計測ユニット）を初期化します。
// MPU9250やSH200Qなど、お使いのM5Stackのモデルに合わせてください。
// 例: M5.IMU.Init();

void setup() {
  // M5Stackの初期化
  M5.begin();
  
  // シリアル通信の開始（ボーレートはPython側と合わせる）
  Serial.begin(115200);

  // IMU（加速度センサ）の初期化
  // 注意: お使いのM5StackのCoreの種類によってIMUが異なります。
  // Basic/Gray -> MPU9250, Core2 -> MPU6886
  // Fire -> MPU9250
  // 下記の行はご自身のM5Stackに合わせて有効にしてください。
  M5.IMU.Init(); 

  M5.Lcd.setTextSize(2);
  M5.Lcd.println("XYZ Accel Sender");
  M5.Lcd.println("Unit: m/s^2");
  M5.Lcd.println("Port: 115200");
}

void loop() {
  float accX_g, accY_g, accZ_g;

  // 加速度センサのデータを取得 (単位: G)
  M5.IMU.getAccelData(&accX_g, &accY_g, &accZ_g);

  // ★変更点: Gからm/s^2に単位を変換
  float accX_ms2 = accX_g * G_TO_MS2;
  float accY_ms2 = accY_g * G_TO_MS2;
  float accZ_ms2 = accZ_g * G_TO_MS2;

  // ★変更点: m/s^2に変換したデータをカンマ区切りで送信
  Serial.printf("%.6f,%.6f,%.6f\n", accX_ms2, accY_ms2, accZ_ms2);

  // 画面にも値を表示（デバッグ用）
  M5.Lcd.setCursor(0, 80);
  M5.Lcd.printf("Acc X: %+.3f m/s^2\nAcc Y: %+.3f m/s^2\nAcc Z: %+.3f m/s^2", accX_ms2, accY_ms2, accZ_ms2);

  // データの送信間隔（10ミリ秒）
  // この値を変更することで、データのサンプリングレートを調整できます。
  delay(10);
}
