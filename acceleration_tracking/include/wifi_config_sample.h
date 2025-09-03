#ifndef WIFI_CONFIG_H
#define WIFI_CONFIG_H

// WiFi設定 - 実際の環境に合わせて変更してください
const char* WIFI_SSID = "YOUR_WIFI_SSID";        // WiFiのSSIDを設定
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"; // WiFiのパスワードを設定

// UDP通信設定 - 実際の環境に合わせて変更してください
const int PC_PORT = 12345;                   // PCが待機するポート
const int M5_PORT = 12346;                   // M5Stackが待機するポート
const char* PC_IP_STR = "192.168.1.100";     // PCのIPアドレス（文字列で設定）

#endif // WIFI_CONFIG_H
