#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audiovisual_experiment.py
---------------------------------
赤・緑ランダムドットとパンニング音声の視聴覚統合実験テンプレート
・Standalone PsychoPy 専用（Python 3）
・Audio backend に必ず PTB (Psychtoolbox) を使用

- 音響パンニングの方式を 'volume', 'itd', 'both' から選択。
- ESCキーが押されるまで実験試行を無限に繰り返す。
- 各試行で音と同期するドットの色をランダムに決定。
"""

FORCE_COND = 'red'
FORCE_COND = 'green'
FORCE_COND = None

# ★★★ 実験条件制御設定 ★★★
# 各条件をcond_typeに対して反対にするかどうかの設定
# 注意: これらの設定は個別にON/OFFできます（互いに独立）
SINGLE_COLOR_DOT = False     # ランダムドット1色のみ表示（デフォルトはcond_typeの反対色のみ）
                            # True: cond_type='red'なら緑ドットのみ、'green'なら赤ドットのみ（デフォルト）
VISUAL_REVERSE = False       # 視覚刺激の逆転（SINGLE_COLOR_DOTでcond_typeの色のみ表示）
                            # True: cond_type='red'なら赤ドットのみ、'green'なら緑ドットのみ（SINGLE_COLOR_DOT有効時）
AUDIO_REVERSE = False        # 音響刺激の反対（cond_typeと反対側にパンニング）
                            # True: cond_type='red'なら緑ドット側に音がパンニング、'green'なら赤ドット側に音がパンニング
GVS_REVERSE = False          # GVS刺激の反対（cond_typeと反対の極性）
                            # True: cond_type='red'なら緑条件のGVS刺激、'green'なら赤条件のGVS刺激

# ------------------------------------------------------------------
# 0. PTB を最優先でロードするための prefs 設定
# ------------------------------------------------------------------
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']      # PTB を第一候補に固定
prefs.hardware['audioLatencyMode'] = 3    # 低レイテンシ 0–4（3 がおすすめ）
# 必要なら出力デバイスを指定:
# prefs.hardware['audioDevice'] = 'Built-in Output'


# ------------------------------------------------------------------
# 1. 必要ライブラリのインポート
# ------------------------------------------------------------------
from psychopy import sound, visual, core, event, constants
import numpy as np
import csv
import os
import random
from datetime import datetime
import socket
import threading
import time
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports


# ------------------------------------------------------------------
# 2. 実験パラメータ（自由に変更可）
# ------------------------------------------------------------------
# M5StackとのWiFi通信設定 - 実際の環境に合わせて変更してください
M5_IP = '192.168.1.235'  # M5StackのIPアドレス（M5Stackの画面で確認して設定）
M5_PORT = 12346          # M5Stackが待機するポート
PC_PORT = 12345          # PCが待機するポート

# シリアル通信設定
SERIAL_PORT = "/dev/cu.wchusbserial556F0046031"       # 自動検出（もしくは '/dev/cu.usbserial-xxx' などを指定）
SERIAL_BAUDRATE = 115200 # M5Stackのボーレート

# 通信方式選択（どちらか一つのみ選択）
# WiFi通信を使用する場合：
#   COMMUNICATION_MODE = 'WIFI'
#   M5Stack側でも COMMUNICATION_MODE = WIFI_MODE に設定
# シリアル通信を使用する場合：
#   COMMUNICATION_MODE = 'SERIAL'
#   M5Stack側でも COMMUNICATION_MODE = SERIAL_MODE に設定
COMMUNICATION_MODE = 'SERIAL'  # 'WIFI' または 'SERIAL'

# GVS（前庭電気刺激）設定
USE_GVS = True               # GVS刺激を使用するかどうか
GVS_SERIAL_PORT = "/dev/cu.usbserial-0001"  # GVS用ESP32のシリアルポート
GVS_BAUDRATE = 115200        # GVSのボーレート

# 光同期用Arduino設定
USE_LIGHT_SYNC = True        # 光同期機能を使用するかどうか
LIGHT_SYNC_SERIAL_PORT = "/dev/cu.usbserial-6"  # 光同期用Arduinoのシリアルポート
LIGHT_SYNC_BAUDRATE = 9600   # 光同期用Arduinoのボーレート

# 同期用表示領域設定
SYNC_SQUARE_SIZE = 25        # 同期用白/黒表示のサイズ [pix]
SYNC_SQUARE_POS_X = 845      # 同期用表示のX座標（画面右下）
SYNC_SQUARE_POS_Y = -455     # 同期用表示のY座標（画面右下）

# 音源の情報を保持するクラス
class SoundSource:
    """音源の周波数を保持するシンプルなクラス"""
    def __init__(self, freqs):
        if not isinstance(freqs, list):
            self.freqs = [freqs]
        else:
            self.freqs = freqs

# ★★★ 音響モードを選択 ★★★
# 'volume': 音量差パンニング
# 'itd'   : 時間差 (ITD) パンニング
# 'both'  : 音量差と時間差の両方
PANNING_MODE = 'volume'

# ★★★ 音響ソースモードを選択 ★★★
# 'simulation': シミュレーション音（現在の方式）
# 'mp3': MP3ファイル再生
AUDIO_SOURCE_MODE = 'mp3'  # 'simulation' または 'mp3'

# MP3ファイル設定（AUDIO_SOURCE_MODE = 'mp3'の場合）
MP3_FILE_RED = "audio_files/red.wav"    # 赤ドット同期用MP3ファイル
MP3_FILE_GREEN = "audio_files/green.wav"  # 緑ドット同期用MP3ファイル
MP3_LOOP = True                                      # MP3をループ再生するかどうか
# 注意: MP3ファイルを使用する場合は、audio_files/フォルダに両方のMP3ファイルを配置してください

# スクロールモードのON/OFFを設定
SCROLLING_MODE = True
VIRTUAL_HEIGHT_MULTIPLIER = 2.0 # スクロールモード時の仮想空間の高さ（画面の何倍か）

# 画面・ドット
WIN_SIZE      = (1920, 1080)  # ウィンドウ解像度
N_DOTS        = 3000          # 赤・緑それぞれのドット数
DOT_SIZE      = 15            # ドット直径 [pix]
FALL_SPEED    = 350           # 落下・スクロール速度 [pix/s]
OSC_FREQ      = 0.1         # 横揺れ周波数 [Hz]
OSC_AMP       = 500           # ドットの横揺れ振幅 [pix]

# 音
# 2D座標系での音響パラメータ
SOUND_INITIAL_X = 0.0       # 音像の初期X座標
SOUND_INITIAL_Y = 1.0       # 音像の初期Y座標
SOUND_OSC_AMPLITUDE = 0.5   # 音像の振動振幅（X軸方向）

# 被験者と耳の位置
LISTENER_X = 0.0            # 被験者の中心X座標
LISTENER_Y = 0.0            # 被験者の中心Y座標
LEFT_EAR_X = -0.1           # 左耳のX座標
LEFT_EAR_Y = 0.0            # 左耳のY座標
RIGHT_EAR_X = 0.1           # 右耳のX座標
RIGHT_EAR_Y = 0.0           # 右耳のY座標

# 距離減衰パラメータ
DISTANCE_ATTENUATION = 5.0  # 距離減衰の係数（音量を適切に保つ）
MIN_DISTANCE_GAIN = 0.3     # 最小ゲイン（より高く設定）

# SoundSourceオブジェクトをシンプルに定義
SOUND_SOURCES = [
#    SoundSource(freqs=[523.25]),  # ド(C5)
]
SAMPLE_RATE   = 44100        # サンプリングレート [Hz]
MAX_ITD_S     = 0.0007       # ITDの最大値 (秒)。'itd'または'both'モードで使用

# 試行
TRIAL_DURATION   = 179.0      # 各試行の刺激掲示時間 [s]
ITI              = 1.0       # 刺激間インターバル [s]
MAX_TRIALS       = 1         # 最大試行回数（デフォルト1回）

# データ出力設定
SAVE_COMBINED_DATA = False   # 統合データファイルを出力するかどうか（互換性用）

# ビープ音設定
USE_BEEP = False              # ビープ音を使用するかどうか
BEEP_FREQ = 1000             # ビープ音の周波数 [Hz]
BEEP_DURATION = 0.2          # ビープ音の長さ [s]
BEEP_VOLUME = 0.5            # ビープ音の音量 (0.0-1.0)
BEEP_SEP = 2.0               # ビープ音と刺激の間の長さ [s]

# 中心線設定
USE_CENTER_LINES = False      # 中心線を表示するかどうか
CENTER_LINE_WIDTH = 8        # 中心線の太さ [pix]

# ログ
LOG_DIR = PANNING_MODE
os.makedirs(LOG_DIR, exist_ok=True)
# experiment_logファイルは最初の試行の stimulus_start_time に基づいて作成

# M5Stackとのシリアル通信を管理するクラス
class SerialCommunicator:
    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.accel_data_buffer = deque()
        self.measuring = False

    def find_m5stack_port(self):
        """M5Stack（ESP32）のシリアルポートを自動検出"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # macOSでのUSBシリアルポート名パターン
            if ('usbserial' in port.device or 'usbmodem' in port.device or
                'SLAB_USBtoUART' in str(port.description) or
                'Silicon Labs' in str(port.manufacturer)):
                print(f"M5Stack候補ポート発見: {port.device} - {port.description}")
                return port.device
        return None

    def start(self):
        """シリアル通信開始"""
        try:
            if self.port is None:
                self.port = self.find_m5stack_port()
                if self.port is None:
                    print("M5Stackのシリアルポートが見つかりません")
                    return False

            print(f"シリアル接続を開始: {self.port} @ {self.baudrate}")
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # 接続安定化のための待機

            self.running = True
            self.thread = threading.Thread(target=self.read_serial_data)
            self.thread.daemon = True
            self.thread.start()

            print("シリアル通信開始完了")
            return True

        except Exception as e:
            print(f"シリアル接続エラー: {e}")
            return False

    def stop(self):
        """シリアル通信停止"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.serial_connection:
            self.serial_connection.close()

    def read_serial_data(self):
        """シリアルデータを読み込むスレッド"""
        line_count = 0
        raw_line_count = 0
        accel_line_count = 0

        print("シリアルデータ読み込みスレッドを開始しました")

        while self.running:
            try:
                if self.serial_connection and self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    raw_line_count += 1

                    if line:  # 空行でない場合のみ処理
                        line_count += 1

                        # デバッグ: 最初の10行と定期的な行を表示
                        if line_count <= 10 or line_count % 500 == 0:
                            print(f"シリアル受信[{line_count}]: {line}")

                        # 加速度データの処理
                        if line.startswith("ACCEL_DATA,"):
                            accel_line_count += 1
                            if accel_line_count <= 5 or accel_line_count % 100 == 0:
                                print(f"加速度データ[{accel_line_count}]: {line}")

                        self.process_serial_line(line)

                # 受信データの統計を定期的に報告
                if raw_line_count > 0 and raw_line_count % 1000 == 0:
                    print(f"シリアル統計: raw_lines={raw_line_count}, valid_lines={line_count}, accel_lines={accel_line_count}")

                time.sleep(0.001)  # 1ms待機
            except Exception as e:
                if self.running:  # 停止中でない場合のみエラー表示
                    print(f"シリアルデータ読み込みエラー: {e}")
                time.sleep(0.01)  # エラー時は少し長めに待機

    def process_serial_line(self, line):
        """受信したシリアル行を処理"""
        if line.startswith("ACCEL_DATA,"):
            # フォーマット: ACCEL_DATA,timestamp,x,y,z
            try:
                parts = line.split(',')
                if len(parts) == 5:
                    timestamp = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])

                    with self.lock:
                        self.accel_data_buffer.append((timestamp, x, y, z))
                        # デバッグ: バッファサイズを定期的に報告
                        if len(self.accel_data_buffer) % 1000 == 0:
                            print(f"加速度データバッファサイズ: {len(self.accel_data_buffer)}")
                else:
                    print(f"加速度データフォーマットエラー: 要素数={len(parts)}, line='{line}'")

            except Exception as e:
                print(f"加速度データ解析エラー: {e} (line: '{line}')")

        elif line == "MEASUREMENT_STARTED":
            print("M5Stack: 測定開始確認（シリアル経由）")
            self.measuring = True

        elif line == "MEASUREMENT_STOPPED":
            print("M5Stack: 測定停止確認（シリアル経由）")
            self.measuring = False

        elif line == "DATA_START":
            print("M5Stack: データ送信開始")

        elif line == "DATA_END":
            print("M5Stack: データ送信完了")

        elif line.startswith("Sending") and "data points" in line:
            print(f"M5Stack: {line}")

        elif line.startswith("Serial data transmission completed"):
            print(f"M5Stack: {line}")

        elif line == "MEASUREMENT_START":
            print("M5Stack: 測定開始（シリアル経由）")
            self.measuring = True

        elif line == "MEASUREMENT_STOP":
            print("M5Stack: 測定停止（シリアル経由）")
            self.measuring = False

        elif "Auto-measurement started" in line:
            print(f"M5Stack: {line}")

        elif "Buffer capacity" in line or "Sampling rate" in line:
            print(f"M5Stack: {line}")

        elif line.startswith("DEBUG:"):
            print(f"M5Stack Debug: {line}")

        else:
            # その他のメッセージもデバッグ出力（起動メッセージ以外）
            if line and not line.startswith("M5Stack ready") and not line.startswith("ets") and not line.startswith("configsip") and not line.startswith("mode"):
                print(f"シリアル受信（その他）: {line}")

    def get_accel_data(self):
        """蓄積された加速度データを取得"""
        with self.lock:
            data = list(self.accel_data_buffer)
            self.accel_data_buffer.clear()
            return data

    def send_command(self, command):
        """M5Stackにシリアルコマンドを送信"""
        try:
            if self.serial_connection:
                command_with_newline = command + '\n'
                self.serial_connection.write(command_with_newline.encode('utf-8'))
                print(f"シリアルコマンド送信: {command}")
                return True
        except Exception as e:
            print(f"シリアルコマンド送信エラー: {e}")
        return False

    def start_measurement(self):
        """測定開始コマンドを送信"""
        return self.send_command("START_MEASUREMENT")

    def stop_measurement(self):
        """測定停止コマンドを送信"""
        return self.send_command("STOP_MEASUREMENT")

    def request_data(self):
        """データ送信要求コマンドを送信"""
        return self.send_command("SEND_DATA")

    def get_all_accel_data(self):
        """全ての蓄積された加速度データを取得（クリアしない）"""
        with self.lock:
            return list(self.accel_data_buffer)

# M5Stackとの無線通信を管理するクラス
class UDPCommunicator(threading.Thread):
    def __init__(self, m5_ip, m5_port, pc_port):
        super(UDPCommunicator, self).__init__()
        self.m5_ip = m5_ip
        self.m5_port = m5_port
        self.pc_port = pc_port
        self.socket = None
        self.running = False
        self.lock = threading.Lock()
        self.accel_data_buffer = []  # 受信した加速度データを保存
        self.daemon = True
        self.measurement_active = False
        self.data_reception_complete = False  # データ受信完了フラグ
        self.data_reception_start_time = 0    # データ受信開始時刻

    def run(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.pc_port))
            self.socket.settimeout(1.0)
            self.running = True
            print(f"UDP communicator started on port {self.pc_port}")
        except Exception as e:
            print(f"Error: Could not start UDP communicator. {e}")
            self.running = False
            return

        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                message = data.decode('utf-8').strip()

                if message == "MEASUREMENT_STARTED":
                    print("M5Stack confirmed measurement start")
                elif message == "MEASUREMENT_STOPPED":
                    print("M5Stack confirmed measurement stop")
                elif message == "DATA_START":
                    with self.lock:
                        self.accel_data_buffer.clear()
                        self.data_reception_complete = False
                        self.data_reception_start_time = time.time()
                    print("Starting to receive acceleration data...")
                elif message == "DATA_END":
                    with self.lock:
                        self.data_reception_complete = True
                    print(f"Data reception completed. Received {len(self.accel_data_buffer)} data points")
                elif "|" in message:  # データチャンク
                    self.parse_accel_data(message)

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"UDP receive error: {e}")

        if self.socket:
            self.socket.close()
        print("UDPCommunicator thread stopped.")

    def parse_accel_data(self, message):
        """受信したデータチャンクを解析して加速度データに変換"""
        try:
            with self.lock:
                data_entries = message.split("|")
                for entry in data_entries:
                    parts = entry.split(",")
                    if len(parts) == 4:
                        timestamp = int(parts[0])  # マイクロ秒
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        self.accel_data_buffer.append((timestamp, x, y, z))
        except Exception as e:
            print(f"Error parsing acceleration data: {e}")

    def send_command(self, command):
        """M5Stackにコマンドを送信"""
        try:
            if self.socket:
                self.socket.sendto(command.encode('utf-8'), (self.m5_ip, self.m5_port))
                print(f"Sent command: {command}")
                return True
        except Exception as e:
            print(f"Error sending command: {e}")
        return False

    def start_measurement(self):
        """測定開始コマンドを送信"""
        self.measurement_active = True
        return self.send_command("START_MEASUREMENT")

    def stop_measurement(self):
        """測定停止コマンドを送信"""
        self.measurement_active = False
        return self.send_command("STOP_MEASUREMENT")

    def request_data(self):
        """データ送信要求コマンドを送信"""
        return self.send_command("SEND_DATA")

    def wait_for_data_reception(self, timeout=10.0):
        """データ受信完了を待機する（タイムアウト付き）"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if self.data_reception_complete:
                    return True
            time.sleep(0.1)  # 100ms間隔でチェック
        print(f"Warning: Data reception timeout after {timeout} seconds")
        return False

    def get_accel_data(self):
        """収集した加速度データを取得"""
        with self.lock:
            return list(self.accel_data_buffer)

    def clear_data(self):
        """データバッファをクリア"""
        with self.lock:
            self.accel_data_buffer.clear()

    def stop(self):
        self.running = False


# GVS制御クラス
class GVSController:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.connected = False

    def connect(self):
        """GVS用ESP32に接続"""
        try:
            import serial
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # 接続待機
            self.connected = True
            print(f"GVS Controller connected to {self.port}")
            return True
        except Exception as e:
            print(f"GVS Controller connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """GVS用ESP32から切断"""
        if self.serial_connection:
            try:
                self.serial_connection.close()
                self.connected = False
                print("GVS Controller disconnected")
            except Exception as e:
                print(f"GVS Controller disconnect error: {e}")

    def send_command(self, command):
        """GVS用ESP32にコマンドを送信"""
        if not self.connected or not self.serial_connection:
            print("GVS Controller not connected")
            return False

        try:
            command_with_newline = command + '\n'
            self.serial_connection.write(command_with_newline.encode('utf-8'))
            print(f"GVS command sent: {command}")
            return True
        except Exception as e:
            print(f"GVS command send error: {e}")
            return False

    def start_stimulation(self, color_sync='red'):
        """GVS刺激開始 - 色に応じて同相/逆相を指定"""
        if color_sync == 'red':
            return self.send_command("START_GVS_RED")  # 赤=同相
        elif color_sync == 'green':
            return self.send_command("START_GVS_GREEN")  # 緑=逆相
        else:
            # デフォルトは赤（同相）
            return self.send_command("START_GVS_RED")

    def stop_stimulation(self):
        """GVS刺激停止"""
        return self.send_command("STOP_GVS")

    def set_amplitude(self, amplitude):
        """GVS振幅設定（0-255）"""
        if 0 <= amplitude <= 255:
            return self.send_command(f"V{amplitude}")
        else:
            print(f"Invalid GVS amplitude: {amplitude} (must be 0-255)")
            return False


# 光同期用Arduino制御クラス
class LightSyncController:
    def __init__(self, port, baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.connected = False

    def connect(self):
        """光同期用Arduinoに接続"""
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduinoの初期化待機
            self.connected = True
            print(f"光同期用Arduinoに接続しました: {self.port}")
            return True
        except Exception as e:
            print(f"光同期用Arduino接続エラー: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """光同期用Arduinoから切断"""
        if self.serial_connection:
            try:
                # 最終停止コマンド送信（readyFlagをfalseに保つ）
                self.force_stop()
                time.sleep(0.1)
                self.serial_connection.close()
                print("Light Sync Arduinoの接続を切断しました")
            except Exception as e:
                print(f"Light Sync Arduino切断エラー: {e}")
        self.connected = False

    def send_command(self, command):
        """光同期用Arduinoにコマンド送信"""
        if not self.connected or not self.serial_connection:
            return False
        try:
            command_str = command + '\n'
            self.serial_connection.write(command_str.encode())
            print(f"光同期コマンド送信: {command}")
            return True
        except Exception as e:
            print(f"光同期コマンド送信エラー: {e}")
            return False

    def ready(self):
        """刺激準備完了を通知（刺激検出準備状態にセット）"""
        return self.send_command('r')  # readyコマンド - 刺激検出準備

    def force_stop(self):
        """刺激強制停止（リセットなし）"""
        return self.send_command('x')  # 強制停止コマンド

    def force_high(self):
        """強制的にHIGHに設定（テスト用）"""
        return self.send_command('s')


# ------------------------------------------------------------------
# 3. ウィンドウと刺激の準備
# ------------------------------------------------------------------

def create_beep_sound(freq, duration, volume=0.5, sample_rate=44100):
    """ビープ音を生成（test_random_dotsと同じ方式）"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * freq * t) * volume
    # ステレオ信号として作成
    stereo = np.column_stack([wave, wave])
    return sound.Sound(value=stereo, sampleRate=sample_rate, stereo=True)

win = visual.Window(size=WIN_SIZE, color=[0, 0, 0], units='pix',
                    fullscr=False, allowGUI=True) # フルスクリーンに変更

# 初期画面で光センサーが反応しないよう、すぐに黒い同期正方形を描画
if USE_LIGHT_SYNC:
    temp_sync_square = visual.Rect(
        win=win,
        width=SYNC_SQUARE_SIZE,
        height=SYNC_SQUARE_SIZE,
        pos=(SYNC_SQUARE_POS_X, SYNC_SQUARE_POS_Y),
        fillColor='black',
        lineColor=None
    )
    temp_sync_square.draw()
    win.flip()
    time.sleep(0.1)  # 確実に表示

def create_dot_stim(color_rgb):
    return visual.ElementArrayStim(
        win, nElements=N_DOTS, elementTex=None, elementMask='circle',
        sizes=DOT_SIZE, colors=color_rgb, xys=np.zeros((N_DOTS, 2)),
        colorSpace='rgb'
    )

ORANGE_RGB = [1.0, 0.294, -1.0]
GREEN_RGB = [-1, 1, -1]
red_dots   = create_dot_stim(ORANGE_RGB)
green_dots = create_dot_stim(GREEN_RGB)

# 光同期用白/黒正方形の作成
sync_square = visual.Rect(
    win=win,
    width=SYNC_SQUARE_SIZE,
    height=SYNC_SQUARE_SIZE,
    pos=(SYNC_SQUARE_POS_X, SYNC_SQUARE_POS_Y),
    fillColor='black',  # 初期状態は黒
    lineColor=None
)

# 中心線を作成（オプション）
if USE_CENTER_LINES:
    center_line_red = visual.Line(
        win,
        start=(0, -win.size[1]//2),
        end=(0, win.size[1]//2),
        lineColor=ORANGE_RGB,  # 赤色
        lineWidth=CENTER_LINE_WIDTH
    )
    center_line_green = visual.Line(
        win,
        start=(0, -win.size[1]//2),
        end=(0, win.size[1]//2),
        lineColor=GREEN_RGB,  # 緑色
        lineWidth=CENTER_LINE_WIDTH
    )
else:
    center_line_red = None
    center_line_green = None

# ビープ音を作成（オプション）
if USE_BEEP:
    beep_sound = create_beep_sound(BEEP_FREQ, BEEP_DURATION, BEEP_VOLUME)
else:
    beep_sound = None

WIN_W, WIN_H = win.size
Y_MAX = WIN_H / 2
Y_MIN = -WIN_H / 2

def init_positions():
    xpos = np.random.uniform(-WIN_W/2 - OSC_AMP, WIN_W/2 + OSC_AMP, N_DOTS)
    if SCROLLING_MODE:
        # スクロールモードでは、より広い仮想空間にドットを配置
        virtual_h = WIN_H * VIRTUAL_HEIGHT_MULTIPLIER
        ypos = np.random.uniform(-virtual_h/2, virtual_h/2, N_DOTS)
    else:
        # 通常モードでは画面内に配置
        ypos = np.random.uniform(Y_MIN, Y_MAX, N_DOTS)
    return np.column_stack([xpos, ypos])


# ------------------------------------------------------------------
# 4. 音響生成システム
# ------------------------------------------------------------------

def load_mp3_sound(file_path: str, duration: float = None) -> sound.Sound:
    """
    MP3ファイルを読み込んで、指定された長さに調整したSound オブジェクトを作成
    """
    try:
        import os

        # ファイルの存在確認
        if not os.path.exists(file_path):
            print(f"MP3ファイルが見つかりません: {file_path}")
            return None

        # PsychoPyのsound.Soundを使ってMP3を読み込み
        mp3_sound = sound.Sound(file_path)

        # ループ設定
        if MP3_LOOP:
            mp3_sound.setLoops(-1)  # -1で無限ループ
        else:
            mp3_sound.setLoops(0)   # 1回再生

        print(f"MP3ファイルを読み込みました: {file_path}")
        print(f"ループ設定: {'有効' if MP3_LOOP else '無効'}")
        return mp3_sound

    except Exception as e:
        print(f"MP3ファイルの読み込みに失敗しました: {e}")
        print("可能な原因:")
        print("- ファイルパスが正しくない")
        print("- MP3ファイルが破損している")
        print("- PsychoPyが対応していない形式")
        return None

def build_audio_source(sync_to_red: bool, mode: str) -> sound.Sound:
    """
    選択された音響ソースモードに応じて音声を生成
    """
    if AUDIO_SOURCE_MODE == 'mp3':
        # MP3ファイル再生モード - 赤・緑の同期に応じてファイルを選択
        if sync_to_red:
            mp3_file_path = MP3_FILE_RED
            sync_type = "赤ドット同期"
        else:
            mp3_file_path = MP3_FILE_GREEN
            sync_type = "緑ドット同期"

        mp3_sound = load_mp3_sound(mp3_file_path, TRIAL_DURATION)
        if mp3_sound:
            print(f"MP3モード: {sync_type} - {mp3_file_path} を再生")
            return mp3_sound
        else:
            print("MP3読み込み失敗のため、シミュレーション音響を使用")
            # フォールバックとしてシミュレーション音響を使用
            return build_multi_stereo_sound(sync_to_red, mode)
    else:
        # シミュレーション音響モード（従来の方式）
        return build_multi_stereo_sound(sync_to_red, mode)

# 2D座標系音響生成（各耳への個別距離減衰）
def build_multi_stereo_sound(sync_to_red: bool, mode: str) -> sound.Sound:
    """
    2D座標系で音像と各耳の距離を計算し、個別の距離減衰を適用した
    ステレオ音響を生成する。
    """
    t = np.linspace(0, TRIAL_DURATION, int(SAMPLE_RATE * TRIAL_DURATION), endpoint=False)

    # ランダムドットと同期する振動波形
    osc_wave = np.cos(2 * np.pi * OSC_FREQ * t)
    if sync_to_red:
        osc_wave *= -1  # 赤ドット同期時は位相反転

    # 音像のX座標の時間変化: (-0.5, 1) から (0.5, 1) を振動
    sound_x = SOUND_INITIAL_X + SOUND_OSC_AMPLITUDE * osc_wave
    sound_y = np.full_like(t, SOUND_INITIAL_Y)  # Y座標は固定

    # 各時間点での左耳と右耳から音像までの距離計算
    distance_left = np.sqrt((sound_x - LEFT_EAR_X)**2 + (sound_y - LEFT_EAR_Y)**2)
    distance_right = np.sqrt((sound_x - RIGHT_EAR_X)**2 + (sound_y - RIGHT_EAR_Y)**2)

    # ゲイン計算
    gain_left = 1.0 / (distance_left**2 + 1e-6)
    gain_right = 1.0 / (distance_right**2 + 1e-6)

    # 各音源の波形を生成
    total_left_wave = np.zeros_like(t)
    total_right_wave = np.zeros_like(t)

    for source in SOUND_SOURCES:
        # 基本波形を生成
        source_wave = np.zeros_like(t)
        for freq in source.freqs:
            source_wave += np.sin(2 * np.pi * freq * t)

        # 周波数正規化
        if source.freqs:
            source_wave /= len(source.freqs)

        # 左右チャンネルに個別の距離減衰を適用
        total_left_wave += source_wave * gain_left
        total_right_wave += source_wave * gain_right

    # 音源数で正規化
    if SOUND_SOURCES:
        total_left_wave /= len(SOUND_SOURCES)
        total_right_wave /= len(SOUND_SOURCES)

    # ステレオ信号の作成と最終音量調整
    stereo = np.column_stack([total_left_wave, total_right_wave])

    return sound.Sound(value=stereo, sampleRate=SAMPLE_RATE, stereo=True, hamming=True)

# 加速度データとドット位置データを統合してDataFrameを作成する共通関数
def create_accel_dot_dataframe(accel_data, red_dot_positions, green_dot_positions, timestamps):
    """加速度データとドット位置データを統合してDataFrameを作成"""
    df_data = []

    if not accel_data or not timestamps:
        print("Warning: accel_data または timestamps が空です")
        return pd.DataFrame()

    # 加速度データを時間でソート
    accel_data_sorted = sorted(accel_data, key=lambda x: x[0])

    print(f"Debug: accel_data_sorted の最初と最後の要素: {accel_data_sorted[0]} ... {accel_data_sorted[-1]}")
    print(f"Debug: timestamps の範囲: {min(timestamps):.6f} - {max(timestamps):.6f}")

    # ドット位置データを加速度データに同期
    for i, (timestamp_us, x, y, z) in enumerate(accel_data_sorted):
        # M5Stackの相対時間（マイクロ秒）を秒に変換
        accel_time_s = timestamp_us / 1000000.0

        # PsychoPyの試行開始からの時間に対応させる
        # 最も近い時刻のドット位置を見つける
        if timestamps:
            # timestampsは試行開始からの時間なので、accel_time_sと比較
            closest_idx = min(range(len(timestamps)), 
                            key=lambda j: abs(timestamps[j] - accel_time_s))

            red_pos = red_dot_positions[closest_idx] if closest_idx < len(red_dot_positions) else [0, 0]
            green_pos = green_dot_positions[closest_idx] if closest_idx < len(green_dot_positions) else [0, 0]

            # デバッグ情報（最初の数個のみ）
            if i < 5:
                print(f"Debug[{i}]: accel_time={accel_time_s:.6f}, closest_psychopy_time={timestamps[closest_idx]:.6f}, red_pos={red_pos}, green_pos={green_pos}")
        else:
            red_pos = [0, 0]
            green_pos = [0, 0]

        df_data.append([
            accel_time_s, accel_time_s, x, y, z,  # psychopy_timeとaccel_timeを同じ値に
            red_pos[0], red_pos[1],
            green_pos[0], green_pos[1]
        ])

    return pd.DataFrame(df_data, columns=[
        'psychopy_time', 'accel_time', 'accel_x', 'accel_y', 'accel_z',
        'red_dot_mean_x', 'red_dot_mean_y', 
        'green_dot_mean_x', 'green_dot_mean_y'
    ])

# 統一されたグラフ作成・表示関数
def create_and_show_acceleration_graph(df, trial_idx, graph_path, communication_type=''):
    """加速度データとドット位置の統一グラフを作成・表示"""
    if df.empty:
        print(f"Trial {trial_idx}: DataFrame is empty. Skipping graph.")
        return

    plt.figure(figsize=(15, 8))

    # 左側のY軸（加速度データ）
    ax1 = plt.gca()
    line1 = ax1.plot(df['psychopy_time'], df['accel_x'], 'b-', linewidth=1, label='Accel X [g]')
    line2 = ax1.plot(df['psychopy_time'], df['accel_y'], 'g-', linewidth=1, label='Accel Y [g]')
    line3 = ax1.plot(df['psychopy_time'], df['accel_z'], 'r-', linewidth=1, label='Accel Z [g]')

    ax1.set_xlabel('PsychoPy Time [s]')
    ax1.set_ylabel('Acceleration [g]', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True)
    ax1.set_title(f'Trial {trial_idx}: {communication_type} Accelerometer Data and Dot Position')

    # 右側のY軸（ドット位置データ）
    ax2 = ax1.twinx()
    line4 = ax2.plot(df['psychopy_time'], df['red_dot_mean_x'], 'orange', linewidth=2, linestyle='--', label='Red Dot X [pix]')
    line5 = ax2.plot(df['psychopy_time'], df['green_dot_mean_x'], 'darkgreen', linewidth=2, linestyle='--', label='Green Dot X [pix]')

    ax2.set_ylabel('Dot Position X [pix]', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # 凡例を統合
    lines = line1 + line2 + line3 + line4 + line5
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.tight_layout()

    # グラフをファイルに保存
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"グラフを保存しました: {graph_path}")

    # グラフを表示
    plt.show()

# 加速度グラフを保存（修正版）
def save_acceleration_graph_from_data(accel_data, red_dot_positions, green_dot_positions, timestamps, trial_idx, log_dir):
    """WiFi経由で受信した加速度データとドット座標からグラフを生成"""
    try:
        if not accel_data or not timestamps:
            print(f"Trial {trial_idx}: No data available. Skipping save.")
            return

        # 加速度センサデータを別ファイルに保存
        accel_csv_path = save_accelerometer_data_only(accel_data, trial_idx, log_dir)

        # ランダムドットデータを別ファイルに保存（生データ完全保持）
        dot_csv_path = save_random_dot_data_only(red_dot_positions, green_dot_positions, timestamps, trial_idx, log_dir)

        # 統合データファイル作成（設定でONの場合のみ）
        if SAVE_COMBINED_DATA:
            # DataFrameを作成
            df = create_accel_dot_dataframe(accel_data, red_dot_positions, green_dot_positions, timestamps)

            if df.empty:
                print(f"Trial {trial_idx}: DataFrame is empty. Skipping combined data and graph.")
                return

            # ファイルパス準備
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            graph_path = os.path.join(log_dir, f'{timestamp}_accel_log_trial_{trial_idx}.png')
            csv_path = os.path.join(log_dir, f'{timestamp}_accel_log_trial_{trial_idx}.csv')

            # グラフを作成・表示
            create_and_show_acceleration_graph(df, trial_idx, graph_path, 'WiFi')

            # 統合CSVファイルも保存（レガシー互換性のため）
            df.to_csv(csv_path, index=False)
            print(f"統合データファイルを保存しました: {csv_path}")
        else:
            print("統合データファイル出力はOFFに設定されています")

    except Exception as e:
        print(f"Error saving data for trial {trial_idx}: {e}")

def save_initial_dot_positions(red_initial_pos, green_initial_pos, trial_idx, log_dir, file_timestamp):
    """ランダムドットの初期位置をCSVファイルに保存（指定されたタイムスタンプを使用）"""
    try:
        csv_path = os.path.join(log_dir, f'{file_timestamp}_initial_dot_positions_trial_{trial_idx}.csv')

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行
            writer.writerow(['dot_index', 'red_x', 'red_y', 'green_x', 'green_y'])

            # 各ドットの初期位置を保存
            for i in range(len(red_initial_pos)):
                red_pos = red_initial_pos[i] if i < len(red_initial_pos) else [0, 0]
                green_pos = green_initial_pos[i] if i < len(green_initial_pos) else [0, 0]
                writer.writerow([i, red_pos[0], red_pos[1], green_pos[0], green_pos[1]])

        print(f"ランダムドットの初期位置を保存しました: {csv_path}")
        print(f"ドット数: {len(red_initial_pos)}")
        return csv_path

    except Exception as e:
        print(f"初期位置データ保存エラー: {e}")
        return None

def save_accelerometer_data_only(accel_data, trial_idx, log_dir, file_timestamp):
    """加速度センサデータのみを別ファイルに保存（accel_time基準、指定されたタイムスタンプを使用）"""
    try:
        csv_path = os.path.join(log_dir, f'{file_timestamp}_accel_sensor_trial_{trial_idx}.csv')

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行（加速度センサデータのみ）
            writer.writerow(['accel_time', 'accel_x', 'accel_y', 'accel_z'])

            # 加速度データを時間でソート
            accel_data_sorted = sorted(accel_data, key=lambda x: x[0])

            # データ行（M5Stack時間をそのまま使用）
            for timestamp_us, x, y, z in accel_data_sorted:
                timestamp_s = timestamp_us / 1000000.0  # マイクロ秒を秒に変換
                writer.writerow([timestamp_s, x, y, z])

        print(f"加速度センサデータを保存しました: {csv_path}")
        print(f"データポイント数: {len(accel_data)}")
        return csv_path

    except Exception as e:
        print(f"加速度センサデータ保存エラー: {e}")
        return None

def save_random_dot_data_only(red_dot_positions, green_dot_positions, timestamps, trial_idx, log_dir, file_timestamp):
    """ランダムドットデータのみを別ファイルに保存（生データ完全保持、元のタイムスタンプ使用、指定されたタイムスタンプを使用）"""
    try:
        csv_path = os.path.join(log_dir, f'{file_timestamp}_random_dot_trial_{trial_idx}.csv')

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行（ランダムドットデータのみ）
            writer.writerow(['psychopy_time', 'red_dot_mean_x', 'red_dot_mean_y', 'green_dot_mean_x', 'green_dot_mean_y'])

            if not timestamps or not red_dot_positions or not green_dot_positions:
                print("Warning: No dot position data available")
                return csv_path

            # 生データをそのまま保存（正規化や間引き処理は行わない）
            for i in range(len(timestamps)):
                if i < len(red_dot_positions) and i < len(green_dot_positions):
                    red_pos = red_dot_positions[i] if i < len(red_dot_positions) else [0, 0]
                    green_pos = green_dot_positions[i] if i < len(green_dot_positions) else [0, 0]

                    # 元のタイムスタンプと位置データをそのまま保存
                    writer.writerow([timestamps[i], red_pos[0], red_pos[1], green_pos[0], green_pos[1]])

        print(f"ランダムドット生データを保存しました: {csv_path}")
        print(f"データポイント数: {len(timestamps)}")
        return csv_path

    except Exception as e:
        print(f"ランダムドットデータ保存エラー: {e}")
        return None

def save_serial_acceleration_data_and_graph(accel_data, red_dot_positions, green_dot_positions, timestamps, trial_idx, log_dir, file_timestamp):
    """シリアル経由で受信した加速度データをCSVファイルに保存"""
    try:
        # 加速度センサデータを別ファイルに保存
        accel_csv_path = save_accelerometer_data_only(accel_data, trial_idx, log_dir, file_timestamp)

        # ランダムドットデータを別ファイルに保存（生データ完全保持）
        dot_csv_path = save_random_dot_data_only(red_dot_positions, green_dot_positions, timestamps, trial_idx, log_dir, file_timestamp)

        # 統合データファイル作成（設定でONの場合のみ）
        if SAVE_COMBINED_DATA:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(log_dir, f'{timestamp}_accel_log_serial_trial_{trial_idx}.csv')

            # DataFrameを作成
            df = create_accel_dot_dataframe(accel_data, red_dot_positions, green_dot_positions, timestamps)

            if df.empty:
                print(f"Trial {trial_idx}: No data available. Skipping combined data and graph.")
                return

            # 統合CSVファイルに保存
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # ヘッダー行
                writer.writerow(['psychopy_time', 'accel_time', 'accel_x', 'accel_y', 'accel_z', 'red_dot_mean_x', 'red_dot_mean_y', 'green_dot_mean_x', 'green_dot_mean_y'])

                # データ行
                for _, row in df.iterrows():
                    writer.writerow(row.tolist())

            print(f"統合データファイルを保存しました: {csv_path}")

            # グラフを作成・表示
            graph_path = os.path.join(log_dir, f'{file_timestamp}_accel_log_serial_trial_{trial_idx}.png')
            create_and_show_acceleration_graph(df, trial_idx, graph_path, 'Serial')
        else:
            print("統合データファイル出力はOFFに設定されています")

        print(f"データポイント数: {len(accel_data)}")

    except Exception as e:
        print(f"データ保存・グラフ作成エラー: {e}")

def save_serial_acceleration_data(accel_data, red_dot_positions, green_dot_positions, timestamps, trial_idx, log_dir):
    """シリアル経由で受信した加速度データをCSVファイルに保存（ドット位置データ付き）"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(log_dir, f'{timestamp}_accel_log_serial_trial_{trial_idx}.csv')

        # DataFrameを作成
        df_data = []

        # 加速度データを時間でソート
        accel_data_sorted = sorted(accel_data, key=lambda x: x[0])

        # ドット位置データを加速度データに同期
        for i, (timestamp_us, x, y, z) in enumerate(accel_data_sorted):
            timestamp_s = timestamp_us / 1000000.0  # マイクロ秒を秒に変換

            # 最も近い時刻のドット位置を見つける
            if timestamps:
                closest_idx = min(range(len(timestamps)), 
                                key=lambda i: abs(timestamps[i] - timestamp_s))
                red_pos = red_dot_positions[closest_idx] if closest_idx < len(red_dot_positions) else [0, 0]
                green_pos = green_dot_positions[closest_idx] if closest_idx < len(green_dot_positions) else [0, 0]
            else:
                red_pos = [0, 0]
                green_pos = [0, 0]

            df_data.append([
                timestamp_s, timestamp_s, x, y, z,  # accel_timeとpsychopy_timeを両方追加
                red_pos[0], red_pos[1],
                green_pos[0], green_pos[1]
            ])

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # ヘッダー行（加速度時間カラムを追加）
            writer.writerow(['psychopy_time', 'accel_time', 'accel_x', 'accel_y', 'accel_z', 'red_dot_mean_x', 'red_dot_mean_y', 'green_dot_mean_x', 'green_dot_mean_y'])

            # データ行
            for row in df_data:
                writer.writerow(row)

        print(f"シリアル加速度データを保存しました: {csv_path}")
        print(f"データポイント数: {len(accel_data)}")

    except Exception as e:
        print(f"シリアル加速度データ保存エラー: {e}")

# ------------------------------------------------------------------
# 5. ログファイル設定（最初の試行開始時に作成）
# ------------------------------------------------------------------
main_log_fh = None
main_log_csv = None
MAIN_LOG_PATH = None


# ------------------------------------------------------------------
# 6. メイン実験ループ
# ------------------------------------------------------------------
# 通信方式に応じて初期化
udp_comm = None
serial_comm = None

if COMMUNICATION_MODE == 'WIFI':
    print("WiFi通信モードで開始します")
    udp_comm = UDPCommunicator(M5_IP, M5_PORT, PC_PORT)
    udp_comm.start()

    # 通信の確立を待つ
    print("Waiting for UDP connection...")
    time.sleep(2.0)

    if not udp_comm.running:
        print("Failed to start UDP communicator. Exiting.")
        core.quit()

elif COMMUNICATION_MODE == 'SERIAL':
    print("シリアル通信モードで開始します")
    serial_comm = SerialCommunicator(SERIAL_PORT, SERIAL_BAUDRATE)
    if not serial_comm.start():
        print("シリアル通信の開始に失敗しました。プログラムを終了します。")
        core.quit()
    print("シリアル通信が正常に開始されました")

else:
    print(f"無効な通信モード: {COMMUNICATION_MODE}")
    print("COMMUNICATION_MODEを 'WIFI' または 'SERIAL' に設定してください")
    core.quit()

# GVSコントローラーの初期化
gvs_controller = None
if USE_GVS:
    print("GVS制御を初期化中...")
    gvs_controller = GVSController(GVS_SERIAL_PORT, GVS_BAUDRATE)
    if gvs_controller.connect():
        print("GVS制御が正常に初期化されました")
        # デフォルト振幅を設定（必要に応じて調整）
        gvs_controller.set_amplitude(127)  # 半分の強度
    else:
        print("GVS制御の初期化に失敗しました。GVS刺激なしで続行します。")
        gvs_controller = None
else:
    print("GVS刺激は無効です")

# 光同期コントローラーの初期化
light_sync_controller = None
if USE_LIGHT_SYNC:
    print("光同期制御を初期化中...")
    light_sync_controller = LightSyncController(LIGHT_SYNC_SERIAL_PORT, LIGHT_SYNC_BAUDRATE)
    if light_sync_controller.connect():
        print("光同期制御が正常に初期化されました")
        # 初期化後に確実にLOW状態にする
        light_sync_controller.force_stop()
    else:
        print("光同期制御の初期化に失敗しました。光同期なしで続行します。")
        light_sync_controller = None

    # 同期用正方形を確実に黒で初期化
    sync_square.fillColor = 'black'
    sync_square.draw()
    win.flip()
    time.sleep(0.1)  # 確実に表示
else:
    print("光同期機能は無効です")

# 音響設定の表示
print(f"\n音響設定:")
print(f"- ソースモード: {AUDIO_SOURCE_MODE}")
if AUDIO_SOURCE_MODE == 'mp3':
    print(f"- 赤ドット同期MP3: {MP3_FILE_RED}")
    print(f"- 緑ドット同期MP3: {MP3_FILE_GREEN}")
    print(f"- ループ再生: {'有効' if MP3_LOOP else '無効'}")
    import os
    red_exists = os.path.exists(MP3_FILE_RED)
    green_exists = os.path.exists(MP3_FILE_GREEN)
    print(f"- 赤ファイル状態: {'存在確認済み' if red_exists else 'ファイルが見つかりません'}")
    print(f"- 緑ファイル状態: {'存在確認済み' if green_exists else 'ファイルが見つかりません'}")
    if not (red_exists and green_exists):
        print("  → 不足ファイルがある場合、シミュレーション音響にフォールバック")
else:
    print(f"- パンニングモード: {PANNING_MODE}")

# データ保存設定の表示
print(f"\nデータ保存設定:")
print(f"- 加速度センサデータ: 個別ファイルに保存 (accel_sensor_trial_N.csv)")
print(f"- ランダムドットデータ: 個別ファイルに保存 (random_dot_trial_N.csv)")
print(f"- ランダムドット初期位置: 個別ファイルに保存 (initial_dot_positions_trial_N.csv)")
print(f"- 統合データファイル: {'有効' if SAVE_COMBINED_DATA else '無効'}")
print(f"- experiment_log: 刺激開始時間を記録")
print(f"- 最大試行回数: {MAX_TRIALS}")
print()

experiment_running = True
trial_idx = 1
response_mapping = {'r': 'red', 'g': 'green'}
stereo_snd = None
file_timestamp = None  # 全試行で使用するタイムスタンプ

try:
    while experiment_running and trial_idx <= MAX_TRIALS:
        print(f"\n=== 試行 {trial_idx}/{MAX_TRIALS} 開始 ===")

        # ----- 光同期正方形を黒で初期化（試行開始時の安全措置） -----
        if USE_LIGHT_SYNC:
            sync_square.fillColor = 'black'

        # ----- この試行のための設定 -----
        cond_type = random.choice(['red', 'green'])
        if FORCE_COND:
            cond_type = FORCE_COND

        print(f"\ncond_type: {cond_type}")
        print(f"制御設定: SINGLE_COLOR_DOT={SINGLE_COLOR_DOT}, VISUAL_REVERSE={VISUAL_REVERSE}, AUDIO_REVERSE={AUDIO_REVERSE}, GVS_REVERSE={GVS_REVERSE}")

        # 音響情報の記録用変数（AUDIO_REVERSEに応じて反転）
        if AUDIO_REVERSE:
            sync_red = (cond_type == 'green')  # 反対にする
            print(f"音響反転: cond_type={cond_type} → 音響同期={sync_red}")
        else:
            sync_red = (cond_type == 'red')

        audio_sync_type = 'red_sync' if sync_red else 'green_sync'

        if AUDIO_SOURCE_MODE == 'mp3':
            audio_file_used = MP3_FILE_RED if sync_red else MP3_FILE_GREEN
        else:
            audio_file_used = 'simulation'

        # ドット位置と時間の履歴を保存するリスト
        red_dot_positions = []
        green_dot_positions = []
        timestamps = []

        # ----- ドット位置と時間の初期化 -----
        red_current_pos, green_current_pos = init_positions(), init_positions()
        red_base_x, green_base_x = red_current_pos[:, 0].copy(), green_current_pos[:, 0].copy()
        last_t = 0.0

        # ----- 刺激開始時間を記録 -----
        stimulus_start_time = datetime.now()
        stimulus_start_timestamp = stimulus_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        stimulus_start_time_str = stimulus_start_time.strftime('%H:%M:%S.%f')[:-3]

        # 最初の試行でファイル名用のタイムスタンプを設定（全試行で同じタイムスタンプを使用）
        if trial_idx == 1:
            file_timestamp = stimulus_start_time.strftime('%Y%m%d_%H%M%S')

        print(f"Trial {trial_idx}: 刺激開始時間 = {stimulus_start_timestamp}")

        # 初期位置を保存
        save_initial_dot_positions(red_current_pos, green_current_pos, trial_idx, LOG_DIR, file_timestamp)

        # ----- サウンド準備 -----
        if stereo_snd and stereo_snd.status != constants.STOPPED:
            stereo_snd.stop()
        stereo_snd = build_audio_source(sync_red, PANNING_MODE)

        # ----- 黒い画面を確実に描画してから光同期準備 -----
        if USE_LIGHT_SYNC and light_sync_controller:
            # 黒い正方形を確実に描画
            sync_square.fillColor = 'black'
            sync_square.draw()
            win.flip()
            time.sleep(0.2)  # 黒い画面が確実に描画されるまで待機

            # 黒い画面が描画された後にArduinoに刺激検出準備コマンド送信
            print(f"Trial {trial_idx}: 光同期準備 - 刺激検出準備状態にセット")
            light_sync_controller.ready()

        # ----- ビープ音再生（試行開始） -----
        if USE_BEEP and beep_sound:
            beep_sound.play()
            print(f"Trial {trial_idx}: ビープ音再生（試行開始）")
            time.sleep(BEEP_SEP)  # 区切り音の間隔

        # ----- 測定開始 -----
        if COMMUNICATION_MODE == 'WIFI' and udp_comm and udp_comm.running:
            udp_comm.clear_data()
            udp_comm.start_measurement()
            time.sleep(0.05)  # 測定開始の確認
        elif COMMUNICATION_MODE == 'SERIAL' and serial_comm:
            # シリアル通信の場合は測定開始コマンドを送信
            print(f"Trial {trial_idx}: シリアル測定開始コマンドを送信")
            serial_comm.start_measurement()
            time.sleep(0.05)  # 測定開始の確認

        # ----- 刺激提示 & 応答取得 -----
        participant_response = 'no_response'
        rt = -1.0

        # 描画順序の初期設定（初回のみランダム決定）
        frame_count = 0
        red_first = random.random() < 0.5  # True: 赤先, False: 緑先

        # ----- 音再生直前：ランダムドットを初期位置でプロット -----
        # 初期位置（t=0）でのドット位置を設定
        phase_initial = 2 * np.pi * OSC_FREQ * 0.0  # t=0での位相
        x_osc_offset_initial = OSC_AMP * np.sin(phase_initial)

        # 初期位置を各ドットの座標に設定
        red_current_pos[:, 0] = red_base_x + x_osc_offset_initial
        green_current_pos[:, 0] = green_base_x - x_osc_offset_initial
        red_dots.xys = red_current_pos
        green_dots.xys = green_current_pos

        # 初期位置でドットを描画（制御変数に基づく）
        if SINGLE_COLOR_DOT:
            if VISUAL_REVERSE:
                # VISUAL_REVERSE=True: cond_typeの色のみ表示
                if cond_type == 'red':
                    red_dots.draw()
                else:
                    green_dots.draw()
            else:
                # VISUAL_REVERSE=False: cond_typeの反対色のみ表示（デフォルト）
                if cond_type == 'red':
                    green_dots.draw()
                else:
                    red_dots.draw()
        else:
            # 通常の描画
            if (red_first and frame_count % 2 == 0) or (not red_first and frame_count % 2 == 1):
                red_dots.draw()
                green_dots.draw()
            else:
                green_dots.draw()
                red_dots.draw()

        # 光同期用正方形も描画
        if USE_LIGHT_SYNC:
            sync_square.fillColor = 'white'
            sync_square.draw()

        # 初期フレームを表示
        win.flip()
        time.sleep(0.2)  # プロジェクターの投影にかかる時間だけ待機

        # ----- GVS刺激開始 -----
        if USE_GVS and gvs_controller:
            # GVS_REVERSEに応じて刺激タイプを決定
            if GVS_REVERSE:
                gvs_condition = 'green' if cond_type == 'red' else 'red'
                print(f"Trial {trial_idx}: GVS刺激開始 ({cond_type} → {gvs_condition}に反転)")
            else:
                gvs_condition = cond_type
                print(f"Trial {trial_idx}: GVS刺激開始 ({gvs_condition})")
            gvs_controller.start_stimulation(gvs_condition)  # 赤なら同相、緑なら逆相
            time.sleep(0.05)  # GVS開始の確認

        # ----- 音刺激開始 -----
        stereo_snd.play()    # 音を再生開始
        trial_clock = core.Clock()
        trial_clock.reset()  # まずクロックをリセット

        while trial_clock.getTime() < TRIAL_DURATION:
            keys = event.getKeys(keyList=['r', 'g', 'escape'], timeStamped=trial_clock)

            if keys:
                key_name, rt = keys[0]
                if key_name == 'escape':
                    experiment_running = False
                    participant_response = 'escape_quit'  # ESCキーでの終了を明示
                    # ESCキーが押された場合は光同期出力を強制停止
                    if USE_LIGHT_SYNC and light_sync_controller:
                        print("ESCキー検出：光同期出力を強制停止")
                        light_sync_controller.force_stop()
                else:
                    participant_response = response_mapping.get(key_name, 'invalid')
                break

            now = trial_clock.getTime()
            dt  = now - last_t
            last_t = now

            # ドットの座標を更新
            if SCROLLING_MODE:
                virtual_h = WIN_H * VIRTUAL_HEIGHT_MULTIPLIER
                camera_y = ((now * FALL_SPEED) % virtual_h)

                temp_red_xys = red_current_pos.copy()
                temp_red_xys[:, 1] -= camera_y

                temp_green_xys = green_current_pos.copy()
                temp_green_xys[:, 1] -= camera_y

                temp_red_xys[:, 1] = ((temp_red_xys[:, 1] + virtual_h/2) % virtual_h) - virtual_h/2
                temp_green_xys[:, 1] = ((temp_green_xys[:, 1] + virtual_h/2) % virtual_h) - virtual_h/2

                phase = 2 * np.pi * OSC_FREQ * now
                x_osc_offset = OSC_AMP * np.sin(phase)
                temp_red_xys[:, 0] = red_base_x + x_osc_offset
                temp_green_xys[:, 0] = green_base_x - x_osc_offset

                red_dots.xys = temp_red_xys
                green_dots.xys = temp_green_xys
            else:
                red_current_pos[:, 1] -= FALL_SPEED * dt
                green_current_pos[:, 1] -= FALL_SPEED * dt

                is_below_screen_red = red_current_pos[:, 1] < Y_MIN
                red_current_pos[is_below_screen_red, 1] += WIN_H

                is_below_screen_green = green_current_pos[:, 1] < Y_MIN
                green_current_pos[is_below_screen_green, 1] += WIN_H

                phase = 2 * np.pi * OSC_FREQ * now
                x_osc_offset = OSC_AMP * np.sin(phase)
                red_current_pos[:, 0] = red_base_x + x_osc_offset
                green_current_pos[:, 0] = green_base_x - x_osc_offset

                red_dots.xys = red_current_pos
                green_dots.xys = green_current_pos

            # ドット位置の履歴を保存
            red_mean_xy = np.mean(red_dots.xys, axis=0)
            green_mean_xy = np.mean(green_dots.xys, axis=0)

            red_dot_positions.append(red_mean_xy.tolist())
            green_dot_positions.append(green_mean_xy.tolist())
            timestamps.append(now)

            # ドット描画の制御
            if SINGLE_COLOR_DOT:
                if VISUAL_REVERSE:
                    # VISUAL_REVERSE=True: cond_typeの色のみ表示
                    if cond_type == 'red':
                        red_dots.draw()
                    else:
                        green_dots.draw()
                else:
                    # VISUAL_REVERSE=False: cond_typeの反対色のみ表示（デフォルト）
                    if cond_type == 'red':
                        green_dots.draw()
                    else:
                        red_dots.draw()
            else:
                # 通常の描画（フレーム毎に交互に描画順を変更）
                if (red_first and frame_count % 2 == 0) or (not red_first and frame_count % 2 == 1):
                    # 赤ドットを先に描画
                    red_dots.draw()
                    green_dots.draw()
                else:
                    # 緑ドットを先に描画
                    green_dots.draw()
                    red_dots.draw()

            frame_count += 1

            # 中心線を描画（オプション）
            if USE_CENTER_LINES:
                # 中心線の位置をオシレーションに合わせて更新
                phase = 2 * np.pi * OSC_FREQ * now
                x_osc_offset = OSC_AMP * np.sin(phase)
                red_center_x = x_osc_offset
                green_center_x = -x_osc_offset

                if center_line_red:
                    center_line_red.start = (red_center_x, -WIN_H//2)
                    center_line_red.end = (red_center_x, WIN_H//2)
                    center_line_red.draw()

                if center_line_green:
                    center_line_green.start = (green_center_x, -WIN_H//2)
                    center_line_green.end = (green_center_x, WIN_H//2)
                    center_line_green.draw()

            # 光同期用白正方形の描画（刺激中は白、ランダムドットより上に描画）
            if USE_LIGHT_SYNC:
                sync_square.fillColor = 'white'  # 刺激中は白
                sync_square.draw()

            win.flip()

        if stereo_snd and stereo_snd.status != constants.STOPPED:
            stereo_snd.stop()

        # ----- GVS刺激停止 -----
        if USE_GVS and gvs_controller:
            print(f"Trial {trial_idx}: GVS刺激停止")
            gvs_controller.stop_stimulation()
            time.sleep(0.05)  # GVS停止の確認

        # ----- 光同期：同期用正方形を黒に戻す -----
        if USE_LIGHT_SYNC:
            # 同期用正方形を黒に戻す
            sync_square.fillColor = 'black'
            print(f"Trial {trial_idx}: 同期用正方形を黒に戻しました")

        # ----- ビープ音再生（試行終了） -----
        if USE_BEEP and beep_sound:
            time.sleep(BEEP_SEP)  # 区切り音の間隔
            beep_sound.play()
            print(f"Trial {trial_idx}: ビープ音再生（試行終了）")

        # 最初の試行でexperiment_logファイルを初期化
        if trial_idx == 1:
            MAIN_LOG_FILENAME = f'{file_timestamp}_experiment_log.csv'
            MAIN_LOG_PATH = os.path.join(LOG_DIR, MAIN_LOG_FILENAME)
            try:
                main_log_fh = open(MAIN_LOG_PATH, 'w', newline='', encoding='utf-8')
                main_log_csv = csv.writer(main_log_fh)
                header = [
                    'trial', 'panning_mode', 'scrolling_mode', 'condition', 'response', 'RT',
                    'stimulus_start_time', 'stimulus_start_timestamp',  # 刺激開始時間の追加
                    'single_color_dot', 'visual_reverse', 'audio_reverse', 'gvs_reverse',  # 新しい制御変数
                    'audio_source_mode', 'audio_sync_type', 'audio_file_used',
                    'win_width', 'win_height', 'n_dots', 'dot_size', 'fall_speed',
                    'dot_osc_freq', 'dot_osc_amp', 'audio_freqs', 'sound_initial_x', 'sound_initial_y', 'sound_osc_amplitude',
                    'left_ear_pos', 'right_ear_pos', 'distance_attenuation', 'min_distance_gain',
                    'sample_rate', 'max_itd_s'
                ]
                main_log_csv.writerow(header)
                print(f"experiment_logファイルを作成しました: {MAIN_LOG_PATH}")
            except IOError as e:
                print(f"ログファイルを開けませんでした: {MAIN_LOG_PATH}")
                print(f"エラー: {e}")
                core.quit()

        # メインログに記録（ESCキーが押された場合でも記録）
        audio_freqs_str = " | ".join([",".join(map(str, s.freqs)) for s in SOUND_SOURCES])
        log_data = [
            trial_idx, PANNING_MODE, SCROLLING_MODE, cond_type, participant_response, f"{rt:.3f}",
            stimulus_start_time_str, stimulus_start_timestamp,  # 刺激開始時間を追加
            SINGLE_COLOR_DOT, VISUAL_REVERSE, AUDIO_REVERSE, GVS_REVERSE,  # 新しい制御変数を追加
            AUDIO_SOURCE_MODE, audio_sync_type, audio_file_used,
            WIN_W, WIN_H, N_DOTS, DOT_SIZE, FALL_SPEED, OSC_FREQ,
            OSC_AMP, audio_freqs_str, SOUND_INITIAL_X, SOUND_INITIAL_Y, SOUND_OSC_AMPLITUDE,
            f"({LEFT_EAR_X},{LEFT_EAR_Y})", f"({RIGHT_EAR_X},{RIGHT_EAR_Y})",
            DISTANCE_ATTENUATION, MIN_DISTANCE_GAIN, SAMPLE_RATE, MAX_ITD_S
        ]
        main_log_csv.writerow(log_data)
        main_log_fh.flush()

        # ----- 測定停止とデータ取得 -----
        accel_data = []

        if COMMUNICATION_MODE == 'WIFI' and udp_comm and udp_comm.running:
            # WiFi通信の場合
            udp_comm.stop_measurement()
            time.sleep(0.05)  # 測定停止の確認

            # データ要求とデータ受信
            print("Requesting acceleration data from M5Stack...")
            udp_comm.request_data()

            # データ受信完了を待機（最大10秒）
            print("Waiting for data reception to complete...")
            if udp_comm.wait_for_data_reception(timeout=10.0):
                print("Data reception completed successfully")
            else:
                print("Warning: Data reception may be incomplete")

            # 受信したデータを取得
            accel_data = udp_comm.get_accel_data()
            print(f"Retrieved {len(accel_data)} acceleration data points via WiFi")

            # グラフを生成
            save_acceleration_graph_from_data(
                accel_data, red_dot_positions, green_dot_positions, 
                timestamps, trial_idx, LOG_DIR, file_timestamp
            )

        elif COMMUNICATION_MODE == 'SERIAL' and serial_comm:
            # シリアル通信の場合
            print("シリアル測定停止コマンドを送信")
            serial_comm.stop_measurement()
            time.sleep(0.1)  # 測定停止の確認

            # リアルタイムで蓄積されたデータを取得（request_data不要）
            accel_data = serial_comm.get_all_accel_data()
            print(f"Retrieved {len(accel_data)} acceleration data points via Serial (realtime)")

            # バッファ状況を詳細に報告
            if len(accel_data) == 0:
                print("警告: 加速度データが全く受信されていません")
                print("M5Stackの状態を確認してください:")
                print("1. M5Stackが正常に起動しているか")
                print("2. IMUセンサーが正常に動作しているか")
                print("3. シリアル通信モードが有効になっているか")
            else:
                print(f"最初のデータ: {accel_data[0]}")
                print(f"最後のデータ: {accel_data[-1]}")

                # 時間オフセットを調整（M5Stackの相対時間を試行開始時間に合わせる）
                if accel_data and timestamps:
                    # M5Stackの最初のタイムスタンプが0に近い場合は、そのまま使用
                    # そうでない場合は、最小タイムスタンプを0として正規化
                    min_accel_time = min(data[0] for data in accel_data) / 1000000.0
                    print(f"M5Stack最小時間: {min_accel_time:.6f}秒")

                    # M5Stack時間を正規化（最小時間を0として開始）
                    normalized_accel_data = []
                    for timestamp_us, x, y, z in accel_data:
                        normalized_time_us = timestamp_us - (min_accel_time * 1000000)
                        normalized_accel_data.append((normalized_time_us, x, y, z))

                    accel_data = normalized_accel_data
                    print(f"正規化後の最初のデータ: {accel_data[0]}")
                    print(f"正規化後の最後のデータ: {accel_data[-1]}")

            # シリアルデータをCSVファイルに保存し、グラフを表示
            if accel_data:
                save_serial_acceleration_data_and_graph(accel_data, red_dot_positions, green_dot_positions, timestamps, trial_idx, LOG_DIR, file_timestamp)

            # 次の試行のためにバッファをクリア
            serial_comm.get_accel_data()

        # 試行インデックスを増加
        trial_idx += 1

        # 最大試行回数に達した場合は実験終了
        if trial_idx > MAX_TRIALS:
            print(f"\n最大試行回数 {MAX_TRIALS} に達しました。実験を終了します。")
            experiment_running = False
            break

        # ESCキーが押された場合は実験終了
        if not experiment_running:
            print("\nESCキーが押されました。実験を終了します。")
            break

        # ----- ITI -----
        fixation = visual.TextStim(win, text='+', color='white', height=30)
        fixation.draw()
        # ITI中は光同期用正方形を黒で表示
        if USE_LIGHT_SYNC:
            sync_square.fillColor = 'black'
            sync_square.draw()
        win.flip()
        iti_clock = core.Clock()
        while iti_clock.getTime() < ITI:
            if event.getKeys(['escape']):
                experiment_running = False
                # ITI中にESCが押された場合も光同期出力を強制停止
                if USE_LIGHT_SYNC and light_sync_controller:
                    print("ITI中ESCキー検出：光同期出力を強制停止")
                    light_sync_controller.force_stop()
                break
            core.wait(0.01)
        if not experiment_running: break

except Exception as e:
    print(f"エラーが発生しました: {e}")
finally:
    # 通信を安全に閉じる
    if COMMUNICATION_MODE == 'WIFI' and udp_comm:
        udp_comm.stop()
        print("WiFi通信を停止しました")
    elif COMMUNICATION_MODE == 'SERIAL' and serial_comm:
        serial_comm.stop()
        print("シリアル通信を停止しました")

    # GVS制御を安全に停止
    if USE_GVS and gvs_controller:
        gvs_controller.stop_stimulation()
        gvs_controller.disconnect()
        print("GVS制御を停止しました")

    # 光同期制御を安全に停止
    if USE_LIGHT_SYNC and light_sync_controller:
        # 実験終了時は強制停止コマンドを送信
        print("実験終了：光同期出力を強制停止")
        light_sync_controller.force_stop()
        time.sleep(0.1)  # コマンド送信完了を待機
        light_sync_controller.disconnect()
        print("光同期制御を停止しました")

    if stereo_snd and stereo_snd.status != constants.STOPPED:
        stereo_snd.stop()
    if 'main_log_fh' in locals() and main_log_fh and not main_log_fh.closed:
        main_log_fh.close()
        print(f"メインログを保存しました: {os.path.abspath(MAIN_LOG_PATH)}")
    if 'win' in locals() and win:
        win.close()

    # 実験終了理由を表示
    if not experiment_running:
        print("実験がESCキーで中断されました。")
        print(f"完了した試行数: {trial_idx}")
    else:
        print("実験が正常に完了しました。")

    core.quit()
    print("実験を終了しました。")
