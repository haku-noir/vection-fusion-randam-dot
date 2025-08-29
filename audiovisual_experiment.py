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
import serial
import threading
import time
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 2. 実験パラメータ（自由に変更可）
# ------------------------------------------------------------------
# M5Stackとのシリアル通信設定
SERIAL_PORT = '/dev/cu.wchusbserial556F0046031'
BAUD_RATE = 115200

# 音源の情報を保持するクラス
class SoundSource:
    """音源の周波数スペクトルと基本位置を保持するクラス"""
    def __init__(self, freqs, base_pos):
        if not isinstance(freqs, list):
            self.freqs = [freqs]
        else:
            self.freqs = freqs
        self.base_pos = base_pos

# ★★★ 音響モードを選択 ★★★
# 'volume': 音量差パンニング
# 'itd'   : 時間差 (ITD) パンニング
# 'both'  : 音量差と時間差の両方
PANNING_MODE = 'both'

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
# 仮想的な「世界空間」を最小値と最大値で定義
WORLD_SPACE_MIN = -100.0 # 世界空間の左端
WORLD_SPACE_MAX = 100.0  # 世界空間の右端
OSC_AMP_WORLD = 80.0     # 世界空間における音の振動の振幅

# 距離減衰に関するパラメータを追加
LISTENER_POS_Z = 50.0      # リスナーのZ座標（音源平面からの距離）
DISTANCE_ATTENUATION = 5000.0 # 距離減衰の係数（大きいほど減衰が緩やか）
MIN_DISTANCE_GAIN = 0.05   # 距離が離れた際の最小ゲイン（音量が0にならないように）

# SoundSourceオブジェクトを周波数のリスト(freqs)で定義
SOUND_SOURCES = [
    # SoundSource(freqs=[440.00, 880.00], base_pos=-25.0), # ラ(A4) とそのオクターブ上
    SoundSource(freqs=[523.25], base_pos=0.0),          # ド(C5) のみ
    # SoundSource(freqs=[659.25, 1318.50], base_pos=25.0)   # ミ(E5) とそのオクターブ上
]
SAMPLE_RATE   = 44100        # サンプリングレート [Hz]
MAX_ITD_S     = 0.0007       # ITDの最大値 (秒)。'itd'または'both'モードで使用

# 試行
TRIAL_DURATION   = 60.0      # 各試行の刺激掲示時間 [s]
ITI              = 1.0       # 刺激間インターバル [s]

# ログ
LOG_DIR = PANNING_MODE
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
MAIN_LOG_FILENAME = f'{timestamp}_experiment_log.csv'
MAIN_LOG_PATH = os.path.join(LOG_DIR, MAIN_LOG_FILENAME)

# M5Stackのデータをバックグラウンドで読み込むためのクラス
class SerialReader(threading.Thread):
    def __init__(self, port, baudrate):
        super(SerialReader, self).__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.running = False
        self.lock = threading.Lock()
        self.data_buffer = deque() # スレッドセーフなdequeを使用
        self.daemon = True # メインスレッド終了時にこのスレッドも終了させる

    def run(self):
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2) # 接続が安定するのを待つ
            self.running = True
            print(f"Successfully connected to M5Stack on {self.port}")
        except serial.SerialException as e:
            print(f"Error: Could not connect to M5Stack on {self.port}. {e}")
            print("Accelerometer data will not be recorded.")
            self.running = False
            return

        while self.running:
            try:
                line = self.serial_connection.readline().decode('utf-8').strip()
                if line:
                    # X,Y,Zデータをパースしてバッファに追加
                    parts = [float(p) for p in line.split(',')]
                    if len(parts) == 3:
                        with self.lock:
                            self.data_buffer.append(tuple(parts))
            except (ValueError, UnicodeDecodeError):
                continue
            except serial.SerialException:
                self.running = False
                break

        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        print("SerialReader thread stopped.")

    def get_all_data(self):
        """バッファから全てのデータを取得し、バッファを空にする"""
        with self.lock:
            data = list(self.data_buffer)
            self.data_buffer.clear()
            return data

    def stop(self):
        self.running = False


# ------------------------------------------------------------------
# 3. ウィンドウと刺激の準備
# ------------------------------------------------------------------
win = visual.Window(size=WIN_SIZE, color=[0, 0, 0], units='pix',
                    fullscr=False, allowGUI=True) # フルスクリーンに変更

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
# 4. サウンド波形生成（モード切替対応）
# ------------------------------------------------------------------
def build_multi_stereo_sound(sync_to_red: bool, mode: str) -> sound.Sound:
    """
    複数の音源を世界空間に配置し、相対距離を保ったままグループとして
    振動させ、ステレオ音響を生成する。
    """
    t = np.linspace(0, TRIAL_DURATION, int(SAMPLE_RATE * TRIAL_DURATION), endpoint=False)

    base_positions = [s.base_pos for s in SOUND_SOURCES]
    min_base_pos = min(base_positions) if base_positions else 0
    max_base_pos = max(base_positions) if base_positions else 0

    osc_wave = np.sin(2 * np.pi * OSC_FREQ * t)

    group_shift_world = OSC_AMP_WORLD * osc_wave
    if not sync_to_red:
        group_shift_world *= -1

    predicted_left_edge = min_base_pos + group_shift_world
    predicted_right_edge = max_base_pos + group_shift_world

    left_overhang = WORLD_SPACE_MIN - predicted_left_edge
    left_overhang[left_overhang < 0] = 0
    right_overhang = WORLD_SPACE_MAX - predicted_right_edge
    right_overhang[right_overhang > 0] = 0

    final_group_shift = group_shift_world + left_overhang + right_overhang

    # --- 波形の合成 ---
    total_left_wave = np.zeros_like(t)
    total_right_wave = np.zeros_like(t)

    for source in SOUND_SOURCES:
        source_pos_world = source.base_pos + final_group_shift

        world_range = WORLD_SPACE_MAX - WORLD_SPACE_MIN
        if world_range == 0: world_range = 1
        final_pan = -1.0 + 2.0 * (source_pos_world - WORLD_SPACE_MIN) / world_range

        left_gain, right_gain = 1.0, 1.0
        delay_L_s, delay_R_s = 0.0, 0.0

        if mode in ['volume', 'both']:
            left_gain = np.sqrt(0.5 * (1 - final_pan))
            right_gain = np.sqrt(0.5 * (1 + final_pan))

        if mode in ['itd', 'both']:
            delay_L_s = np.maximum(0, final_pan) * MAX_ITD_S
            delay_R_s = np.maximum(0, -final_pan) * MAX_ITD_S

        t_left = t - delay_L_s
        t_right = t - delay_R_s

        source_wave_left = np.zeros_like(t)
        source_wave_right = np.zeros_like(t)
        for freq in source.freqs:
            source_wave_left += np.sin(2 * np.pi * freq * t_left)
            source_wave_right += np.sin(2 * np.pi * freq * t_right)

        source_wave_left *= left_gain
        source_wave_right *= right_gain

        distance = np.sqrt(source_pos_world**2 + LISTENER_POS_Z**2)
        distance_gain = DISTANCE_ATTENUATION / (distance**2 + 1e-6)
        final_distance_gain = MIN_DISTANCE_GAIN + distance_gain
        final_distance_gain = np.clip(final_distance_gain, 0.0, 1.0)

        source_wave_left *= final_distance_gain
        source_wave_right *= final_distance_gain

        if source.freqs:
            source_wave_left /= len(source.freqs)
            source_wave_right /= len(source.freqs)

        total_left_wave += source_wave_left
        total_right_wave += source_wave_right

    if SOUND_SOURCES:
        total_left_wave /= len(SOUND_SOURCES)
        total_right_wave /= len(SOUND_SOURCES)

    stereo = np.column_stack([total_left_wave, total_right_wave])
    stereo *= 0.9

    return sound.Sound(value=stereo, sampleRate=SAMPLE_RATE, stereo=True, hamming=True)

# 加速度グラフを保存
def save_acceleration_graph(log_path, trial_idx):
    """加速度ログを読み込み、グラフとして保存する"""
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(log_path)
        if df.empty:
            print(f"Trial {trial_idx}: Accelerometer log is empty. Skipping graph.")
            return

        # グラフを作成
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(df['psychopy_time'], df['accel_x'], label='X-axis', alpha=0.8)
        ax.plot(df['psychopy_time'], df['accel_y'], label='Y-axis', alpha=0.8)
        ax.plot(df['psychopy_time'], df['accel_z'], label='Z-axis', alpha=0.8)

        # ラベルとタイトルを設定
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s^2)")
        ax.set_title(f"Trial {trial_idx}: Accelerometer Data")
        ax.legend()
        ax.grid(True)

        # グラフをファイルとして保存
        graph_path = log_path.replace('.csv', '.png')
        plt.savefig(graph_path)
        plt.close(fig) # メモリを解放
        print(f"Saved accelerometer graph to {graph_path}")

    except FileNotFoundError:
        print(f"Error: Could not find log file {log_path} to create graph.")
    except Exception as e:
        print(f"An error occurred while creating graph for trial {trial_idx}: {e}")

# ------------------------------------------------------------------
# 5. ログファイルオープン
# ------------------------------------------------------------------
try:
    main_log_fh = open(MAIN_LOG_PATH, 'w', newline='', encoding='utf-8')
    main_log_csv = csv.writer(main_log_fh)
    header = [
        'trial', 'panning_mode', 'scrolling_mode', 'condition', 'response', 'RT',
        'win_width', 'win_height', 'n_dots', 'dot_size', 'fall_speed',
        'dot_osc_freq', 'dot_osc_amp', 'audio_freqs', 'audio_positions_world', 
        'world_space_min', 'world_space_max', 'sound_osc_amp_world', 
        'listener_pos_z', 'distance_attenuation', 'min_distance_gain',
        'sample_rate', 'max_itd_s'
    ]
    main_log_csv.writerow(header)
except IOError as e:
    print(f"ログファイルを開けませんでした: {MAIN_LOG_PATH}")
    print(f"エラー: {e}")
    core.quit()


# ------------------------------------------------------------------
# 6. メイン実験ループ
# ------------------------------------------------------------------
# シリアル通信スレッドを開始
serial_reader = SerialReader(SERIAL_PORT, BAUD_RATE)
serial_reader.start()

# スレッドが起動し、シリアルポートに接続するのを待つ
print("Waiting for serial connection...")
time.sleep(2.5) # Arduino側の準備とSerialReaderのsleep(2)より少し長く待つ

if not serial_reader.running:
    print("Failed to start serial reader. Continuing without accelerometer data.")

experiment_running = True
trial_idx = 1
response_mapping = {'r': 'red', 'g': 'green'}
stereo_snd = None

try:
    while experiment_running:
        # ----- この試行のための設定 -----
        cond_type = random.choice(['red', 'green'])

        # 試行ごとに加速度ログファイルを作成
        accel_log_fh = None
        accel_log_csv = None
        accel_log_path = None
        if serial_reader.running:
            try:
                accel_log_filename = f'{timestamp}_accel_log_trial_{trial_idx}.csv'
                accel_log_path = os.path.join(LOG_DIR, accel_log_filename)
                accel_log_fh = open(accel_log_path, 'w', newline='', encoding='utf-8')
                accel_log_csv = csv.writer(accel_log_fh)
                accel_log_csv.writerow(['psychopy_time', 'accel_x', 'accel_y', 'accel_z'])
            except IOError as e:
                print(f"加速度ログファイルを作成できませんでした: {e}")
                accel_log_fh = None

        # ----- ドット位置と時間の初期化 -----
        red_current_pos, green_current_pos = init_positions(), init_positions()
        red_base_x, green_base_x = red_current_pos[:, 0].copy(), green_current_pos[:, 0].copy()
        last_t = 0.0

        # ----- サウンド準備 -----
        sync_red = (cond_type == 'red')
        if stereo_snd and stereo_snd.status != constants.STOPPED:
            stereo_snd.stop()
        stereo_snd = build_multi_stereo_sound(sync_red, PANNING_MODE)

        # ----- 刺激提示 & 応答取得 -----
        participant_response = 'no_response'
        rt = -1.0

        trial_clock = core.Clock()
        stereo_snd.play()
        trial_clock.reset()

        while trial_clock.getTime() < TRIAL_DURATION:
            keys = event.getKeys(keyList=['r', 'g', 'escape'], timeStamped=trial_clock)

            if keys:
                key_name, rt = keys[0]
                if key_name == 'escape':
                    experiment_running = False
                else:
                    participant_response = response_mapping.get(key_name, 'invalid')
                break

            # 毎フレーム、加速度データを取得してログに記録
            if accel_log_fh:
                current_time = trial_clock.getTime()
                accel_data_points = serial_reader.get_all_data()
                for data_point in accel_data_points:
                    accel_log_csv.writerow([f"{current_time:.6f}", data_point[0], data_point[1], data_point[2]])

            now = trial_clock.getTime()
            dt  = now - last_t
            last_t = now

            if SCROLLING_MODE:
                virtual_h = WIN_H * VIRTUAL_HEIGHT_MULTIPLIER
                # カメラのY座標を計算（時間と共に増加していく）
                camera_y = ((now * FALL_SPEED) % virtual_h)

                # 描画用の座標を計算
                temp_red_xys = red_current_pos.copy()
                temp_red_xys[:, 1] -= camera_y

                temp_green_xys = green_current_pos.copy()
                temp_green_xys[:, 1] -= camera_y

                # 座標を仮想空間の範囲にラップ（トーラス状に折り返し）
                temp_red_xys[:, 1] = ((temp_red_xys[:, 1] + virtual_h/2) % virtual_h) - virtual_h/2
                temp_green_xys[:, 1] = ((temp_green_xys[:, 1] + virtual_h/2) % virtual_h) - virtual_h/2

                # X座標の横揺れを適用
                phase = 2 * np.pi * OSC_FREQ * now
                x_osc_offset = OSC_AMP * np.sin(phase)
                temp_red_xys[:, 0] = red_base_x + x_osc_offset
                temp_green_xys[:, 0] = green_base_x - x_osc_offset

                red_dots.xys = temp_red_xys
                green_dots.xys = temp_green_xys
            else:
                # 通常モードではドットが落下し、画面下でループする
                red_current_pos[:, 1] -= FALL_SPEED * dt
                green_current_pos[:, 1] -= FALL_SPEED * dt

                is_below_screen_red = red_current_pos[:, 1] < Y_MIN
                red_current_pos[is_below_screen_red, 1] += WIN_H

                is_below_screen_green = green_current_pos[:, 1] < Y_MIN
                green_current_pos[is_below_screen_green, 1] += WIN_H

                # X座標の横揺れを適用
                phase = 2 * np.pi * OSC_FREQ * now
                x_osc_offset = OSC_AMP * np.sin(phase)
                red_current_pos[:, 0] = red_base_x + x_osc_offset
                green_current_pos[:, 0] = green_base_x - x_osc_offset

                red_dots.xys = red_current_pos
                green_dots.xys = green_current_pos

            red_dots.draw()
            green_dots.draw()
            win.flip()

        if stereo_snd and stereo_snd.status != constants.STOPPED:
            stereo_snd.stop()

        # 試行ごとの加速度ログファイルを閉じる
        if accel_log_fh:
            accel_log_fh.close()
            save_acceleration_graph(accel_log_path, trial_idx)

        if not experiment_running:
            break

        # メインログに記録
        audio_freqs_str = " | ".join([",".join(map(str, s.freqs)) for s in SOUND_SOURCES])
        audio_pos_str = " | ".join([str(s.base_pos) for s in SOUND_SOURCES])
        log_data = [
            trial_idx, PANNING_MODE, SCROLLING_MODE, cond_type, participant_response, f"{rt:.3f}",
            WIN_W, WIN_H, N_DOTS, DOT_SIZE, FALL_SPEED, OSC_FREQ,
            OSC_AMP, audio_freqs_str, audio_pos_str, WORLD_SPACE_MIN, WORLD_SPACE_MAX,
            OSC_AMP_WORLD, LISTENER_POS_Z, DISTANCE_ATTENUATION, MIN_DISTANCE_GAIN,
            SAMPLE_RATE, MAX_ITD_S
        ]
        main_log_csv.writerow(log_data)
        main_log_fh.flush()

        # ----- ITI -----
        fixation = visual.TextStim(win, text='+', color='white', height=30)
        fixation.draw()
        win.flip()
        iti_clock = core.Clock()
        while iti_clock.getTime() < ITI:
            if event.getKeys(['escape']):
                experiment_running = False
                break
            core.wait(0.01)
        if not experiment_running: break

        trial_idx += 1

except Exception as e:
    print(f"エラーが発生しました: {e}")
finally:
    # スレッドとシリアルポートを安全に閉じる
    if serial_reader:
        serial_reader.stop()
    if stereo_snd and stereo_snd.status != constants.STOPPED:
        stereo_snd.stop()
    if 'main_log_fh' in locals() and main_log_fh and not main_log_fh.closed:
        main_log_fh.close()
        print(f"メインログを保存しました: {os.path.abspath(MAIN_LOG_PATH)}")
    if 'accel_log_fh' in locals() and accel_log_fh and not accel_log_fh.closed:
        accel_log_fh.close() # 念のため閉じる
    if 'win' in locals() and win:
        win.close()
    core.quit()
    print("実験を終了しました。")
