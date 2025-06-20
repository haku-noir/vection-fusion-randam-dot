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
#    （prefs 設定より後に行うこと）
# ------------------------------------------------------------------
from psychopy import sound, visual, core, event, constants
import numpy as np
import csv
import os
import random
from datetime import datetime


# ------------------------------------------------------------------
# 2. 実験パラメータ（自由に変更可）
# ------------------------------------------------------------------
# 音源の情報を保持するクラスを定義
class SoundSource:
    """音源の周波数と基本位置を保持するクラス"""
    def __init__(self, freq, base_pos):
        self.freq = freq
        self.base_pos = base_pos

# ★★★ 音響モードを選択 ★★★
# 'volume': 音量差パンニング
# 'itd'   : 時間差 (ITD) パンニング
# 'both'  : 音量差と時間差の両方
PANNING_MODE = 'both'

# 画面・ドット
WIN_SIZE      = (1280, 720)  # ウィンドウ解像度
N_DOTS        = 200          # 赤・緑それぞれのドット数
DOT_SIZE      = 5            # ドット直径 [pix]
FALL_SPEED    = 0            # 落下速度 [pix/s]
OSC_FREQ      = 0.24         # 横揺れ周波数 [Hz]
OSC_AMP       = 60           # ドットの横揺れ振幅 [pix]

# 音
# 仮想的な「世界空間」を最小値と最大値で定義
WORLD_SPACE_MIN = -100.0 # 世界空間の左端
WORLD_SPACE_MAX = 100.0  # 世界空間の右端
OSC_AMP_WORLD = 80.0     # 世界空間における音の振動の振幅

# SoundSourceオブジェクトのリストを世界空間の座標で定義
SOUND_SOURCES = [
    SoundSource(freq=440.00, base_pos=-25.0), # ラ(A4)
    SoundSource(freq=523.25, base_pos=0.0),   # ド(C5)
    SoundSource(freq=659.25, base_pos=25.0)    # ミ(E5)
]
SAMPLE_RATE   = 44100        # サンプリングレート [Hz]
MAX_ITD_S     = 0.0007       # ITDの最大値 (秒)。'itd'または'both'モードで使用

# 試行
TRIAL_DURATION   = 30.0      # 各試行の刺激掲示時間 [s]
ITI              = 1.0       # 刺激間インターバル [s]

# ログ
LOG_DIR = PANNING_MODE
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILENAME = f'experiment_log_{timestamp}.csv'
LOG_PATH = os.path.join(LOG_DIR, LOG_FILENAME)


# ------------------------------------------------------------------
# 3. ウィンドウと刺激の準備
# ------------------------------------------------------------------
win = visual.Window(size=WIN_SIZE, color=[-1, -1, -1], units='pix',
                    fullscr=False, allowGUI=True)

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

WIN_W, WIN_H = WIN_SIZE
Y_MAX = WIN_H / 2
Y_MIN = -WIN_H / 2

def init_positions():
    xpos = np.random.uniform(-WIN_W/2, WIN_W/2, N_DOTS)
    ypos = np.random.uniform(-WIN_H/2, WIN_H/2, N_DOTS)
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
    
    # 世界空間で音源グループの移動を計算
    base_positions = [s.base_pos for s in SOUND_SOURCES]
    min_base_pos = min(base_positions) if base_positions else 0
    max_base_pos = max(base_positions) if base_positions else 0

    osc_wave = np.sin(2 * np.pi * OSC_FREQ * t)
    
    group_shift_world = OSC_AMP_WORLD * osc_wave
    if not sync_to_red:
        group_shift_world *= -1

    predicted_left_edge = min_base_pos + group_shift_world
    predicted_right_edge = max_base_pos + group_shift_world

    # 世界空間の最小・最大値を使ってはみ出し量を計算
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
        
        # 世界空間座標をパンニング位置(-1.0から1.0)に線形変換
        world_range = WORLD_SPACE_MAX - WORLD_SPACE_MIN
        if world_range == 0: world_range = 1 # ゼロ除算を避ける
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
        
        wave_left = np.sin(2 * np.pi * source.freq * t_left) * left_gain
        wave_right = np.sin(2 * np.pi * source.freq * t_right) * right_gain

        total_left_wave += wave_left
        total_right_wave += wave_right

    stereo = np.column_stack([total_left_wave, total_right_wave])
    if SOUND_SOURCES:
        stereo /= len(SOUND_SOURCES)
    stereo *= 0.9
    
    return sound.Sound(value=stereo, sampleRate=SAMPLE_RATE, stereo=True, hamming=True)


# ------------------------------------------------------------------
# 5. ログファイルオープン
# ------------------------------------------------------------------
try:
    log_fh = open(LOG_PATH, 'w', newline='', encoding='utf-8')
    log_csv = csv.writer(log_fh)
    header = [
        'trial', 'panning_mode', 'condition', 'response', 'RT',
        'win_width', 'win_height', 'n_dots', 'dot_size', 'fall_speed',
        'dot_osc_freq', 'dot_osc_amp', 'audio_freqs', 'audio_positions_world', 
        'world_space_min', 'world_space_max', 'sound_osc_amp_world', 'sample_rate', 'max_itd_s'
    ]
    log_csv.writerow(header)
except IOError as e:
    print(f"ログファイルを開けませんでした: {LOG_PATH}")
    print(f"エラー: {e}")
    core.quit()


# ------------------------------------------------------------------
# 6. メイン実験ループ
# ------------------------------------------------------------------
experiment_running = True
trial_idx = 1
response_mapping = {'r': 'red', 'g': 'green'}
stereo_snd = None

try:
    while experiment_running:
        # ----- この試行のための設定 -----
        cond_type = random.choice(['red', 'green'])
        
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
            
            now = trial_clock.getTime()
            dt  = now - last_t
            last_t = now

            red_current_pos[:, 1]   -= FALL_SPEED * dt
            green_current_pos[:, 1] -= FALL_SPEED * dt

            phase = 2 * np.pi * OSC_FREQ * now
            x_osc_offset = OSC_AMP * np.sin(phase)
            red_current_pos[:, 0]   = red_base_x + x_osc_offset
            green_current_pos[:, 0] = green_base_x - x_osc_offset
            
            red_dots.xys, green_dots.xys = red_current_pos, green_current_pos
            red_dots.draw()
            green_dots.draw()
            win.flip()

        if stereo_snd and stereo_snd.status != constants.STOPPED:
            stereo_snd.stop()

        if not experiment_running:
            break

        # ログに世界空間の最小・最大値を追加
        audio_freqs_str = "-".join([str(s.freq) for s in SOUND_SOURCES])
        audio_pos_str = "-".join([str(s.base_pos) for s in SOUND_SOURCES])
        log_data = [
            trial_idx, PANNING_MODE, cond_type, participant_response, f"{rt:.3f}",
            WIN_SIZE[0], WIN_SIZE[1], N_DOTS, DOT_SIZE, FALL_SPEED, OSC_FREQ,
            OSC_AMP, audio_freqs_str, audio_pos_str, WORLD_SPACE_MIN, WORLD_SPACE_MAX,
            OSC_AMP_WORLD, SAMPLE_RATE, MAX_ITD_S
        ]
        log_csv.writerow(log_data)
        log_fh.flush()

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
    if stereo_snd and stereo_snd.status != constants.STOPPED:
        stereo_snd.stop()
    if 'log_fh' in locals() and log_fh and not log_fh.closed:
        log_fh.close()
        print(f"ログを保存しました: {os.path.abspath(LOG_PATH)}")
    if 'win' in locals() and win:
        win.close()
    core.quit()
    print("実験を終了しました。")
