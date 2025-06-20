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
OSC_AMP       = 60           # 横揺れ振幅 [pix]

# 試行
TRIAL_DURATION   = 10.0      # 各試行の刺激呈示時間 [s]
ITI              = 1.0       # 刺激間インターバル [s]

# 音
AUDIO_FREQ    = 440          # ベーストーン周波数 [Hz]
SAMPLE_RATE   = 44100        # サンプリングレート [Hz]
MAX_ITD_S     = 0.0007       # ITDの最大値 (秒)。'itd'または'both'モードで使用

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
    ypos = np.random.uniform(-WIN_H/2, WIN_H/2, N_DOTS) # Y座標も全域に
    return np.column_stack([xpos, ypos])

# ------------------------------------------------------------------
# 4. サウンド波形生成（モード切替対応）
# ------------------------------------------------------------------
def build_stereo_wave(sync_to_red: bool, mode: str) -> sound.Sound:
    """
    指定されたモードに応じてステレオ音響を生成する。
    mode: 'volume', 'itd', 'both'
    """
    t = np.linspace(0, TRIAL_DURATION, int(SAMPLE_RATE * TRIAL_DURATION), endpoint=False)
    
    pan = np.sin(2 * np.pi * OSC_FREQ * t)
    if not sync_to_red:
        pan *= -1
        
    # モードに応じてゲインと遅延を初期化
    left_gain, right_gain = 1.0, 1.0
    delay_L_s, delay_R_s = 0.0, 0.0

    # 音量差を適用 (volume or both)
    if mode in ['volume', 'both']:
        left_gain = np.sqrt(0.5 * (1 - pan))
        right_gain = np.sqrt(0.5 * (1 + pan))
        
    # 時間差を適用 (itd or both)
    if mode in ['itd', 'both']:
        delay_L_s = np.maximum(0, pan) * MAX_ITD_S
        delay_R_s = np.maximum(0, -pan) * MAX_ITD_S

    # 遅延を適用した時間配列を作成
    t_left = t - delay_L_s
    t_right = t - delay_R_s

    # 各チャンネルの波形を生成（ゲインと時間差を両方適用）
    base_wave = np.sin(2 * np.pi * AUDIO_FREQ * t)
    left_wave = np.sin(2 * np.pi * AUDIO_FREQ * t_left) * left_gain
    right_wave = np.sin(2 * np.pi * AUDIO_FREQ * t_right) * right_gain

    # ステレオに結合し、クリッピング防止のために全体の音量を調整
    stereo = np.column_stack([left_wave, right_wave]) * 0.9
    
    return sound.Sound(value=stereo, sampleRate=SAMPLE_RATE, stereo=True, hamming=True)

# ------------------------------------------------------------------
# 5. ログファイルオープン
# ------------------------------------------------------------------
try:
    # ★変更点: 完全なファイルパスでファイルを開く
    log_fh = open(LOG_PATH, 'w', newline='', encoding='utf-8')
    log_csv = csv.writer(log_fh)
    log_csv.writerow(['trial', 'condition', 'panning_mode', 'response', 'RT'])
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
        stereo_snd = build_stereo_wave(sync_red, PANNING_MODE)

        # ----- 刺激呈示 -----
        trial_clock = core.Clock()
        stereo_snd.play()
        trial_clock.reset()

        while trial_clock.getTime() < TRIAL_DURATION:
            if event.getKeys(['escape']):
                experiment_running = False
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
            
            # ドットは落下しない設定 (FALL_SPEED=0) なので、画面外処理は不要
            
            red_dots.xys, green_dots.xys = red_current_pos, green_current_pos
            red_dots.draw()
            green_dots.draw()
            win.flip()

        if not experiment_running: break
        if stereo_snd and stereo_snd.status != constants.STOPPED: stereo_snd.stop()

        # ----- 応答取得 -----
        prompt = visual.TextStim(win, text="どちらのドット群と音が同期していましたか？\n[R] 赤 / [G] 緑\n\nESCキーで実験終了",
                                 color='white', height=24, wrapWidth=WIN_W*0.8)
        prompt.draw()
        win.flip()
        
        keys = event.waitKeys(keyList=['r', 'g', 'escape'], timeStamped=core.Clock())
        
        if not keys or keys[0][0] == 'escape':
            experiment_running = False
            break
        
        key_name, rt = keys[0]
        participant_response = response_mapping.get(key_name, 'invalid')

        log_csv.writerow([trial_idx, PANNING_MODE, cond_type, participant_response, f"{rt:.3f}"])
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
        # ★変更点: 正しいファイルパスを表示
        print(f"ログを保存しました: {os.path.abspath(LOG_PATH)}")
    if 'win' in locals() and win:
        win.close()
    core.quit()
    print("実験を終了しました。")
