#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audiovisual_experiment_ptb.py
---------------------------------
赤・緑ランダムドットとパンニング音声の視聴覚統合実験テンプレート
・Standalone PsychoPy 専用（Python 3）
・Audio backend に必ず PTB (Psychtoolbox) を使用

- ESCキーが押されるまで実験試行を無限に繰り返すように変更。
- 各試行で音と同期するドットの色（赤か緑か）をランダムに決定。
"""

# ------------------------------------------------------------------
# 0. PTB を最優先でロードするための prefs 設定
# ------------------------------------------------------------------
from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']      # PTB を第一候補に固定
prefs.hardware['audioLatencyMode'] = 3     # 低レイテンシ 0–4（3 がおすすめ）
# 必要なら出力デバイスを指定:
# prefs.hardware['audioDevice'] = 'Built-in Output'

# ------------------------------------------------------------------
# 1. 必要ライブラリのインポート
#    （prefs 設定より後に行うこと）
# ------------------------------------------------------------------
from psychopy import sound, visual, core, event, constants # constants を追加
import numpy as np
import csv
import os
import random # random を追加

# ------------------------------------------------------------------
# 2. 実験パラメータ（自由に変更可）
# ------------------------------------------------------------------
# 画面・ドット
WIN_SIZE      = (1280, 720)  # ウィンドウ解像度
N_DOTS        = 200          # 赤・緑それぞれのドット数
DOT_SIZE      = 5            # ドット直径 [pix]
FALL_SPEED    = 0          # 落下速度 [pix/s]
OSC_FREQ      = 1.0          # 横揺れ周波数 [Hz]
OSC_AMP       = 60           # 横揺れ振幅 [pix]

# 試行
TRIAL_DURATION   = 10.0         # 各試行の刺激呈示時間 [s]
# TRIAL_CONDS は不要になったため削除 (各試行でランダムに決定)
ITI              = 1.0         # 刺激間インターバル [s]

# 音
AUDIO_FREQ    = 440          # ベーストーン周波数 [Hz]
SAMPLE_RATE   = 44100        # サンプリングレート [Hz]

# ログ
LOG_NAME      = 'experiment_log_continuous.csv' # ログファイル名を変更

# ------------------------------------------------------------------
# 3. ウィンドウと刺激の準備
# ------------------------------------------------------------------
win = visual.Window(size=WIN_SIZE, color=[-1, -1, -1], units='pix', # 背景色を黒 (-1,-1,-1) に変更
                    fullscr=False, allowGUI=True)

# ドット刺激（ElementArrayStim）を色別に作成
def create_dot_stim(color_rgb):
    stim = visual.ElementArrayStim(
        win,
        nElements=N_DOTS,
        elementTex=None,
        elementMask='circle',
        sizes=DOT_SIZE,
        colors=color_rgb,
        xys=np.zeros((N_DOTS, 2)), # 初期座標は後で設定
        colorSpace='rgb'
    )
    return stim

RED_RGB   = [1, -1, -1]    # PsychoPy は -1〜1
GREEN_RGB = [-1, 1, -1]

red_dots   = create_dot_stim(RED_RGB)
green_dots = create_dot_stim(GREEN_RGB)

# 画面境界
WIN_W, WIN_H = WIN_SIZE
Y_MAX = WIN_H / 2
Y_MIN = -WIN_H / 2

def init_positions():
    """ドットを画面上部にランダム配置し、[x, y] 座標の配列を返す"""
    xpos = np.random.uniform(-WIN_W/2, WIN_W/2, N_DOTS)
    ypos = np.random.uniform(-WIN_W/2, WIN_W/2, N_DOTS) # 初期Y位置を画面上半分に
    return np.column_stack([xpos, ypos])

# ------------------------------------------------------------------
# 4. サウンド波形生成（NumPy ステレオ配列）
# ------------------------------------------------------------------
def build_stereo_wave(sync_to_red: bool) -> sound.Sound:
    """
    sync_to_red=True  -> 赤の横揺れと同位相
    sync_to_red=False -> 緑（=赤と逆位相）
    """
    t = np.linspace(0, TRIAL_DURATION, int(SAMPLE_RATE * TRIAL_DURATION),
                    endpoint=False)
    base = np.sin(2*np.pi*AUDIO_FREQ*t)      # 440 Hz トーン
    pan  = np.sin(2*np.pi*OSC_FREQ*t)        # ±1 のパン信号
    if not sync_to_red:                      # 緑同期なら逆位相
        pan *= -1
    
    left_gain = np.sqrt(0.5 * (1 - pan))
    right_gain = np.sqrt(0.5 * (1 + pan))
    
    left  = base * left_gain
    right = base * right_gain
    
    stereo = np.column_stack([left, right])
    return sound.Sound(value=stereo, sampleRate=SAMPLE_RATE, stereo=True, hamming=True)

# ------------------------------------------------------------------
# 5. ログファイルオープン
# ------------------------------------------------------------------
log_fh = open(LOG_NAME, 'w', newline='', encoding='utf-8') # 'w'モードで起動ごとに新規作成
log_csv = csv.writer(log_fh)
log_csv.writerow(['trial', 'condition', 'response', 'RT'])

# ------------------------------------------------------------------
# 6. メイン実験ループ
# ------------------------------------------------------------------
experiment_running = True
trial_idx = 1
response_mapping = {'r': 'red', 'g': 'green'}
stereo_snd = None # try-finally で参照できるようにスコープ外で宣言

try:
    while experiment_running:
        # ----- この試行のための設定 -----
        cond_type = random.choice(['red', 'green']) # 同期対象をランダムに決定
        
        # ----- ドット位置と時間の初期化 -----
        red_current_pos   = init_positions()
        green_current_pos = init_positions()
        red_base_x   = red_current_pos[:, 0].copy()
        green_base_x = green_current_pos[:, 0].copy()
        last_t = 0.0

        # ----- サウンド準備 -----
        sync_red = (cond_type == 'red')
        if stereo_snd: # 前の試行のサウンドオブジェクトが残っていれば停止
             if stereo_snd.status != constants.STOPPED:
                  stereo_snd.stop()
        stereo_snd = build_stereo_wave(sync_red)

        # ----- 刺激呈示 -----
        trial_clock = core.Clock()
        stereo_snd.play()
        trial_clock.reset()

        # 刺激呈示ループ
        while trial_clock.getTime() < TRIAL_DURATION:
            # ESCキーチェック
            if event.getKeys(['escape']):
                experiment_running = False
                if stereo_snd.status != constants.STOPPED:
                    stereo_snd.stop()
                break # 刺激呈示ループを抜ける
            
            now = trial_clock.getTime()
            dt  = now - last_t
            last_t = now

            # Y 座標を落下
            red_current_pos[:, 1]   -= FALL_SPEED * dt
            green_current_pos[:, 1] -= FALL_SPEED * dt

            # 横揺れオフセット計算とX座標更新
            phase = 2 * np.pi * OSC_FREQ * now
            x_oscillation_offset = OSC_AMP * np.sin(phase)
            red_current_pos[:, 0]   = red_base_x + x_oscillation_offset
            green_current_pos[:, 0] = green_base_x - x_oscillation_offset

            # 画面外ドットの再配置処理
            dot_arrays = [
                (red_current_pos, red_base_x),
                (green_current_pos, green_base_x)
            ]
            for current_pos_arr, base_x_arr_for_color in dot_arrays:
                below_screen_idx = current_pos_arr[:, 1] < Y_MIN
                num_dots_to_reset = below_screen_idx.sum()
                if num_dots_to_reset > 0:
                    current_pos_arr[below_screen_idx, 1] = Y_MAX + np.random.uniform(0, 50, num_dots_to_reset)
                    new_x_values = np.random.uniform(-WIN_W/2, WIN_W/2, num_dots_to_reset)
                    current_pos_arr[below_screen_idx, 0] = new_x_values
                    base_x_arr_for_color[below_screen_idx] = new_x_values
            
            # 座標更新 & 描画
            red_dots.xys   = red_current_pos
            green_dots.xys = green_current_pos
            red_dots.draw()
            green_dots.draw()
            win.flip()
        # 刺激呈示ループ終了

        if not experiment_running: # ESCで刺激呈示が中断された場合
            break # メインの while experiment_running ループを抜ける

        # 刺激呈示時間が正常に終了した場合、音を止める
        if stereo_snd.status != constants.STOPPED:
             stereo_snd.stop()

        # ----- 応答取得 -----
        prompt_text = ("どちらのドット群と音が同期してパンニングしているように感じましたか？\n"
                       "（どちらのドット群が、音と一体となって左右に揺れているように見えましたか？）\n\n"
                       "[R] 赤いドット / [G] 緑のドット\n\n"
                       "ESCキーで実験中断")
        prompt = visual.TextStim(win, text=prompt_text, color='white', height=24, wrapWidth=WIN_W*0.8)
        prompt.draw()
        win.flip()
        
        resp_clock = core.Clock()
        response_keys = event.waitKeys(keyList=['r', 'g', 'escape'], timeStamped=resp_clock)
        
        key_name, rt = 'timeout', -1.0 
        if response_keys:
            key_name, rt = response_keys[0]

        if key_name == 'escape':
            experiment_running = False
            break # メインの while experiment_running ループを抜ける
        
        participant_response = response_mapping.get(key_name, 'invalid')

        # ----- ログ保存 -----
        log_csv.writerow([trial_idx, cond_type, participant_response, f"{rt:.3f}"])
        log_fh.flush()

        # ----- 刺激間インターバル -----
        fixation = visual.TextStim(win, text='+', color='white', height=30)
        fixation.draw()
        win.flip()
        
        iti_clock = core.Clock()
        while iti_clock.getTime() < ITI:
            if event.getKeys(['escape']): # ITI中にESC
                experiment_running = False
                break
            core.wait(0.01) # CPU負荷を抑えつつキー入力をポーリング

        if not experiment_running: # ITI中にESCが押された場合
            break # メインの while experiment_running ループを抜ける

        trial_idx += 1 # 次の試行番号へ
    # メインの while experiment_running ループ終了

except KeyboardInterrupt: # Ctrl+C などによる予期せぬ中断
    print("Experiment interrupted by user (e.g., Ctrl+C).")
    experiment_running = False # finallyブロックでのメッセージ制御のため
except Exception as e: # その他の予期せぬエラー
    print(f"An unexpected error occurred: {e}")
    experiment_running = False # finallyブロックでのメッセージ制御のため
finally:
    # リソース解放
    print_msg = "Experiment finished."
    if not experiment_running: # experiment_runningがFalseならESCかエラーで終了した
        # KeyboardInterrupt のメッセージが先に出ている場合は、これは補足的なもの
        if not isinstance(e if 'e' in locals() else None, KeyboardInterrupt): # Ctrl+C以外での中断
             print("Experiment session terminated by ESC or an error.")

    if stereo_snd and stereo_snd.status != constants.STOPPED:
        stereo_snd.stop()
    
    if 'log_fh' in locals() and log_fh and not log_fh.closed:
        log_fh.close()
        print(f"Log saved to {os.path.abspath(LOG_NAME)}")
    
    if 'win' in locals() and win:
        win.close()
    
    core.quit()
    print(print_msg) # 最終的な終了メッセージ

