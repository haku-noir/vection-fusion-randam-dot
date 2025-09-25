#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
加速度データの解析プログラム
CSVファイルから加速度データを読み込み、角度変化を計算し、
視覚刺激との時間窓相関解析を行ってプロットする
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import re
from scipy.signal import butter, lfilter

# --- 設定 ---
# グラフを180秒ごとに分割して出力するかどうか
SPLIT_PLOTS_BY_TIME = True
SPLIT_DURATION = 200  # 分割する時間（秒）
START_TIME = 0 # 解析開始時間(s)
FILTER_CUTOFF_HZ = 2 # ローパスフィルタのカットオフ周波数 (Hz)

# --- 相関解析の設定 ---
CORR_WINDOW_SIZE_SEC = 10  # 相関解析の窓のサイズ（秒）
CORR_STEP_SIZE_SEC = 1     # 相関解析の窓をずらすステップサイズ（秒）

# --- 各ファイル処理の設定 ---
SKIP_PROCESS_FILE = True   # 各ファイルの処理をスキップするかどうか

# --- ファイル出力の設定 ---
SKIP_EXISTING_FILE = True   # 既存のファイルがある場合はスキップするかどうか


def apply_lowpass_filter(df, cutoff=5, fs=120, order=4):
    """
    データフレームの加速度データにローパスフィルタを適用

    Args:
        df (pd.DataFrame): 加速度データを含むデータフレーム
        cutoff (float): カットオフ周波数 (Hz)
        fs (float): サンプリング周波数 (Hz)
        order (int): フィルタの次数

    Returns:
        pd.DataFrame: フィルタ適用後のデータフレーム
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    df_filtered = df.copy()
    df_filtered['accel_x'] = lfilter(b, a, df['accel_x'])
    df_filtered['accel_y'] = lfilter(b, a, df['accel_y'])
    df_filtered['accel_z'] = lfilter(b, a, df['accel_z'])

    return df_filtered

def calculate_acceleration_magnitude(csv_file_path, use_filter=False):
    """
    CSVファイルから加速度データを読み込み、加速度ベクトルの大きさを計算

    Args:
        csv_file_path (str): CSVファイルのパス
        use_filter (bool): ローパスフィルタを適用するかどうか

    Returns:
        pd.DataFrame: 時間と加速度ベクトルの大きさを含むデータフレーム
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"データを読み込みました: {len(df)} 行")

        required_columns = ['psychopy_time', 'accel_x', 'accel_y', 'accel_z']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"必要な列 '{col}' が見つかりません")

        if use_filter:
            print(f"ローパスフィルタ（{FILTER_CUTOFF_HZ}Hz）を適用します...")
            df = apply_lowpass_filter(df, cutoff=FILTER_CUTOFF_HZ)

        df['accel_magnitude'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)

        output_columns = ['psychopy_time', 'accel_x', 'accel_y', 'accel_z', 'accel_magnitude']
        visual_columns = ['red_dot_mean_x', 'red_dot_mean_y', 'green_dot_mean_x', 'green_dot_mean_y']
        for col in visual_columns:
            if col in df.columns:
                output_columns.append(col)

        return df[output_columns]

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def calculate_gravity_and_angles(df):
    """
    重力加速度ベクトルを推定し、各時刻での角度を計算

    Args:
        df (pd.DataFrame): 加速度データを含むデータフレーム

    Returns:
        pd.DataFrame: 角度情報を追加したデータフレーム
    """
    gravity_x = df['accel_x'].mean()
    gravity_y = df['accel_y'].mean()
    gravity_z = df['accel_z'].mean()
    gravity_vector = np.array([gravity_x, gravity_y, gravity_z])
    gravity_magnitude = np.linalg.norm(gravity_vector)

    print(f"\n=== 推定平均加速度ベクトル ===")
    print(f"平均X: {gravity_x:.3f} m/s²")
    print(f"平均Y: {gravity_y:.3f} m/s²")
    print(f"平均Z: {gravity_z:.3f} m/s²")
    print(f"平均の大きさ: {gravity_magnitude:.3f} m/s²")

    angles, angle_changes = [], []
    initial_vector = np.array([df.iloc[0]['accel_x'], df.iloc[0]['accel_y'], df.iloc[0]['accel_z']]) if len(df) > 0 else gravity_vector
    initial_magnitude = np.linalg.norm(initial_vector)

    for idx, row in df.iterrows():
        current_vector = np.array([row['accel_x'], row['accel_y'], row['accel_z']])
        current_magnitude = np.linalg.norm(current_vector)

        if current_magnitude > 0 and gravity_magnitude > 0:
            cos_angle_gravity = np.clip(np.dot(current_vector, gravity_vector) / (current_magnitude * gravity_magnitude), -1.0, 1.0)
            angles.append(np.degrees(np.arccos(cos_angle_gravity)))
        else:
            angles.append(0.0)

        if current_magnitude > 0 and initial_magnitude > 0:
            cos_angle_initial = np.clip(np.dot(current_vector, initial_vector) / (current_magnitude * initial_magnitude), -1.0, 1.0)
            angle_deg_initial = np.degrees(np.arccos(cos_angle_initial))
            cross_product = np.cross(initial_vector, current_vector)
            sign = np.sign(cross_product[2]) if len(cross_product) == 3 and abs(cross_product[2]) > 1e-10 else 1
            angle_changes.append(sign * angle_deg_initial)
        else:
            angle_changes.append(0.0)

    df_with_angles = df.copy()
    df_with_angles['angle_to_gravity'] = angles
    df_with_angles['angle_change'] = angle_changes

    if 'red_dot_mean_x' in df.columns and 'green_dot_mean_x' in df.columns:
        red_initial = df['red_dot_mean_x'].iloc[0] if len(df) > 0 else 0
        green_initial = df['green_dot_mean_x'].iloc[0] if len(df) > 0 else 0
        df_with_angles['red_dot_x_change'] = df['red_dot_mean_x'] - red_initial
        df_with_angles['green_dot_x_change'] = df['green_dot_mean_x'] - green_initial

    return df_with_angles, gravity_vector

def calculate_windowed_correlation(df):
    """
    時間窓相関解析を実行する

    Args:
        df (pd.DataFrame): 角度変化と視覚刺激データを含むデータフレーム

    Returns:
        pd.DataFrame: 時間ごとの相関係数を含むデータフレーム
    """
    print("時間窓相関解析を実行します...")
    if 'angle_change' not in df.columns or 'red_dot_x_change' not in df.columns:
        print("相関解析に必要な列が見つかりません。スキップします。")
        return pd.DataFrame()

    time_diffs = np.diff(df['psychopy_time'])
    sampling_rate = 1.0 / np.median(time_diffs)

    window_samples = int(CORR_WINDOW_SIZE_SEC * sampling_rate)
    step_samples = int(CORR_STEP_SIZE_SEC * sampling_rate)

    results = []
    start_index = 0
    while start_index + window_samples <= len(df):
        end_index = start_index + window_samples

        window_time = df['psychopy_time'].iloc[start_index:end_index].mean()

        # 窓内のデータを抽出
        angle_window = df['angle_change'].iloc[start_index:end_index]
        red_window = df['red_dot_x_change'].iloc[start_index:end_index]
        green_window = df['green_dot_x_change'].iloc[start_index:end_index]

        # トレンド除去
        angle_detrended = angle_window - angle_window.mean()
        red_detrended = red_window - red_window.mean()
        green_detrended = green_window - green_window.mean()

        # 相関係数を計算
        corr_red = angle_detrended.corr(red_detrended)
        corr_green = angle_detrended.corr(green_detrended)

        results.append({'time': window_time, 'corr_red': corr_red, 'corr_green': corr_green})
        start_index += step_samples

    return pd.DataFrame(results)


def plot_angle_analysis(df, gravity_vector, df_corr=None, output_file=None, split_plots=False, segment_duration=60):
    """
    角度解析と相関解析のグラフをプロット

    Args:
        df (pd.DataFrame): 角度データを含むデータフレーム
        gravity_vector (np.array): 重力ベクトル
        df_corr (pd.DataFrame, optional): 相関データを含むデータフレーム. Defaults to None.
        output_file (str): 保存するファイル名（Noneの場合は表示のみ）
        split_plots (bool): Trueの場合、segment_durationごとにグラフを分割して保存
        segment_duration (int): 分割する場合の時間（秒）
    """
    if not split_plots:
        _plot_single_chart(df, gravity_vector, df_corr, output_file, show_plot=True)
    else:
        if df.empty:
            print("データが空のため、分割プロットは作成されません。")
            return

        t_min, t_max = df['psychopy_time'].min(), df['psychopy_time'].max()
        print(f"データを{segment_duration}秒ごとに分割してプロットを作成します... (範囲: {t_min:.2f}s - {t_max:.2f}s)")
        base_name, ext = os.path.splitext(output_file)

        for t_start in np.arange(t_min, t_max, segment_duration):
            t_end = t_start + segment_duration
            df_segment = df[(df['psychopy_time'] >= t_start) & (df['psychopy_time'] < t_end)]

            if df_segment.empty: continue

            df_corr_segment = None
            if df_corr is not None and not df_corr.empty:
                df_corr_segment = df_corr[(df_corr['time'] >= t_start) & (df_corr['time'] < t_end)]

            segment_output_file = f"{base_name}_{int(t_start)}-{int(t_end)}s{ext}"
            # 既存ファイルのチェック
            if SKIP_EXISTING_FILE and os.path.exists(segment_output_file):
                print(f"既存ファイルをスキップしました: {segment_output_file}")
                return
            print(f"  - {int(t_start)}s から {int(t_end)}s の区間をプロット中...")
            _plot_single_chart(df_segment, gravity_vector, df_corr_segment, segment_output_file, show_plot=False)
        print("分割プロットの作成が完了しました。")


def _plot_single_chart(df, gravity_vector, df_corr, output_file, show_plot=True):
    """
    単一のグラフ（4つのサブプロット）を作成・保存・表示する内部関数
    """
    if df.empty:
        print("プロットするデータがありません。")
        return

    plt.figure(figsize=(15, 16)) # グラフの高さを変更
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

    # --- サブプロット1: 加速度の各成分 ---
    plt.subplot(4, 1, 1)
    plt.plot(df['psychopy_time'], df['accel_x'], label='X軸加速度', alpha=0.7)
    plt.plot(df['psychopy_time'], df['accel_y'], label='Y軸加速度', alpha=0.7)
    plt.plot(df['psychopy_time'], df['accel_z'], label='Z軸加速度', alpha=0.7)
    plt.axhline(y=gravity_vector[0], color='red', linestyle='--', alpha=0.8, label=f'平均X: {gravity_vector[0]:.2f}')
    plt.axhline(y=gravity_vector[1], color='green', linestyle='--', alpha=0.8, label=f'平均Y: {gravity_vector[1]:.2f}')
    plt.axhline(y=gravity_vector[2], color='blue', linestyle='--', alpha=0.8, label=f'平均Z: {gravity_vector[2]:.2f}')
    plt.title(f'各軸の加速度変化と推定平均加速度 ({df["psychopy_time"].min():.1f}s - {df["psychopy_time"].max():.1f}s)')
    plt.ylabel('加速度 (m/s²)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- サブプロット2: 加速度ベクトルの大きさ ---
    plt.subplot(4, 1, 2)
    plt.plot(df['psychopy_time'], df['accel_magnitude'], color='purple', linewidth=1.5, label='加速度ベクトルの大きさ')
    gravity_magnitude = np.linalg.norm(gravity_vector)
    plt.axhline(y=gravity_magnitude, color='red', linestyle='--', alpha=0.8, label=f'平均の大きさ: {gravity_magnitude:.2f}')
    plt.ylabel('加速度の大きさ (m/s²)')
    plt.ylim(9.5, 10.2)
    plt.title('加速度ベクトルの大きさと平均の大きさ')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- サブプロット3: 角度変化 + 視覚刺激 ---
    plt.subplot(4, 1, 3)
    ax1 = plt.gca()
    line1 = ax1.plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=1.5, label='初期位置からの角度変化')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_ylabel('角度変化 (度)', color='orange')
    ax1.set_ylim(-10, 10)
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.grid(True, alpha=0.3)

    if 'red_dot_x_change' in df.columns and 'green_dot_x_change' in df.columns:
        ax2 = ax1.twinx()
        line2 = ax2.plot(df['psychopy_time'], df['red_dot_x_change'], color='red', alpha=0.7, linewidth=1.0, label='赤点X座標変化')
        line3 = ax2.plot(df['psychopy_time'], df['green_dot_x_change'], color='green', alpha=0.7, linewidth=1.0, label='緑点X座標変化')
        ax2.set_ylabel('X座標変化 (pixel)')
        ax2.set_ylim(-500, 500)
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
    else:
        ax1.legend(loc='upper left')
    plt.title('角度変化と視覚刺激X座標変化')

    # --- サブプロット4: 時間窓相関解析 ---
    plt.subplot(4, 1, 4)
    if df_corr is not None and not df_corr.empty:
        plt.plot(df_corr['time'], df_corr['corr_red'], label='角度変化 vs 赤ドット', color='red', marker='o', markersize=3, linestyle='-')
        plt.plot(df_corr['time'], df_corr['corr_green'], label='角度変化 vs 緑ドット', color='green', marker='x', markersize=3, linestyle='--')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.ylim(-1.0, 1.0)
        plt.ylabel('相互相関係数')
        plt.legend()
        plt.title(f'時間窓相関解析 (窓: {CORR_WINDOW_SIZE_SEC}秒, ステップ: {CORR_STEP_SIZE_SEC}秒)')
    else:
        plt.text(0.5, 0.5, '相関解析データなし', ha='center', va='center')
        plt.title('時間窓相関解析')

    plt.xlabel('時間 (秒)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # --- 統計情報の表示 ---
    print(f"\n=== 統計情報 ({df['psychopy_time'].min():.1f}s - {df['psychopy_time'].max():.1f}s) ===")
    print(f"加速度ベクトルの平均: {df['accel_magnitude'].mean():.3f} m/s² (標準偏差: {df['accel_magnitude'].std():.3f})")
    print(f"角度変化の平均: {df['angle_change'].mean():.3f} 度 (標準偏差: {df['angle_change'].std():.3f})")

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {output_file}")
    if show_plot:
        plt.show()
    plt.close()

def save_angle_data_to_csv(df, output_file):
    """
    角度データをCSVファイルに保存

    Args:
        df (pd.DataFrame): 角度データを含むデータフレーム
        output_file (str): 出力CSVファイルのパス
    """
    # 既存ファイルのチェック
    if SKIP_EXISTING_FILE and os.path.exists(output_file):
        print(f"既存ファイルをスキップしました: {output_file}")
        return

    # 必要な列のみを選択
    columns_to_save = [
        'psychopy_time', 'accel_x', 'accel_y', 'accel_z', 'angle_change'
    ]

    # 存在する列のみを選択
    available_columns = [col for col in columns_to_save if col in df.columns]
    df_to_save = df[available_columns].copy()

    # CSVファイルに保存
    df_to_save.to_csv(output_file, index=False)
    print(f"角度データを保存しました: {output_file}")

def save_correlation_to_csv(df_corr, output_file):
    """
    相関データをCSVファイルに保存

    Args:
        df_corr (pd.DataFrame): 相関データを含むデータフレーム
        output_file (str): 出力CSVファイルのパス
    """
    # 既存ファイルのチェック
    if SKIP_EXISTING_FILE and os.path.exists(output_file):
        print(f"既存ファイルをスキップしました: {output_file}")
        return

    if df_corr is not None and not df_corr.empty:
        df_corr.to_csv(output_file, index=False)
        print(f"相関データを保存しました: {output_file}")
    else:
        print(f"相関データが空のため保存をスキップしました: {output_file}")

def load_correlation_from_csv(csv_file):
    """
    相関データをCSVファイルから読み込む

    Args:
        csv_file (str): 相関データCSVファイルのパス

    Returns:
        pd.DataFrame: 相関データを含むデータフレーム、読み込みに失敗した場合はNone
    """
    try:
        if os.path.exists(csv_file):
            df_corr = pd.read_csv(csv_file)
            if not df_corr.empty and 'time' in df_corr.columns:
                return df_corr
    except Exception as e:
        print(f"相関データの読み込みに失敗しました ({csv_file}): {e}")
    return None

def get_or_calculate_correlation(csv_file, force_calculate=False):
    """
    相関データを取得または計算する（スキップ設定を考慮）

    Args:
        csv_file (str): 元のCSVファイルのパス
        force_calculate (bool): 強制的に再計算するかどうか

    Returns:
        tuple: (df_corr_raw, df_corr_filtered) 相関データのタプル
    """
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    # 相関CSVファイルのパス
    corr_raw_csv = os.path.join(os.path.dirname(csv_file), f"{base_name}_correlation_raw_window{CORR_WINDOW_SIZE_SEC}s_step{CORR_STEP_SIZE_SEC}s.csv")
    corr_filtered_csv = os.path.join(os.path.dirname(csv_file), f"{base_name}_correlation_filtered_{FILTER_CUTOFF_HZ}Hz_window{CORR_WINDOW_SIZE_SEC}s_step{CORR_STEP_SIZE_SEC}s.csv")

    df_corr_raw = None
    df_corr_filtered = None

    # スキップ設定が有効で、既存CSVがある場合は読み込み
    if SKIP_EXISTING_FILE and not force_calculate:
        df_corr_raw = load_correlation_from_csv(corr_raw_csv)
        df_corr_filtered = load_correlation_from_csv(corr_filtered_csv)

        if df_corr_raw is not None and df_corr_filtered is not None:
            print(f"既存の相関データを読み込みました: {csv_file}")
            return df_corr_raw, df_corr_filtered

    # CSVがない、または強制計算の場合は完全処理を実行
    print(f"相関データを新規計算します（完全処理実行）: {csv_file}")

    try:
        # process_single_fileを実行してCSVファイルを生成
        process_single_file(csv_file)

        # 処理後にCSVファイルを読み込み
        df_corr_raw = load_correlation_from_csv(corr_raw_csv)
        df_corr_filtered = load_correlation_from_csv(corr_filtered_csv)

        if df_corr_raw is not None and df_corr_filtered is not None:
            print(f"処理完了後に相関データを読み込みました: {csv_file}")
            return df_corr_raw, df_corr_filtered
        else:
            print(f"警告: 処理後にCSVファイルが見つかりませんでした: {csv_file}")

    except Exception as e:
        print(f"エラー: 相関データ処理に失敗しました ({csv_file}): {e}")

    # フォールバック: 直接計算
    print(f"フォールバック: 直接相関計算を実行します: {csv_file}")

    # Raw data processing
    df_raw = calculate_acceleration_magnitude(csv_file, use_filter=False)
    if df_raw is not None:
        df_raw_with_angles, _ = calculate_gravity_and_angles(df_raw)
        df_corr_raw = calculate_windowed_correlation(df_raw_with_angles)

    # Filtered data processing
    df_filtered = calculate_acceleration_magnitude(csv_file, use_filter=True)
    if df_filtered is not None:
        df_filtered_with_angles, _ = calculate_gravity_and_angles(df_filtered)
        df_corr_filtered = calculate_windowed_correlation(df_filtered_with_angles)

    return df_corr_raw, df_corr_filtered

def plot_correlation_overview(target_files, output_dir):
    """
    複数ファイルの相関係数グラフを一覧表示

    Args:
        target_files (list): 対象CSVファイルのパスリスト
        output_dir (str): 出力ディレクトリ
    """
    if not target_files:
        return

    print(f"\n相関係数の一覧グラフを作成中...")

    # グリッドサイズを計算
    n_files = len(target_files)
    n_cols = min(3, n_files)  # 最大3列
    n_rows = (n_files + n_cols - 1) // n_cols

    # 各データタイプ（raw, filtered）について個別のグラフを作成
    for data_type in ['raw', 'filtered']:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle(f'相関係数一覧 ({data_type.title()})', fontsize=16, y=0.98)

        # axesを1次元配列に変換（単一プロットの場合も対応）
        if n_files == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()

        valid_plots = 0

        for i, csv_file in enumerate(target_files):
            try:
                # 相関データを取得
                df_corr_raw, df_corr_filtered = get_or_calculate_correlation(csv_file)
                df_corr = df_corr_raw if data_type == 'raw' else df_corr_filtered

                if df_corr is None or df_corr.empty:
                    axes[i].text(0.5, 0.5, 'データなし', ha='center', va='center')
                    axes[i].set_title(f'データなし', fontsize=10)
                else:
                    # 相関グラフをプロット
                    if 'corr_red' in df_corr.columns:
                        axes[i].plot(df_corr['time'], df_corr['corr_red'], 
                                   label='vs 赤ドット', color='red', linewidth=1.5)
                    if 'corr_green' in df_corr.columns:
                        axes[i].plot(df_corr['time'], df_corr['corr_green'], 
                                   label='vs 緑ドット', color='green', linewidth=1.5, linestyle='--')

                    axes[i].axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.5)
                    axes[i].set_ylim(-1.0, 1.0)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].legend(fontsize=8)

                    # タイトルの設定（フォルダ名とファイル名）
                    rel_path = os.path.relpath(csv_file, output_dir)
                    folder_name = os.path.dirname(rel_path) if os.path.dirname(rel_path) else '.'
                    file_name = os.path.splitext(os.path.basename(csv_file))[0]
                    title = f"{folder_name}/{file_name}"
                    axes[i].set_title(title, fontsize=9, wrap=True)

                    valid_plots += 1

                axes[i].set_xlabel('時間 (秒)', fontsize=8)
                axes[i].set_ylabel('相関係数', fontsize=8)

            except Exception as e:
                print(f"相関グラフ作成エラー ({csv_file}): {e}")
                axes[i].text(0.5, 0.5, f'エラー\n{str(e)[:50]}', ha='center', va='center', fontsize=8)
                axes[i].set_title('エラー', fontsize=10)

        # 余ったサブプロットを非表示
        for i in range(n_files, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # 保存
        overview_file = os.path.join(output_dir, f"correlation_overview_{data_type}_window{CORR_WINDOW_SIZE_SEC}s_step{CORR_STEP_SIZE_SEC}s.png")
        plt.savefig(overview_file, dpi=300, bbox_inches='tight')
        print(f"相関係数一覧グラフを保存しました: {overview_file}")

        plt.close()

def find_target_csv_files(input_path):
    """
    指定されたパスからYYYYMMDD_HHMMSS_accel_log_serial_trial_1.csvファイルを再帰的に検索

    Args:
        input_path (str): ファイルまたはフォルダのパス

    Returns:
        list: 対象CSVファイルのパスリスト
    """
    target_files = []

    if os.path.isfile(input_path):
        # 単一ファイルが指定された場合
        if input_path.endswith('.csv'):
            target_files.append(input_path)
    elif os.path.isdir(input_path):
        # フォルダが指定された場合、再帰的に検索
        pattern = re.compile(r'\d{8}_\d{6}_accel_log_serial_trial_1\.csv$')

        for root, dirs, files in os.walk(input_path):
            for file in files:
                if pattern.match(file):
                    target_files.append(os.path.join(root, file))

    return sorted(target_files)

def process_single_file(csv_file):
    """
    単一CSVファイルの処理

    Args:
        csv_file (str): 処理するCSVファイルのパス
    """
    print(f"\n{'='*60}")
    print(f"解析対象ファイル: {csv_file}")
    print(f"{'='*60}")

    # --- フィルタなしで解析 ---
    print("\n--- フィルタなし (生データ) で解析 ---")
    df_raw = calculate_acceleration_magnitude(csv_file, use_filter=False)
    if df_raw is None: 
        print(f"エラー: {csv_file} の処理をスキップします")
        return

    df_raw_with_angles, gravity_vector_raw = calculate_gravity_and_angles(df_raw)
    df_corr_raw = calculate_windowed_correlation(df_raw_with_angles) # 相関を計算

    base_name_raw = os.path.splitext(os.path.basename(csv_file))[0]
    output_raw = os.path.join(os.path.dirname(csv_file), f"{base_name_raw}_angle_analysis_raw_window{CORR_WINDOW_SIZE_SEC}s_step{CORR_STEP_SIZE_SEC}s.png")

    # CSV出力ファイル名を生成
    output_angles_raw_csv = os.path.join(os.path.dirname(csv_file), f"{base_name_raw}_angles_raw.csv")
    output_corr_raw_csv = os.path.join(os.path.dirname(csv_file), f"{base_name_raw}_correlation_raw_window{CORR_WINDOW_SIZE_SEC}s_step{CORR_STEP_SIZE_SEC}s.csv")

    # CSVファイルに保存
    save_angle_data_to_csv(df_raw_with_angles, output_angles_raw_csv)
    save_correlation_to_csv(df_corr_raw, output_corr_raw_csv)

    print("フィルタなしのグラフを作成します...")
    plot_angle_analysis(
        df_raw_with_angles[df_raw_with_angles["psychopy_time"] >= START_TIME],
        gravity_vector_raw,
        df_corr=df_corr_raw, # プロット関数に相関データを渡す
        output_file=output_raw,
        split_plots=SPLIT_PLOTS_BY_TIME,
        segment_duration=SPLIT_DURATION
    )

    # --- フィルタありで解析 ---
    print("\n--- ローパスフィルタありで解析 ---")
    df_filtered = calculate_acceleration_magnitude(csv_file, use_filter=True)
    if df_filtered is None: 
        print(f"エラー: {csv_file} のフィルタ処理をスキップします")
        return

    df_filtered_with_angles, gravity_vector_filtered = calculate_gravity_and_angles(df_filtered)
    df_corr_filtered = calculate_windowed_correlation(df_filtered_with_angles) # 相関を計算

    base_name_filtered = os.path.splitext(os.path.basename(csv_file))[0]
    output_filtered = os.path.join(os.path.dirname(csv_file), f"{base_name_filtered}_angle_analysis_filtered_{FILTER_CUTOFF_HZ}Hz_window{CORR_WINDOW_SIZE_SEC}s.png")

    # CSV出力ファイル名を生成
    output_angles_filtered_csv = os.path.join(os.path.dirname(csv_file), f"{base_name_filtered}_angles_filtered_{FILTER_CUTOFF_HZ}Hz.csv")
    output_corr_filtered_csv = os.path.join(os.path.dirname(csv_file), f"{base_name_filtered}_correlation_filtered_{FILTER_CUTOFF_HZ}Hz_window{CORR_WINDOW_SIZE_SEC}s_step{CORR_STEP_SIZE_SEC}s.csv")

    # CSVファイルに保存
    save_angle_data_to_csv(df_filtered_with_angles, output_angles_filtered_csv)
    save_correlation_to_csv(df_corr_filtered, output_corr_filtered_csv)

    print("フィルタありのグラフを作成します...")
    plot_angle_analysis(
        df_filtered_with_angles[df_filtered_with_angles["psychopy_time"] >= START_TIME],
        gravity_vector_filtered,
        df_corr=df_corr_filtered, # プロット関数に相関データを渡す
        output_file=output_filtered,
        split_plots=SPLIT_PLOTS_BY_TIME,
        segment_duration=SPLIT_DURATION
    )

    print(f"{csv_file} の処理が完了しました。")

def main():
    """メイン処理"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python analyze_acceleration.py <CSVファイルのパス>")
        print("  python analyze_acceleration.py <フォルダのパス>")
        print("")
        print("例:")
        print("  python analyze_acceleration.py both/20250924_131546_accel_log_serial_trial_1.csv")
        print("  python analyze_acceleration.py both/")
        return

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"ファイルまたはフォルダが見つかりません: {input_path}")
        return

    # 対象ファイルを検索
    target_files = find_target_csv_files(input_path)

    if not target_files:
        print(f"対象ファイル (YYYYMMDD_HHMMSS_accel_log_serial_trial_1.csv) が見つかりません: {input_path}")
        return

    print(f"見つかった対象ファイル数: {len(target_files)}")
    for i, file in enumerate(target_files, 1):
        print(f"  {i:2d}. {file}")

    # フォルダが指定された場合は相関係数の一覧表示
    if os.path.isdir(input_path) and len(target_files) > 1:
        print(f"\n=== 相関係数一覧表示 ===")
        plot_correlation_overview(target_files, input_path)

    if SKIP_PROCESS_FILE:
        return

    # 各ファイルを処理
    for i, csv_file in enumerate(target_files, 1):
        try:
            print(f"\n[{i}/{len(target_files)}] 処理中...")
            process_single_file(csv_file)
        except Exception as e:
            print(f"エラーが発生しました ({csv_file}): {e}")
            continue

    print(f"\n{'='*60}")
    print(f"全ての処理が完了しました。処理ファイル数: {len(target_files)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
