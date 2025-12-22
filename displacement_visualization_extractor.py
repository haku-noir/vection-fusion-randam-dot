#!/usr/bin/env python3
"""
頭部変位-視覚刺激相関解析プログラム (displacement_visualization_extractor.py)

phase_visualization_extractorをベースに、angle_changeの代わりにdisplacement_x_cmを使用して
頭部変位と赤・緑ドットのx座標の相関を計算・可視化する

機能:
1. _head_displacement.csvファイルとintegrated_analysis_sway.csvファイルを検索・読み込み
2. displacement_x_cmと赤・緑ドットx座標のフィルタ済み信号の可視化とCSV出力
3. 窓位相相関の可視化とCSV出力
4. 各図を個別のPNGファイルとして保存

使用例:
    python displacement_visualization_extractor.py A/
    python displacement_visualization_extractor.py A/ 3.0
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt
import os
import sys
import glob
import re
from pathlib import Path

# analyze_datasから共通関数をインポート
try:
    from analyze_datas import (
        get_folder_type,
        get_experiment_settings_from_log,
        get_condition_from_experiment_log
    )
    print("analyze_datasから共通関数をインポートしました")
except ImportError as e:
    print(f"警告: analyze_datasからのインポートに失敗: {e}")
    print("独立実行モードで動作します")

    def get_experiment_settings_from_log(experiment_log_path, trial_number=1):
        return {'single_color_dot': False, 'visual_reverse': False, 'audio_reverse': False, 'gvs_reverse': False}

    def get_condition_from_experiment_log(experiment_log_path, trial_number=1):
        return 'red'

    def get_folder_type(folder_path):
        folder_name = os.path.basename(folder_path).lower()
        if 'audio' in folder_name:
            return 'audio'
        elif 'gvs' in folder_name:
            return 'gvs'
        elif 'vis' in folder_name:
            return 'visual'
        else:
            return 'all'

# phase_correlation_analyzerから共通関数をインポート
try:
    from phase_correlation_analyzer import (
        normalize_signal,
        apply_lowpass_filter,
        calculate_phase_from_signal,
        calculate_phase_difference,
        calculate_circular_correlation,
        calculate_windowed_phase_correlation,
        trim_data_by_time_range,
        find_integrated_analysis_files
    )
    print("phase_correlation_analyzerから共通関数をインポートしました")
except ImportError as e:
    print(f"警告: phase_correlation_analyzerからのインポートに失敗: {e}")
    print("ローカル関数を使用します")

    def apply_lowpass_filter(data, cutoff_freq=3.0, fs=60.0, order=4):
        """ローパスフィルタを適用"""
        try:
            nyq = fs / 2
            normalized_cutoff = cutoff_freq / nyq
            if normalized_cutoff >= 1:
                normalized_cutoff = 0.99
            b, a = butter(order, normalized_cutoff, btype='low')

            clean_data = data.copy()
            nan_mask = np.isnan(clean_data)
            if nan_mask.any():
                clean_data = np.interp(
                    np.arange(len(data)),
                    np.where(~nan_mask)[0],
                    data[~nan_mask]
                )

            filtered = filtfilt(b, a, clean_data)
            return filtered
        except Exception as e:
            print(f"フィルタエラー: {e}")
            return data.copy()

    def calculate_phase_from_signal(signal):
        """ヒルベルト変換で瞬時位相を計算"""
        try:
            clean_signal = signal.copy()
            nan_mask = np.isnan(clean_signal)
            if nan_mask.any():
                clean_signal = np.interp(
                    np.arange(len(signal)),
                    np.where(~nan_mask)[0],
                    signal[~nan_mask]
                )

            analytic = hilbert(clean_signal)
            phase = np.angle(analytic)
            return phase
        except Exception as e:
            print(f"位相計算エラー: {e}")
            return np.full(len(signal), np.nan)

    def calculate_windowed_phase_correlation(phase1, phase2, window_sec=10.0, fs=60.0):
        """窓位相相関を計算"""
        try:
            window_samples = int(window_sec * fs)
            half_window = window_samples // 2

            correlations = np.full(len(phase1), np.nan)

            for i in range(half_window, len(phase1) - half_window):
                p1_window = phase1[i-half_window:i+half_window]
                p2_window = phase2[i-half_window:i+half_window]

                valid = ~(np.isnan(p1_window) | np.isnan(p2_window))
                if valid.sum() > window_samples * 0.5:
                    cos1 = np.cos(p1_window[valid])
                    sin1 = np.sin(p1_window[valid])
                    cos2 = np.cos(p2_window[valid])
                    sin2 = np.sin(p2_window[valid])

                    numerator = np.sum(np.sin(p1_window[valid] - p2_window[valid]))
                    denominator = np.sqrt(
                        np.sum(np.sin(p1_window[valid])**2) * 
                        np.sum(np.sin(p2_window[valid])**2)
                    )

                    if denominator > 0:
                        correlations[i] = np.mean(np.cos(p1_window[valid] - p2_window[valid]))

            return correlations
        except Exception as e:
            print(f"窓相関計算エラー: {e}")
            return np.full(len(phase1), np.nan)

    def trim_data_by_time_range(df, start_time=None, end_time=None, time_column='psychopy_time'):
        """時間範囲でデータを切り出し"""
        result = df.copy()
        if start_time is not None:
            result = result[result[time_column] >= start_time]
        if end_time is not None:
            result = result[result[time_column] <= end_time]
        return result, {'trimmed': True, 'reason': 'OK'}


# データ切り出し設定
DATA_START_TIME = 20  # 開始時刻（秒）
DATA_END_TIME = None  # 終了時刻（秒）

# 日本語フォント設定
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
plt.rcParams["font.size"] = 15


def find_sway_files(input_path):
    """
    指定パスからsway CSVファイルを検索

    Args:
        input_path (str): ファイルまたはディレクトリのパス

    Returns:
        list: 見つかったCSVファイルのパスリスト
    """
    input_path = Path(input_path)

    if input_path.is_file():
        return [str(input_path)]

    if not input_path.is_dir():
        print(f"エラー: パスが見つかりません: {input_path}")
        return []

    # 再帰的に検索（_sway_を含むintegrated_analysisファイル）
    pattern = "*integrated_analysis_sway*.csv"
    all_files = list(input_path.rglob(pattern))

    # _head_displacementを除外
    filtered_files = [f for f in all_files if '_head_displacement' not in f.name]

    return sorted([str(f) for f in filtered_files])


def save_normalization_info_csv(normalization_info, output_file):
    """
    規格化情報をCSVファイルに保存
    """
    try:
        data_rows = []
        for signal_name, info in normalization_info.items():
            data_rows.append({
                'signal_name': signal_name,
                'original_mean_amplitude': info.get('mean_amplitude', np.nan),
                'scaling_factor': info.get('scaling_factor', np.nan),
                'data_mean': info.get('data_mean', np.nan),
                'target_mean_amplitude': info.get('target_mean_amplitude', np.nan)
            })

        norm_df = pd.DataFrame(data_rows)
        norm_df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"規格化情報CSV保存エラー: {e}")


def normalize_signal_with_info(data, target_range=(-1, 1)):
    """
    信号を平均振幅が1となるように規格化し、情報も返す
    """
    try:
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            return data.copy(), {'mean_amplitude': np.nan, 'scaling_factor': np.nan, 'data_mean': np.nan}

        data_mean = np.mean(clean_data)
        centered_data = data - data_mean
        mean_amplitude = np.mean(np.abs(centered_data[~np.isnan(centered_data)]))

        if mean_amplitude == 0:
            return np.zeros_like(data), {'mean_amplitude': 0, 'scaling_factor': 0, 'data_mean': data_mean}

        target_min, target_max = target_range
        target_mean_amplitude = (target_max - target_min) / 2
        scaling_factor = target_mean_amplitude / mean_amplitude
        amplitude_normalized = centered_data * scaling_factor
        scaled = amplitude_normalized + (target_max + target_min) / 2

        norm_info = {
            'mean_amplitude': mean_amplitude,
            'scaling_factor': scaling_factor,
            'data_mean': data_mean,
            'target_mean_amplitude': target_mean_amplitude
        }

        return scaled, norm_info

    except Exception as e:
        print(f"平均振幅規格化エラー: {e}")
        return data.copy(), {'mean_amplitude': np.nan, 'scaling_factor': np.nan, 'data_mean': np.nan}


def load_and_merge_data(sway_filepath, cutoff_freq=3.0):
    """
    swayファイルとhead_displacementファイルを読み込み、位相解析用にフィルタ処理

    Args:
        sway_filepath (str): integrated_analysis_sway.csvファイルのパス
        cutoff_freq (float): ローパスフィルタのカットオフ周波数 (Hz)

    Returns:
        tuple: (フィルタ処理済みデータフレーム, 規格化情報辞書)
    """
    try:
        # swayデータ読み込み
        df_sway = pd.read_csv(sway_filepath)
        print(f"  - swayデータ: {len(df_sway)} samples")

        # 対応するhead_displacementファイルを検索
        sway_dir = os.path.dirname(sway_filepath)
        sway_basename = os.path.basename(sway_filepath)

        # セッションIDを抽出
        session_match = re.search(r'(\d{8}_\d{6})', sway_basename)
        if session_match:
            session_id = session_match.group(1)
        else:
            session_id = None

        # カットオフ周波数を抽出
        freq_match = re.search(r'_sway_(\d+\.?\d*)Hz', sway_basename)
        if freq_match:
            file_cutoff = freq_match.group(1)
        else:
            file_cutoff = str(cutoff_freq)

        # head_displacementファイルを検索（globで柔軟に検索）
        displacement_pattern = f"{session_id}*sway*{file_cutoff}*head_displacement.csv"
        displacement_files = glob.glob(os.path.join(sway_dir, displacement_pattern))

        df = df_sway.copy()

        if displacement_files:
            displacement_path = displacement_files[0]
            df_displacement = pd.read_csv(displacement_path)
            print(f"  - 変位データ: {len(df_displacement)} samples ({os.path.basename(displacement_path)})")

            # 同じ長さの場合はインデックスベースでマージ（浮動小数点誤差を回避）
            if len(df_sway) == len(df_displacement):
                print(f"  - 同一長さのためインデックスベースでマージ")
                for col in ['displacement_x_cm', 'displacement_y_cm', 'displacement_x_relative_cm', 'displacement_y_relative_cm']:
                    if col in df_displacement.columns:
                        df[col] = df_displacement[col].values
            else:
                # 長さが異なる場合はmerge_asofで近似マージ
                df_sway_sorted = df_sway.sort_values('psychopy_time').reset_index(drop=True)
                df_disp_sorted = df_displacement.sort_values('psychopy_time').reset_index(drop=True)

                df = pd.merge_asof(
                    df_sway_sorted, 
                    df_disp_sorted[['psychopy_time', 'displacement_x_cm', 'displacement_y_cm', 
                                    'displacement_x_relative_cm', 'displacement_y_relative_cm']],
                    on='psychopy_time',
                    direction='nearest',
                    tolerance=0.001  # 1ms以内の誤差を許容
                )

            print(f"  - マージ後: {len(df)} samples, displacement_x_cm有効: {df['displacement_x_cm'].notna().sum()}")
        else:
            print(f"  - 警告: 変位データが見つかりません: {displacement_pattern}")
            # displacement_x_cmが無い場合、roll_swayから計算
            if 'roll_sway' in df_sway.columns:
                effective_length_cm = 162.0  # デフォルト有効長
                df['displacement_x_cm'] = effective_length_cm * np.sin(np.radians(df_sway['roll_sway']))
                df['displacement_y_cm'] = effective_length_cm * np.sin(np.radians(df_sway['pitch_sway']))
                print(f"  - roll_swayからdisplacement_x_cmを計算（L={effective_length_cm}cm）")

        # データ切り出し処理
        if DATA_START_TIME is not None or DATA_END_TIME is not None:
            df, trim_info = trim_data_by_time_range(df, DATA_START_TIME, DATA_END_TIME)
            if not trim_info['trimmed']:
                print(f"  - データ切り出し失敗: {trim_info['reason']}")

        # サンプリング周波数の推定
        if len(df) > 1:
            time_interval = df['psychopy_time'].iloc[1] - df['psychopy_time'].iloc[0]
            estimated_fs = 1.0 / time_interval
            print(f"  - 推定サンプリング周波数: {estimated_fs:.1f}Hz")
        else:
            estimated_fs = 60.0

        # フィルタ処理結果を格納するデータフレーム
        filtered_df = df.copy()
        normalization_info = {}

        print(f"  - {cutoff_freq}Hzローパスフィルタ適用・平均振幅規格化:")

        # 頭部変位データのフィルタ処理・平均振幅規格化
        if 'displacement_x_cm' in df.columns:
            filtered_data = apply_lowpass_filter(df['displacement_x_cm'].values, cutoff_freq, estimated_fs)
            filtered_df['displacement_x_filtered'], norm_info = normalize_signal_with_info(filtered_data)
            normalization_info['displacement_x'] = norm_info
            print(f"    - displacement_x_cm (平均振幅: {norm_info['mean_amplitude']:.4f} cm)")

        # 視覚刺激データのフィルタ処理・平均振幅規格化
        # sway用の列名をチェック
        red_col = 'red_dot_x_change_sway' if 'red_dot_x_change_sway' in df.columns else 'red_dot_x_change'
        green_col = 'green_dot_x_change_sway' if 'green_dot_x_change_sway' in df.columns else 'green_dot_x_change'

        if red_col in df.columns:
            filtered_data = apply_lowpass_filter(df[red_col].values, cutoff_freq, estimated_fs)
            filtered_df['red_dot_x_change_filtered'], norm_info = normalize_signal_with_info(filtered_data)
            normalization_info['red_dot_x_change'] = norm_info
            print(f"    - {red_col} (平均振幅: {norm_info['mean_amplitude']:.4f})")

        if green_col in df.columns:
            filtered_data = apply_lowpass_filter(df[green_col].values, cutoff_freq, estimated_fs)
            filtered_df['green_dot_x_change_filtered'], norm_info = normalize_signal_with_info(filtered_data)
            normalization_info['green_dot_x_change'] = norm_info
            print(f"    - {green_col} (平均振幅: {norm_info['mean_amplitude']:.4f})")

        # 位相計算
        if 'displacement_x_filtered' in filtered_df.columns:
            filtered_df['displacement_x_phase'] = calculate_phase_from_signal(filtered_df['displacement_x_filtered'].values)

        # 各刺激の位相計算
        stimulus_cols = [
            ('red_dot_x_change_filtered', 'red_dot_phase'),
            ('green_dot_x_change_filtered', 'green_dot_phase')
        ]

        for filtered_col, phase_col in stimulus_cols:
            if filtered_col in filtered_df.columns:
                filtered_df[phase_col] = calculate_phase_from_signal(filtered_df[filtered_col].values)

        # 窓位相相関の計算
        if 'displacement_x_phase' in filtered_df.columns:
            window_corr_cols = [
                ('red_dot_phase', 'phase_correlation_displacement_red_dot'),
                ('green_dot_phase', 'phase_correlation_displacement_green_dot')
            ]

            for phase_col, corr_col in window_corr_cols:
                if phase_col in filtered_df.columns:
                    filtered_df[corr_col] = calculate_windowed_phase_correlation(
                        filtered_df['displacement_x_phase'].values,
                        filtered_df[phase_col].values,
                        window_sec=10.0,
                        fs=estimated_fs
                    )

        return filtered_df, normalization_info

    except Exception as e:
        print(f"エラー: データ処理に失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_output_directory(folder_path):
    """
    出力ディレクトリを作成
    """
    try:
        output_base = os.path.join(folder_path, 'displacement_phase')
        phase_dir = os.path.join(output_base, 'filtered_signals')
        correlation_dir = os.path.join(output_base, 'correlation_visualizations')

        os.makedirs(phase_dir, exist_ok=True)
        os.makedirs(correlation_dir, exist_ok=True)

        return phase_dir, correlation_dir

    except Exception as e:
        print(f"出力ディレクトリ作成エラー: {e}")
        return None, None


def plot_filtered_signals(df, session_id, output_dir, cutoff_freq=3.0, experiment_settings=None, condition='red', normalization_info=None):
    """
    フィルタ済み信号のグラフを作成・保存
    """
    try:
        if df is None or df.empty:
            print("フィルタ済み信号のデータがありません")
            return

        fig, ax = plt.subplots(1, 1, figsize=(15, 6))

        # フィルタ済み信号をプロット
        if 'displacement_x_filtered' in df.columns:
            ax.plot(df['psychopy_time'], df['displacement_x_filtered'], label='頭部左右変位', color='orange', linewidth=1.5)
        if 'red_dot_x_change_filtered' in df.columns:
            ax.plot(df['psychopy_time'], df['red_dot_x_change_filtered'], label='視覚刺激（赤色フロー）', color='red', alpha=0.7)
        if 'green_dot_x_change_filtered' in df.columns:
            ax.plot(df['psychopy_time'], df['green_dot_x_change_filtered'], label='視覚刺激（緑色フロー）', color='green', alpha=0.7)

        ax.set_ylabel('規格化振幅')
        ax.set_xlabel('時間 [s]')
        ax.set_ylim(-2.0, 2.2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # ファイル名作成
        base_name = os.path.splitext(session_id)[0] if '.' in session_id else session_id
        output_file = os.path.join(output_dir, f"{base_name}_displacement_filtered_signals_{cutoff_freq}Hz.png")

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"フィルタ済み信号グラフを保存: {os.path.basename(output_file)}")
        plt.close()

        # CSVファイルも保存
        csv_columns = ['psychopy_time']
        if 'displacement_x_filtered' in df.columns:
            csv_columns.append('displacement_x_filtered')
        if 'red_dot_x_change_filtered' in df.columns:
            csv_columns.append('red_dot_x_change_filtered')
        if 'green_dot_x_change_filtered' in df.columns:
            csv_columns.append('green_dot_x_change_filtered')

        csv_df = df[csv_columns].copy()
        csv_file = os.path.join(output_dir, f"{base_name}_displacement_filtered_signals_{cutoff_freq}Hz.csv")
        csv_df.to_csv(csv_file, index=False)
        print(f"フィルタ済み信号データを保存: {os.path.basename(csv_file)}")

        # 規格化情報をCSVファイルに保存
        if normalization_info:
            norm_file = os.path.join(output_dir, f"{base_name}_displacement_normalization_info_{cutoff_freq}Hz.csv")
            save_normalization_info_csv(normalization_info, norm_file)
            print(f"規格化情報を保存: {os.path.basename(norm_file)}")

    except Exception as e:
        print(f"フィルタ済み信号グラフ作成エラー: {e}")


def plot_window_correlations(df, session_id, output_dir, cutoff_freq=3.0, experiment_settings=None, condition='red', normalization_info=None):
    """
    窓位相相関のグラフを作成・保存
    """
    try:
        if df is None or df.empty:
            print("窓位相相関のデータがありません")
            return

        fig, ax = plt.subplots(1, 1, figsize=(15, 6))

        # 窓位相相関をプロット
        window_corr_cols = [
            ('phase_correlation_displacement_red_dot', '対赤色フロー位相相関', 'red'),
            ('phase_correlation_displacement_green_dot', '対緑色フロー位相相関', 'green')
        ]

        for corr_col, label, color in window_corr_cols:
            if corr_col in df.columns:
                ax.plot(df['psychopy_time'], df[corr_col], label=label, color=color, alpha=0.7)

        ax.set_ylabel('位相相関係数 [-1, 1]')
        ax.set_xlabel('時間 [s]')
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.3)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # ファイル名作成
        base_name = os.path.splitext(session_id)[0] if '.' in session_id else session_id
        output_file = os.path.join(output_dir, f"{base_name}_displacement_window_correlations_{cutoff_freq}Hz.png")

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"窓位相相関グラフを保存: {os.path.basename(output_file)}")
        plt.close()

        # CSVファイルも保存
        csv_columns = ['psychopy_time']
        for corr_col, _, _ in window_corr_cols:
            if corr_col in df.columns:
                csv_columns.append(corr_col)

        csv_df = df[csv_columns].copy()
        csv_file = os.path.join(output_dir, f"{base_name}_displacement_window_correlations_{cutoff_freq}Hz.csv")
        csv_df.to_csv(csv_file, index=False)
        print(f"窓位相相関データを保存: {os.path.basename(csv_file)}")

        # 規格化情報をCSVファイルに保存
        if normalization_info:
            norm_file = os.path.join(output_dir, f"{base_name}_displacement_window_correlations_normalization_info_{cutoff_freq}Hz.csv")
            save_normalization_info_csv(normalization_info, norm_file)
            print(f"規格化情報を保存: {os.path.basename(norm_file)}")

    except Exception as e:
        print(f"窓位相相関グラフ作成エラー: {e}")


def process_sway_file(filepath, cutoff_freq=3.0):
    """
    単一のswayファイルを処理して図とCSVを作成

    Args:
        filepath (str): integrated_analysis_sway.csvファイルのパス
        cutoff_freq (float): ローパスフィルタのカットオフ周波数

    Returns:
        bool: 処理成功時True、失敗時False
    """
    print(f"\n{'='*80}")
    print(f"処理中: {filepath}")
    print(f"{'='*80}")

    folder_path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    # セッションIDを抽出
    session_match = re.search(r'(\d{8}_\d{6})', filename)
    if session_match:
        session_id = session_match.group(1)
    else:
        session_id = os.path.splitext(filename)[0]

    # カットオフ周波数を抽出（3.0Hz形式）
    freq_match = re.search(r'_sway_(\d+\.\d+)Hz', filename)
    if freq_match:
        file_cutoff = float(freq_match.group(1))
        session_id = f"{session_id}_sway_{freq_match.group(1)}Hz"
    else:
        file_cutoff = cutoff_freq
        session_id = f"{session_id}_sway_{file_cutoff}Hz"

    print(f"セッションID: {session_id}")

    # 出力ディレクトリを作成
    phase_dir, correlation_dir = create_output_directory(folder_path)
    if phase_dir is None or correlation_dir is None:
        print(f"出力ディレクトリ作成失敗")
        return False

    # 実験設定を取得
    experiment_settings = {'single_color_dot': False, 'visual_reverse': False, 'audio_reverse': False, 'gvs_reverse': False}
    condition = 'red'

    try:
        log_files = list(Path(folder_path).glob('*experiment_log.csv'))
        if log_files:
            experiment_log_file = str(log_files[0])
            experiment_settings = get_experiment_settings_from_log(experiment_log_file)
            condition = get_condition_from_experiment_log(experiment_log_file)
    except Exception as e:
        print(f"実験設定取得エラー: {e}")

    # データを読み込み・処理
    filtered_df, normalization_info = load_and_merge_data(filepath, file_cutoff)

    if filtered_df is not None:
        # フィルタ済み信号の図とCSVを作成
        plot_filtered_signals(filtered_df, session_id, phase_dir, file_cutoff, experiment_settings, condition, normalization_info)

        # 窓位相相関の図とCSVを作成
        plot_window_correlations(filtered_df, session_id, correlation_dir, file_cutoff, experiment_settings, condition, normalization_info)

        print(f"処理完了: {session_id}")
        return True
    else:
        print(f"処理失敗: {session_id}")
        return False


def main():
    """メイン処理関数"""
    print("頭部変位-視覚刺激相関解析プログラム")
    print("=" * 50)

    # コマンドライン引数の処理
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = '.'
        print("引数が指定されていません。現在のディレクトリを検索します。")

    # カットオフ周波数の設定
    cutoff_freq = 3.0
    if len(sys.argv) > 2:
        try:
            cutoff_freq = float(sys.argv[2])
        except ValueError:
            print(f"警告: 無効な周波数指定 '{sys.argv[2]}'。デフォルト3Hzを使用します。")

    print(f"入力パス: {input_path}")
    print(f"ローパスフィルタ周波数: {cutoff_freq}Hz")

    # データ切り出し設定の表示
    if DATA_START_TIME is not None or DATA_END_TIME is not None:
        start_str = f'{DATA_START_TIME:.1f}s' if DATA_START_TIME is not None else '開始から'
        end_str = f'{DATA_END_TIME:.1f}s' if DATA_END_TIME is not None else '終了まで'
        print(f"データ切り出し: {start_str} - {end_str}")
    else:
        print(f"データ切り出し: 無効（全データを使用）")
    print()

    # swayファイルを検索
    sway_files = find_sway_files(input_path)

    if not sway_files:
        print(f"エラー: swayファイルが見つかりません: {input_path}")
        sys.exit(1)

    print(f"見つかったファイル数: {len(sway_files)}")

    # 各ファイルを処理
    success_count = 0
    total_count = len(sway_files)

    for filepath in sway_files:
        if process_sway_file(filepath, cutoff_freq):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"全体処理完了")
    print(f"総ファイル数: {total_count}")
    print(f"成功: {success_count}")
    print(f"失敗: {total_count - success_count}")
    print(f"出力先: 各フォルダ内の displacement_phase/filtered_signals/ と displacement_phase/correlation_visualizations/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
