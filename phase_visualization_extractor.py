#!/usr/bin/env python3
"""
位相可視化抽出プログラム (phase_visualization_extractor.py)

phase_correlation_analyzerから特定の図（フィルタ済み信号と窓位相相関）を抽出し、
指定フォルダ内のphase/correlation_visualizationsに個別のPNGファイルとして保存する

機能:
1. integrated_analysis.csvファイルを再帰的に検索・読み込み
2. フィルタ済み信号の可視化とCSV出力
3. 窓位相相関の可視化とCSV出力
4. 各図を個別のPNGファイルとして保存
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

    # インポートできない場合のフォールバック関数
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
    print("フォールバック関数を使用します")

    # フォールバック関数（エラー時の最小限の実装）
    def normalize_signal(data, target_range=(-1, 1)):
        return data.copy()

    def apply_lowpass_filter(data, cutoff_freq=3.0, fs=60.0, order=4):
        return data.copy()

    def calculate_phase_from_signal(signal):
        return np.full(len(signal), np.nan)

    def calculate_phase_difference(phase1, phase2):
        return np.full(len(phase1), np.nan)

    def calculate_circular_correlation(phase1, phase2):
        return np.nan

    def calculate_windowed_phase_correlation(phase1, phase2, window_sec=10.0, fs=60.0):
        return np.full(len(phase1), np.nan)

    def trim_data_by_time_range(df, start_time=None, end_time=None, time_column='psychopy_time'):
        return df.copy(), {'trimmed': False, 'reason': 'Fallback mode'}

    def find_integrated_analysis_files(input_path):
        return []

# データ切り出し設定
DATA_START_TIME = 20  # 開始時刻（秒）
DATA_END_TIME = None    # 終了時刻（秒）


def save_normalization_info_csv(normalization_info, output_file):
    """
    規格化情報をCSVファイルに保存

    Args:
        normalization_info (dict): 規格化情報辞書
        output_file (str): 出力ファイルパス
    """
    try:
        # データを整理
        data_rows = []
        for signal_name, info in normalization_info.items():
            data_rows.append({
                'signal_name': signal_name,
                'original_mean_amplitude': info.get('mean_amplitude', np.nan),
                'scaling_factor': info.get('scaling_factor', np.nan),
                'data_mean': info.get('data_mean', np.nan),
                'target_mean_amplitude': info.get('target_mean_amplitude', np.nan)
            })

        # DataFrameを作成して保存
        norm_df = pd.DataFrame(data_rows)
        norm_df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"規格化情報CSV保存エラー: {e}")


def normalize_signal_with_info(data, target_range=(-1, 1)):
    """
    信号を平均振幅が1となるように規格化し、情報も返す

    Args:
        data (array): 入力データ
        target_range (tuple): 規格化後の範囲 (min, max) - デフォルト(-1, 1)

    Returns:
        tuple: (規格化データ, 規格化情報辞書)
    """
    try:
        # NaNを除外して統計量を計算
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            return data.copy(), {'mean_amplitude': np.nan, 'scaling_factor': np.nan, 'data_mean': np.nan}

        # 平均を計算
        data_mean = np.mean(clean_data)

        # 平均を引いて中心化
        centered_data = data - data_mean

        # 振幅（絶対値）の平均を計算
        mean_amplitude = np.mean(np.abs(centered_data[~np.isnan(centered_data)]))

        # 平均振幅がゼロの場合（一定値）はゼロ配列を返す
        if mean_amplitude == 0:
            return np.zeros_like(data), {'mean_amplitude': 0, 'scaling_factor': 0, 'data_mean': data_mean}

        # 平均振幅が指定した範囲での目標振幅になるように規格化
        target_min, target_max = target_range
        target_mean_amplitude = (target_max - target_min) / 2  # 目標平均振幅

        # スケーリング係数を計算
        scaling_factor = target_mean_amplitude / mean_amplitude

        # 平均振幅が目標値になるようにスケーリング
        amplitude_normalized = centered_data * scaling_factor

        # 中心をオフセット
        scaled = amplitude_normalized + (target_max + target_min) / 2

        # 規格化情報を作成
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


def load_and_process_data(filepath, cutoff_freq=3.0):
    """
    integrated_analysisファイルを読み込み、位相解析用にフィルタ処理

    Args:
        filepath (str): integrated_analysis.csvファイルのパス
        cutoff_freq (float): ローパスフィルタのカットオフ周波数 (Hz)

    Returns:
        tuple: (フィルタ処理済みデータフレーム, 規格化情報辞書)
    """
    try:
        # データ読み込み
        df = pd.read_csv(filepath)
        print(f"  - 元データ: {len(df)} samples, {len(df.columns)} columns")

        # データ切り出し処理
        if DATA_START_TIME is not None or DATA_END_TIME is not None:
            df, trim_info = trim_data_by_time_range(df, DATA_START_TIME, DATA_END_TIME)
            if not trim_info['trimmed']:
                print(f"  - データ切り出し失敗: {trim_info['reason']}")
        else:
            print(f"  - データ切り出し: 無効（全データを使用）")

        # サンプリング周波数の推定
        if len(df) > 1:
            time_interval = df['psychopy_time'].iloc[1] - df['psychopy_time'].iloc[0]
            estimated_fs = 1.0 / time_interval
            print(f"  - 推定サンプリング周波数: {estimated_fs:.1f}Hz")
        else:
            estimated_fs = 60.0

        # フィルタ処理結果を格納するデータフレーム
        filtered_df = df.copy()

        # 規格化情報を格納する辞書
        normalization_info = {}

        print(f"  - {cutoff_freq}Hzローパスフィルタ適用・平均振幅規格化:")

        # 角度データのフィルタ処理・平均振幅規格化
        if 'angle_change' in df.columns:
            filtered_data = apply_lowpass_filter(df['angle_change'].values, cutoff_freq, estimated_fs)
            filtered_df['angle_change_filtered'], norm_info = normalize_signal_with_info(filtered_data)
            normalization_info['angle_change'] = norm_info
            print(f"    - angle_change (平均振幅: {norm_info['mean_amplitude']:.4f})")

        # 視覚刺激データのフィルタ処理・平均振幅規格化
        visual_cols = ['red_dot_x_change', 'green_dot_x_change']
        for col in visual_cols:
            if col in df.columns:
                filtered_data = apply_lowpass_filter(df[col].values, cutoff_freq, estimated_fs)
                filtered_df[f'{col}_filtered'], norm_info = normalize_signal_with_info(filtered_data)
                normalization_info[col] = norm_info
                print(f"    - {col} (平均振幅: {norm_info['mean_amplitude']:.4f})")

        # GVSデータのフィルタ処理・平均振幅規格化
        if 'gvs_dac_output' in df.columns:
            filtered_data = apply_lowpass_filter(df['gvs_dac_output'].values, cutoff_freq, estimated_fs)
            filtered_df['gvs_dac_output_filtered'], norm_info = normalize_signal_with_info(filtered_data)
            normalization_info['gvs_dac_output'] = norm_info
            print(f"    - gvs_dac_output (平均振幅: {norm_info['mean_amplitude']:.4f})")

        # 音響データのフィルタ処理・平均振幅規格化
        if 'audio_angle_change' in df.columns:
            filtered_data = apply_lowpass_filter(df['audio_angle_change'].values, cutoff_freq, estimated_fs)
            filtered_df['audio_angle_change_filtered'], norm_info = normalize_signal_with_info(filtered_data)
            normalization_info['audio_angle_change'] = norm_info
            print(f"    - audio_angle_change (平均振幅: {norm_info['mean_amplitude']:.4f})")

        # 位相計算
        if 'angle_change_filtered' in filtered_df.columns:
            filtered_df['angle_change_phase'] = calculate_phase_from_signal(filtered_df['angle_change_filtered'].values)

        # 各刺激の位相計算
        stimulus_cols = [
            ('audio_angle_change_filtered', 'audio_phase'),
            ('gvs_dac_output_filtered', 'gvs_phase'),
            ('red_dot_x_change_filtered', 'red_dot_phase'),  
            ('green_dot_x_change_filtered', 'green_dot_phase')
        ]

        for filtered_col, phase_col in stimulus_cols:
            if filtered_col in filtered_df.columns:
                filtered_df[phase_col] = calculate_phase_from_signal(filtered_df[filtered_col].values)

        # 窓位相相関の計算
        if 'angle_change_phase' in filtered_df.columns:
            window_corr_cols = [
                ('audio_phase', 'phase_correlation_angle_audio'),
                ('gvs_phase', 'phase_correlation_angle_gvs'),
                ('red_dot_phase', 'phase_correlation_angle_red_dot'),
                ('green_dot_phase', 'phase_correlation_angle_green_dot')
            ]

            for phase_col, corr_col in window_corr_cols:
                if phase_col in filtered_df.columns:
                    filtered_df[corr_col] = calculate_windowed_phase_correlation(
                        filtered_df['angle_change_phase'].values,
                        filtered_df[phase_col].values,
                        window_sec=10.0,
                        fs=estimated_fs
                    )

        return filtered_df, normalization_info

    except Exception as e:
        print(f"エラー: データ処理に失敗: {e}")
        return None, None


def create_output_directory(folder_path):
    """
    出力ディレクトリを作成

    Args:
        folder_path (str): ベースフォルダパス

    Returns:
        tuple: (phase_dir, correlation_dir) 出力ディレクトリのパス
    """
    try:
        # phase/correlation_visualizationsディレクトリを作成
        output_base = os.path.join(folder_path, 'phase')
        phase_dir = os.path.join(output_base, 'filtered_signals')
        correlation_dir = os.path.join(output_base, 'correlation_visualizations')

        os.makedirs(phase_dir, exist_ok=True)
        os.makedirs(correlation_dir, exist_ok=True)

        return phase_dir, correlation_dir

    except Exception as e:
        print(f"出力ディレクトリ作成エラー: {e}")
        return None, None


def plot_filtered_signals(df, session_id, output_dir, cutoff_freq=3.0, is_scope_data=False, experiment_settings=None, condition='red', normalization_info=None):
    """
    フィルタ済み信号のグラフを作成・保存

    Args:
        df (pd.DataFrame): フィルタ処理済みデータフレーム
        session_id (str): セッションID
        output_dir (str): 出力ディレクトリ
        cutoff_freq (float): カットオフ周波数
        is_scope_data (bool): オシロスコープデータかどうか
        experiment_settings (dict): 実験設定
        condition (str): 実験条件
        normalization_info (dict): 規格化情報
    """
    try:
        if df is None or df.empty:
            print("フィルタ済み信号のデータがありません")
            return

        plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
        plt.rcParams["font.size"] = 15

        # 単一グラフを作成
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))

        data_source = "SCOPE" if is_scope_data else "SYNTH"

        # リバース設定の表示文字列を作成
        reverse_indicators = []
        if experiment_settings and experiment_settings.get('visual_reverse', False):
            reverse_indicators.append('視覚反転')
        if experiment_settings and experiment_settings.get('audio_reverse', False):
            reverse_indicators.append('音響反転')
        if experiment_settings and experiment_settings.get('gvs_reverse', False):
            reverse_indicators.append('GVS反転')
        if experiment_settings and experiment_settings.get('single_color_dot', False):
            visual_reverse = experiment_settings.get('visual_reverse', False)
            if visual_reverse:
                target_condition = condition
            else:
                target_condition = 'green' if condition == 'red' else 'red'
            reverse_indicators.append(f'単色ドット({target_condition})')

        reverse_suffix = f" [{', '.join(reverse_indicators)}]" if reverse_indicators else ""
        # title = f'フィルタ済み信号 ({cutoff_freq}Hz LPF) - {session_id} [{data_source}]{reverse_suffix}'

        # フィルタ済み信号をプロット
        # if 'angle_change_filtered' in df.columns:
        #     ax.plot(df['psychopy_time'], df['angle_change_filtered'], label='姿勢変化', color='orange', linewidth=1.5)
        # if 'audio_angle_change_filtered' in df.columns:
        #     ax.plot(df['psychopy_time'], df['audio_angle_change_filtered'], label='音響姿勢変化', color='darkblue', alpha=0.7)
        # if 'gvs_dac_output_filtered' in df.columns:
        #     ax.plot(df['psychopy_time'], df['gvs_dac_output_filtered'], label='GVS電流量', color='blue', alpha=0.7)
        # if 'red_dot_x_change_filtered' in df.columns:
        #     ax.plot(df['psychopy_time'], df['red_dot_x_change_filtered'], label='赤ドットX座標変化', color='red', alpha=0.7)
        # if 'green_dot_x_change_filtered' in df.columns:
        #     ax.plot(df['psychopy_time'], df['green_dot_x_change_filtered'], label='緑ドットX座標変化', color='green', alpha=0.7)

        # フィルタ済み信号をプロット
        if 'angle_change_filtered' in df.columns:
            ax.plot(df['psychopy_time'], df['angle_change_filtered'], label='身体動揺', color='orange', linewidth=1.5)
        if 'audio_angle_change_filtered' in df.columns:
            ax.plot(df['psychopy_time'], df['audio_angle_change_filtered'], label='聴覚刺激', color='darkblue', alpha=0.7)
        if 'gvs_dac_output_filtered' in df.columns:
            ax.plot(df['psychopy_time'], df['gvs_dac_output_filtered'], label='平衡感覚刺激', color='blue', alpha=0.7)
        if 'red_dot_x_change_filtered' in df.columns:
            ax.plot(df['psychopy_time'], df['red_dot_x_change_filtered'], label='視覚刺激（赤色フロー）', color='red', alpha=0.7)
        if 'green_dot_x_change_filtered' in df.columns:
            ax.plot(df['psychopy_time'], df['green_dot_x_change_filtered'], label='視覚刺激（赤色フロー）', color='green', alpha=0.7)

        # ax.set_title(title, fontsize=14)
        ax.set_ylabel('規格化振幅')
        ax.set_xlabel('時間 [s]')
        ax.set_ylim(-2.0, 2.2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # ファイル名作成
        base_name = os.path.splitext(session_id)[0] if '.' in session_id else session_id
        scope_suffix = "_scope" if is_scope_data else ""
        output_file = os.path.join(output_dir, f"{base_name}{scope_suffix}_filtered_signals_{cutoff_freq}Hz.png")

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"フィルタ済み信号グラフを保存: {os.path.basename(output_file)}")
        plt.close()

        # CSVファイルも保存
        csv_columns = ['psychopy_time']
        if 'angle_change_filtered' in df.columns:
            csv_columns.append('angle_change_filtered')
        if 'audio_angle_change_filtered' in df.columns:
            csv_columns.append('audio_angle_change_filtered')
        if 'gvs_dac_output_filtered' in df.columns:
            csv_columns.append('gvs_dac_output_filtered')
        if 'red_dot_x_change_filtered' in df.columns:
            csv_columns.append('red_dot_x_change_filtered')
        if 'green_dot_x_change_filtered' in df.columns:
            csv_columns.append('green_dot_x_change_filtered')

        csv_df = df[csv_columns].copy()
        csv_file = os.path.join(output_dir, f"{base_name}{scope_suffix}_filtered_signals_{cutoff_freq}Hz.csv")
        csv_df.to_csv(csv_file, index=False)
        print(f"フィルタ済み信号データを保存: {os.path.basename(csv_file)}")

        # 規格化情報をCSVファイルに保存
        if normalization_info:
            norm_file = os.path.join(output_dir, f"{base_name}{scope_suffix}_normalization_info_{cutoff_freq}Hz.csv")
            save_normalization_info_csv(normalization_info, norm_file)
            print(f"規格化情報を保存: {os.path.basename(norm_file)}")

    except Exception as e:
        print(f"フィルタ済み信号グラフ作成エラー: {e}")


def plot_window_correlations(df, session_id, output_dir, cutoff_freq=3.0, is_scope_data=False, experiment_settings=None, condition='red', normalization_info=None):
    """
    窓位相相関のグラフを作成・保存

    Args:
        df (pd.DataFrame): フィルタ処理済みデータフレーム
        session_id (str): セッションID
        output_dir (str): 出力ディレクトリ
        cutoff_freq (float): カットオフ周波数
        is_scope_data (bool): オシロスコープデータかどうか
        experiment_settings (dict): 実験設定
        condition (str): 実験条件
        normalization_info (dict): 規格化情報
    """
    try:
        if df is None or df.empty:
            print("窓位相相関のデータがありません")
            return

        plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

        # 単一グラフを作成
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))

        data_source = "SCOPE" if is_scope_data else "SYNTH"

        # リバース設定の表示文字列を作成
        reverse_indicators = []
        if experiment_settings and experiment_settings.get('visual_reverse', False):
            reverse_indicators.append('視覚反転')
        if experiment_settings and experiment_settings.get('audio_reverse', False):
            reverse_indicators.append('音響反転')
        if experiment_settings and experiment_settings.get('gvs_reverse', False):
            reverse_indicators.append('GVS反転')
        if experiment_settings and experiment_settings.get('single_color_dot', False):
            visual_reverse = experiment_settings.get('visual_reverse', False)
            if visual_reverse:
                target_condition = condition
            else:
                target_condition = 'green' if condition == 'red' else 'red'
            reverse_indicators.append(f'単色ドット({target_condition})')

        reverse_suffix = f" [{', '.join(reverse_indicators)}]" if reverse_indicators else ""
        # title = f'窓位相相関 (10秒窓) - {session_id} [{data_source}]{reverse_suffix}'

        # 窓位相相関をプロット
        window_corr_cols = [
            ('phase_correlation_angle_audio', '音響位相相関', 'darkblue'),
            ('phase_correlation_angle_gvs', 'GVS位相相関', 'blue'),
            ('phase_correlation_angle_red_dot', '赤ドット位相相関', 'red'),
            ('phase_correlation_angle_green_dot', '緑ドット位相相関', 'green')
        ]

        # 窓位相相関をプロット
        window_corr_cols = [
            ('phase_correlation_angle_audio', '聴覚刺激録音時の身体動揺位相相関', 'darkblue'),
            ('phase_correlation_angle_gvs', 'GVS刺激位相相関', 'blue'),
            ('phase_correlation_angle_red_dot', '対赤色フロー位相相関', 'red'),
            ('phase_correlation_angle_green_dot', '対緑色フロー位相相関', 'green')
        ]

        for corr_col, label, color in window_corr_cols:
            if corr_col in df.columns:
                ax.plot(df['psychopy_time'], df[corr_col], label=label, color=color, alpha=0.7)

        # ax.set_title(title, fontsize=14)
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
        scope_suffix = "_scope" if is_scope_data else ""
        output_file = os.path.join(output_dir, f"{base_name}{scope_suffix}_window_correlations_{cutoff_freq}Hz.png")

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"窓位相相関グラフを保存: {os.path.basename(output_file)}")
        plt.close()

        # CSVファイルも保存
        csv_columns = ['psychopy_time']
        for corr_col, _, _ in window_corr_cols:
            if corr_col in df.columns:
                csv_columns.append(corr_col)

        csv_df = df[csv_columns].copy()
        csv_file = os.path.join(output_dir, f"{base_name}{scope_suffix}_window_correlations_{cutoff_freq}Hz.csv")
        csv_df.to_csv(csv_file, index=False)
        print(f"窓位相相関データを保存: {os.path.basename(csv_file)}")

        # 規格化情報をCSVファイルに保存
        if normalization_info:
            norm_file = os.path.join(output_dir, f"{base_name}{scope_suffix}_window_correlations_normalization_info_{cutoff_freq}Hz.csv")
            save_normalization_info_csv(normalization_info, norm_file)
            print(f"規格化情報を保存: {os.path.basename(norm_file)}")

    except Exception as e:
        print(f"窓位相相関グラフ作成エラー: {e}")


def process_integrated_analysis_file(filepath, cutoff_freq=3.0):
    """
    単一のintegrated_analysisファイルを処理して図とCSVを作成

    Args:
        filepath (str): integrated_analysis.csvファイルのパス
        cutoff_freq (float): ローパスフィルタのカットオフ周波数

    Returns:
        bool: 処理成功時True、失敗時False
    """
    print(f"\n{'='*80}")
    print(f"処理中: {filepath}")
    print(f"{'='*80}")

    # ファイルパスからセッション情報を抽出
    folder_path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    # オシロスコープデータかどうかを判定
    is_scope_data = 'integrated_analysis_scope.csv' in filename
    data_source = "オシロスコープ実測" if is_scope_data else "合成音響"
    print(f"データソース: {data_source}")

    # セッションIDを抽出
    session_match = re.search(r'(\d{8}_\d{6})', filename)
    if session_match:
        session_id = session_match.group(1)
    else:
        session_id = os.path.splitext(filename)[0]

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
    filtered_df, normalization_info = load_and_process_data(filepath, cutoff_freq)

    if filtered_df is not None:
        # フィルタ済み信号の図とCSVを作成
        plot_filtered_signals(filtered_df, session_id, phase_dir, cutoff_freq, is_scope_data, experiment_settings, condition, normalization_info)

        # 窓位相相関の図とCSVを作成
        plot_window_correlations(filtered_df, session_id, correlation_dir, cutoff_freq, is_scope_data, experiment_settings, condition, normalization_info)

        print(f"処理完了: {session_id}")
        return True
    else:
        print(f"処理失敗: {session_id}")
        return False


def main():
    """メイン処理関数"""
    print("位相可視化抽出プログラム")
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

    # integrated_analysis.csvファイルを検索
    analysis_files = find_integrated_analysis_files(input_path)

    if not analysis_files:
        print(f"エラー: integrated_analysis.csvファイルが見つかりません: {input_path}")
        sys.exit(1)

    print(f"見つかったファイル数: {len(analysis_files)}")

    # 各ファイルを処理
    success_count = 0
    total_count = len(analysis_files)

    for filepath in analysis_files:
        if process_integrated_analysis_file(filepath, cutoff_freq):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"全体処理完了")
    print(f"総ファイル数: {total_count}")
    print(f"成功: {success_count}")
    print(f"失敗: {total_count - success_count}")
    print(f"出力先: 各フォルダ内の phase/filtered_signals/ と phase/correlation_visualizations/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
