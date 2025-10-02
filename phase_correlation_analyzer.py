#!/usr/bin/env python3
"""
位相相関解析プログラム (phase_correlation_analyzer.py)

integrated_analysisファイルを読み込み、3Hzローパスフィルタを適用して
angle_changeと各刺激との位相差・位相相関を計算・解析する

機能:
1. integrated_analysis.csvファイルを再帰的に検索・読み込み
2. 加速度、視覚刺激、音響、GVSデータに3Hzローパスフィルタ適用
3. angle_changeと各刺激との位相差計算
4. 位相相関（circular correlation）の計算
5. 位相解析結果のCSV出力とグラフ作成

使用例:
    python phase_correlation_analyzer.py hatano/
    python phase_correlation_analyzer.py .
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
        plot_integrated_data,
        save_angle_data_to_csv,
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

# データ切り出し設定
# 指定した時間範囲でデータを切り出して処理する
# Noneの場合は全データを使用
DATA_START_TIME = 20  # 開始時刻（秒）、例: 10.0
DATA_END_TIME = None    # 終了時刻（秒）、例: 60.0

# 例: 10秒から60秒までのデータを使用する場合
# DATA_START_TIME = 10.0
# DATA_END_TIME = 60.0

ENABLE_AUDIO_FILTER = False


def normalize_signal(data, target_range=(-1, 1)):
    """
    信号を指定した範囲に正規化

    Args:
        data (array): 入力データ
        target_range (tuple): 正規化後の範囲 (min, max) - デフォルト(-1, 1)

    Returns:
        array: 正規化されたデータ
    """
    try:
        # NaNを除外してデータ範囲を計算
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            return data.copy()
        
        data_min = np.min(clean_data)
        data_max = np.max(clean_data)
        
        # データ範囲がゼロの場合は正規化しない
        if data_max == data_min:
            return np.zeros_like(data)
        
        # 0~1に正規化
        normalized = (data - data_min) / (data_max - data_min)
        
        # 目標範囲にスケーリング
        target_min, target_max = target_range
        scaled = normalized * (target_max - target_min) + target_min
        
        return scaled
        
    except Exception as e:
        print(f"正規化エラー: {e}")
        return data.copy()


def apply_lowpass_filter(data, cutoff_freq=3.0, fs=60.0, order=4):
    """
    ローパスフィルタを適用して身体動揺成分を抽出

    Args:
        data (array): 入力データ
        cutoff_freq (float): カットオフ周波数 (Hz) - デフォルト3Hz
        fs (float): サンプリング周波数 (Hz) - デフォルト60Hz
        order (int): フィルタ次数 - デフォルト4次

    Returns:
        array: フィルタ処理されたデータ
    """
    # ナイキスト周波数
    nyquist = fs / 2.0

    # 正規化されたカットオフ周波数
    normal_cutoff = cutoff_freq / nyquist

    # カットオフ周波数がナイキスト周波数以下であることを確認
    if normal_cutoff >= 1.0:
        print(f"警告: カットオフ周波数({cutoff_freq}Hz)がナイキスト周波数({nyquist}Hz)以上です")
        return data.copy()

    try:
        # 4次バターワースローパスフィルタを設計
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # ゼロ位相フィルタ適用（位相歪みなし）
        filtered_data = filtfilt(b, a, data)

        return filtered_data

    except Exception as e:
        print(f"フィルタ適用エラー: {e}")
        return data.copy()


def calculate_phase_from_signal(signal):
    """
    信号から位相を計算（ヒルベルト変換を使用）
    
    Args:
        signal (array): 入力信号
        
    Returns:
        array: 位相（ラジアン、-π ~ π）
    """
    try:
        # ヒルベルト変換で複素包絡を計算
        analytic_signal = hilbert(signal)
        
        # 位相を抽出
        phase = np.angle(analytic_signal)
        
        return phase
        
    except Exception as e:
        print(f"位相計算エラー: {e}")
        return np.full(len(signal), np.nan)


def calculate_phase_difference(phase1, phase2):
    """
    2つの位相間の位相差を計算
    
    Args:
        phase1 (array): 位相1（ラジアン）
        phase2 (array): 位相2（ラジアン）
        
    Returns:
        array: 位相差（ラジアン、-π ~ π）
    """
    try:
        # 位相差を計算
        phase_diff = phase1 - phase2
        
        # -π ~ π の範囲に正規化
        phase_diff = np.angle(np.exp(1j * phase_diff))
        
        return phase_diff
        
    except Exception as e:
        print(f"位相差計算エラー: {e}")
        return np.full(len(phase1), np.nan)


def calculate_circular_correlation(phase1, phase2):
    """
    2つの位相間の円形相関係数を計算
    
    Args:
        phase1 (array): 位相1（ラジアン）
        phase2 (array): 位相2（ラジアン）
        
    Returns:
        float: 円形相関係数（-1 ~ +1）
                +1: 同位相（0度差）
                 0: 直交位相（90度差）
                -1: 逆位相（180度差）
    """
    try:
        # NaNを除去
        mask = ~(np.isnan(phase1) | np.isnan(phase2))
        if np.sum(mask) < 10:  # 最小サンプル数チェック
            return np.nan
        
        clean_phase1 = phase1[mask]
        clean_phase2 = phase2[mask]
        
        # 位相差を計算
        phase_diff = calculate_phase_difference(clean_phase1, clean_phase2)
        
        # 円形相関係数を計算
        # mean(exp(i * phase_diff))の実部を取得
        # cos(位相差の平均) = 位相の一致度
        complex_mean = np.mean(np.exp(1j * phase_diff))
        circular_correlation = np.real(complex_mean)  # 実部を取得（-1 ~ +1）
        
        return circular_correlation
        
    except Exception as e:
        print(f"円形相関計算エラー: {e}")
        return np.nan


def calculate_windowed_phase_correlation(phase1, phase2, window_sec=10.0, fs=60.0):
    """
    窓相関（移動窓位相相関）を計算
    
    Args:
        phase1 (array): 位相1（例: angle_changeの位相）
        phase2 (array): 位相2（例: 刺激の位相）
        window_sec (float): 窓のサイズ（秒） - デフォルト10秒
        fs (float): サンプリング周波数 (Hz) - デフォルト60Hz
        
    Returns:
        array: 窓位相相関係数の時系列データ
    """
    try:
        # 窓サイズをサンプル数に変換
        window_samples = int(window_sec * fs)
        
        if window_samples >= len(phase1):
            return np.full(len(phase1), np.nan)
        
        # 結果配列の初期化
        correlations = np.full(len(phase1), np.nan)
        
        # 移動窓で位相相関を計算
        for i in range(window_samples // 2, len(phase1) - window_samples // 2):
            start_idx = i - window_samples // 2
            end_idx = i + window_samples // 2 + 1
            
            window_phase1 = phase1[start_idx:end_idx]
            window_phase2 = phase2[start_idx:end_idx]
            
            # 窓内の位相相関を計算
            correlations[i] = calculate_circular_correlation(window_phase1, window_phase2)
        
        return correlations
        
    except Exception as e:
        print(f"窓位相相関計算エラー: {e}")
        return np.full(len(phase1), np.nan)


def trim_data_by_time_range(df, start_time=None, end_time=None, time_column='psychopy_time'):
    """
    指定した時間範囲でデータフレームを切り出し
    
    Args:
        df (pd.DataFrame): 入力データフレーム
        start_time (float): 開始時刻（秒）、Noneの場合は最初から
        end_time (float): 終了時刻（秒）、Noneの場合は最後まで
        time_column (str): 時刻カラム名
        
    Returns:
        pd.DataFrame: 切り出されたデータフレーム
        dict: 切り出し情報
    """
    try:
        if time_column not in df.columns:
            print(f"警告: 時刻カラム'{time_column}'が見つかりません")
            return df.copy(), {'trimmed': False, 'reason': f'Missing {time_column} column'}
        
        original_length = len(df)
        original_duration = df[time_column].max() - df[time_column].min()
        
        # データの時間範囲を取得
        data_start = df[time_column].min()
        data_end = df[time_column].max()
        
        # 切り出し範囲を決定
        trim_start = data_start if start_time is None else max(data_start, start_time)
        trim_end = data_end if end_time is None else min(data_end, end_time)
        
        # 範囲チェック
        if trim_start >= trim_end:
            print(f"警告: 不正な時間範囲 - 開始: {trim_start:.1f}s, 終了: {trim_end:.1f}s")
            return df.copy(), {'trimmed': False, 'reason': 'Invalid time range'}
        
        # データを切り出し
        mask = (df[time_column] >= trim_start) & (df[time_column] <= trim_end)
        trimmed_df = df[mask].copy()
        
        if len(trimmed_df) == 0:
            print(f"警告: 指定した時間範囲にデータがありません")
            return df.copy(), {'trimmed': False, 'reason': 'No data in specified range'}
        
        # インデックスをリセット
        trimmed_df = trimmed_df.reset_index(drop=True)
        
        trimmed_length = len(trimmed_df)
        trimmed_duration = trimmed_df[time_column].max() - trimmed_df[time_column].min()
        
        trim_info = {
            'trimmed': True,
            'original_length': original_length,
            'trimmed_length': trimmed_length,
            'original_duration': original_duration,
            'trimmed_duration': trimmed_duration,
            'data_start': data_start,
            'data_end': data_end,
            'trim_start': trim_start,
            'trim_end': trim_end,
            'start_specified': start_time is not None,
            'end_specified': end_time is not None
        }
        
        print(f"  データ切り出し:")
        print(f"    元データ: {original_length}サンプル ({original_duration:.1f}秒)")
        print(f"    切り出し範囲: {trim_start:.1f}s - {trim_end:.1f}s")
        print(f"    切り出し後: {trimmed_length}サンプル ({trimmed_duration:.1f}秒)")
        
        return trimmed_df, trim_info
        
    except Exception as e:
        print(f"データ切り出しエラー: {e}")
        return df.copy(), {'trimmed': False, 'reason': f'Error: {e}'}


def find_integrated_analysis_files(input_path):
    """
    integrated_analysis.csvファイルを再帰的に検索
    """
    analysis_files = []

    if os.path.isfile(input_path) and ('integrated_analysis.csv' in input_path or 'integrated_analysis_scope.csv' in input_path):
        return [input_path]

    # 再帰的にintegrated_analysis関連ファイルを検索
    search_patterns = [
        '**/*integrated_analysis.csv',
        '**/*integrated_analysis_scope.csv'
    ]

    for pattern in search_patterns:
        files = list(Path(input_path).glob(pattern))
        analysis_files.extend(files)

    return [str(f) for f in sorted(analysis_files)]


def load_and_filter_integrated_data(filepath, cutoff_freq=3.0):
    """
    integrated_analysisファイルを読み込み、位相解析用にフィルタ処理
    
    Args:
        filepath (str): integrated_analysis.csvファイルのパス
        cutoff_freq (float): ローパスフィルタのカットオフ周波数 (Hz)
        
    Returns:
        pd.DataFrame: フィルタ処理済みデータフレーム
    """
    try:
        # データ読み込み
        df = pd.read_csv(filepath)
        print(f"\n統合データを読み込み: {os.path.basename(filepath)}")
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

        print(f"  - {cutoff_freq}Hzローパスフィルタ適用・正規化（位相解析用）:")

        # 角度データのフィルタ処理・正規化
        if 'angle_change' in df.columns:
            filtered_data = apply_lowpass_filter(df['angle_change'].values, cutoff_freq, estimated_fs)
            filtered_df['angle_change_filtered'] = normalize_signal(filtered_data)
            print(f"    - angle_change (正規化: -1~1)")

        # 視覚刺激データのフィルタ処理・正規化
        visual_cols = ['red_dot_mean_x', 'green_dot_mean_x', 'red_dot_x_change', 'green_dot_x_change']
        for col in visual_cols:
            if col in df.columns:
                filtered_data = apply_lowpass_filter(df[col].values, cutoff_freq, estimated_fs)
                filtered_df[f'{col}_filtered'] = normalize_signal(filtered_data)
                print(f"    - {col} (正規化: -1~1)")

        # GVSデータのフィルタ処理・正規化
        if 'gvs_dac_output' in df.columns:
            filtered_data = apply_lowpass_filter(df['gvs_dac_output'].values, cutoff_freq, estimated_fs)
            filtered_df['gvs_dac_output_filtered'] = normalize_signal(filtered_data)
            print(f"    - gvs_dac_output (正規化: -1~1)")

        # 音響データのフィルタ処理・正規化
        if 'audio_angle_change' in df.columns:
            filtered_data = apply_lowpass_filter(df['audio_angle_change'].values, cutoff_freq, estimated_fs)
            filtered_df['audio_angle_change_filtered'] = normalize_signal(filtered_data)
            print(f"    - audio_angle_change (正規化: -1~1)")

        print(f"  - フィルタ処理完了: {len(filtered_df)} samples, {len(filtered_df.columns)} columns")

        # 位相計算
        print(f"  - 位相計算:")
        
        if 'angle_change_filtered' in filtered_df.columns:
            filtered_df['angle_change_phase'] = calculate_phase_from_signal(filtered_df['angle_change_filtered'].values)
            print(f"    - angle_change位相")

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
                print(f"    - {filtered_col}の位相")

        # 位相差計算
        if 'angle_change_phase' in filtered_df.columns:
            print(f"  - 位相差計算:")
            
            phase_diff_cols = [
                ('audio_phase', 'audio_phase_diff'),
                ('gvs_phase', 'gvs_phase_diff'),
                ('red_dot_phase', 'red_dot_phase_diff'),
                ('green_dot_phase', 'green_dot_phase_diff')
            ]
            
            for phase_col, diff_col in phase_diff_cols:
                if phase_col in filtered_df.columns:
                    filtered_df[diff_col] = calculate_phase_difference(
                        filtered_df['angle_change_phase'].values,
                        filtered_df[phase_col].values
                    )
                    print(f"    - angle_change vs {phase_col}")

        # 窓位相相関の計算
        if 'angle_change_phase' in filtered_df.columns:
            print(f"  - 窓位相相関計算 (窓サイズ: 10秒):")
            
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
                    print(f"    - angle_change vs {phase_col}")

        print(f"  - 位相解析完了: {len(filtered_df)} samples, {len(filtered_df.columns)} columns")

        return filtered_df

    except Exception as e:
        print(f"エラー: データ読み込み・位相解析に失敗: {e}")
        return None


def calculate_phase_statistics(phase_data):
    """
    位相データの統計情報を計算
    
    Args:
        phase_data (array): 位相データ（ラジアン）
        
    Returns:
        dict: 統計情報
    """
    try:
        clean_data = phase_data[~np.isnan(phase_data)]
        if len(clean_data) == 0:
            return {
                'mean_phase': np.nan, 'std_phase': np.nan, 'concentration': np.nan,
                'mean_phase_deg': np.nan, 'std_phase_deg': np.nan
            }
        
        # 円形統計
        mean_phase = circmean(clean_data)
        std_phase = circstd(clean_data)
        
        # 集中度パラメータ（von Mises分布の推定）
        concentration = 1.0 / (std_phase ** 2) if std_phase > 0 else np.inf
        
        return {
            'mean_phase': mean_phase,
            'std_phase': std_phase,
            'concentration': concentration,
            'mean_phase_deg': np.degrees(mean_phase),
            'std_phase_deg': np.degrees(std_phase)
        }
        
    except Exception as e:
        print(f"位相統計計算エラー: {e}")
        return {
            'mean_phase': np.nan, 'std_phase': np.nan, 'concentration': np.nan,
            'mean_phase_deg': np.nan, 'std_phase_deg': np.nan
        }


def calculate_file_phase_correlations(df, session_id, experiment_settings, condition, cutoff_freq=3.0):
    """
    単一ファイルの位相相関係数と統計情報を計算
    
    Args:
        df (pd.DataFrame): フィルタ処理済みデータフレーム
        session_id (str): セッションID
        experiment_settings (dict): 実験設定
        condition (str): 実験条件
        cutoff_freq (float): 使用したカットオフ周波数
        
    Returns:
        dict: 位相相関係数、統計情報、実験情報を含むディクショナリ
    """
    try:
        result = {
            'session_id': session_id,
            'condition': condition,
            'single_color_dot': experiment_settings.get('single_color_dot', False),
            'visual_reverse': experiment_settings.get('visual_reverse', False),
            'audio_reverse': experiment_settings.get('audio_reverse', False),
            'gvs_reverse': experiment_settings.get('gvs_reverse', False),
            'data_points': len(df),
            'cutoff_freq': cutoff_freq
        }

        # 全体位相相関を計算
        if 'angle_change_phase' in df.columns:
            angle_phase = df['angle_change_phase'].values
            
            # 各刺激との位相相関
            phase_correlations = [
                ('audio_phase', 'audio_phase_corr'),
                ('gvs_phase', 'gvs_phase_corr'),
                ('red_dot_phase', 'red_dot_phase_corr'),
                ('green_dot_phase', 'green_dot_phase_corr')
            ]
            
            for phase_col, corr_col in phase_correlations:
                if phase_col in df.columns:
                    stimulus_phase = df[phase_col].values
                    result[corr_col] = calculate_circular_correlation(angle_phase, stimulus_phase)
                else:
                    result[corr_col] = np.nan

            # 位相差統計
            phase_diff_stats = [
                ('audio_phase_diff', 'audio_phase_diff'),
                ('gvs_phase_diff', 'gvs_phase_diff'),
                ('red_dot_phase_diff', 'red_dot_phase_diff'),
                ('green_dot_phase_diff', 'green_dot_phase_diff')
            ]
            
            for diff_col, prefix in phase_diff_stats:
                if diff_col in df.columns:
                    phase_stats = calculate_phase_statistics(df[diff_col].values)
                    result.update({
                        f'{prefix}_mean': phase_stats['mean_phase_deg'],
                        f'{prefix}_std': phase_stats['std_phase_deg'],
                        f'{prefix}_concentration': phase_stats['concentration']
                    })
                else:
                    result.update({
                        f'{prefix}_mean': np.nan,
                        f'{prefix}_std': np.nan,
                        f'{prefix}_concentration': np.nan
                    })

            # 窓位相相関統計
            window_corr_stats = [
                ('phase_correlation_angle_audio', 'audio_window_phase_corr'),
                ('phase_correlation_angle_gvs', 'gvs_window_phase_corr'),
                ('phase_correlation_angle_red_dot', 'red_dot_window_phase_corr'),
                ('phase_correlation_angle_green_dot', 'green_dot_window_phase_corr')
            ]
            
            for corr_col, prefix in window_corr_stats:
                if corr_col in df.columns:
                    window_corr = df[corr_col].values
                    clean_corr = window_corr[~np.isnan(window_corr)]
                    if len(clean_corr) > 0:
                        result.update({
                            f'{prefix}_mean': np.mean(clean_corr),
                            f'{prefix}_std': np.std(clean_corr),
                            f'{prefix}_min': np.min(clean_corr),
                            f'{prefix}_max': np.max(clean_corr)
                        })
                    else:
                        result.update({
                            f'{prefix}_mean': np.nan,
                            f'{prefix}_std': np.nan,
                            f'{prefix}_min': np.nan,
                            f'{prefix}_max': np.nan
                        })
                else:
                    result.update({
                        f'{prefix}_mean': np.nan,
                        f'{prefix}_std': np.nan,
                        f'{prefix}_min': np.nan,
                        f'{prefix}_max': np.nan
                    })

        return result

    except Exception as e:
        print(f"エラー: 位相相関係数・統計情報計算に失敗: {e}")
        return None


def save_phase_data(df, output_path, cutoff_freq=3.0):
    """
    位相解析データをCSVファイルに保存
    
    Args:
        df (pd.DataFrame): フィルタ処理済みデータフレーム
        output_path (str): 出力ファイルパス
        cutoff_freq (float): 使用したカットオフ周波数
    """
    try:
        # ファイル名に周波数情報を含める
        base_name = os.path.splitext(output_path)[0]
        if base_name.endswith('_scope'):
            phase_output_path = f"{base_name}_phase_analysis_{cutoff_freq}Hz.csv"
        else:
            phase_output_path = f"{base_name}_phase_analysis_{cutoff_freq}Hz.csv"

        # 位相解析関連の列のみを選択
        phase_columns = ['psychopy_time']

        # フィルタ済み信号
        filtered_cols = [col for col in df.columns if col.endswith('_filtered')]
        phase_columns.extend(filtered_cols)

        # 位相データ
        phase_cols = [col for col in df.columns if col.endswith('_phase')]
        phase_columns.extend(phase_cols)

        # 位相差データ
        phase_diff_cols = [col for col in df.columns if col.endswith('_phase_diff')]
        phase_columns.extend(phase_diff_cols)

        # 窓位相相関データ
        window_corr_cols = [col for col in df.columns if 'phase_correlation' in col]
        phase_columns.extend(window_corr_cols)

        # 元の重要な列も保持
        original_cols_to_keep = ['angle_change', 'audio_angle_change', 'gvs_dac_output', 
                               'red_dot_mean_x', 'green_dot_mean_x']
        for col in original_cols_to_keep:
            if col in df.columns:
                phase_columns.append(col)

        # データフレームを作成
        phase_df = df[phase_columns].copy()

        # 保存
        phase_df.to_csv(phase_output_path, index=False)

        print(f"位相解析データを保存: {os.path.basename(phase_output_path)}")
        print(f"  - 保存列数: {len(phase_columns)}")
        print(f"  - サンプル数: {len(phase_df)}")
        print(f"  - カットオフ周波数: {cutoff_freq}Hz")
        print(f"  - 位相相関データを含む")

        return phase_output_path

    except Exception as e:
        print(f"エラー: ファイル保存に失敗: {e}")
        return None


def save_integrated_phase_correlation_summary(correlation_data_list, folder_path, cutoff_freq=3.0):
    """
    フォルダ内の全ファイルの位相相関係数サマリーを統合して保存
    
    Args:
        correlation_data_list (list): 各ファイルの位相相関係数データのリスト
        folder_path (str): 出力フォルダパス
        cutoff_freq (float): 使用したカットオフ周波数
    """
    try:
        if not correlation_data_list:
            print("位相相関係数データがありません")
            return

        # 出力ファイルパス
        output_file = os.path.join(folder_path, f"phase_correlation_summary_{cutoff_freq}Hz.csv")

        # データフレーム作成
        df = pd.DataFrame(correlation_data_list)

        # CSV保存
        df.to_csv(output_file, index=False)

        print(f"統合位相相関係数サマリーを保存: {os.path.basename(output_file)}")
        print(f"  - 対象ファイル数: {len(correlation_data_list)}")
        print(f"  - 保存パス: {output_file}")

        return output_file

    except Exception as e:
        print(f"エラー: 統合位相相関係数サマリー保存に失敗: {e}")
        return None


def plot_phase_analysis(df, session_id, folder_path, folder_type, cutoff_freq=3.0, is_scope_data=False, experiment_settings=None, condition='red'):
    """
    位相解析データのグラフを作成
    
    Args:
        df (pd.DataFrame): フィルタ処理済みデータフレーム
        session_id (str): セッションID
        folder_path (str): フォルダパス
        folder_type (str): フォルダタイプ
        cutoff_freq (float): 使用したカットオフ周波数
        is_scope_data (bool): オシロスコープデータかどうか
        experiment_settings (dict): 実験設定
        condition (str): 実験条件
    """
    if experiment_settings is None:
        experiment_settings = {'single_color_dot': False, 'visual_reverse': False, 'audio_reverse': False, 'gvs_reverse': False}
    if df is None or df.empty:
        print("グラフ作成用のデータがありません")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

    # サブプロット数を決定（位相解析用に6個）
    fig, axes = plt.subplots(6, 1, figsize=(15, 18))

    data_source = "SCOPE" if is_scope_data else "SYNTH"

    # リバース設定の表示文字列を作成
    reverse_indicators = []
    if experiment_settings.get('visual_reverse', False):
        reverse_indicators.append('視覚反転')
    if experiment_settings.get('audio_reverse', False):
        reverse_indicators.append('音響反転')
    if experiment_settings.get('gvs_reverse', False):
        reverse_indicators.append('GVS反転')
    if experiment_settings.get('single_color_dot', False):
        visual_reverse = experiment_settings.get('visual_reverse', False)
        if visual_reverse:
            target_condition = condition
        else:
            target_condition = 'green' if condition == 'red' else 'red'
        reverse_indicators.append(f'単色ドット({target_condition})')

    reverse_suffix = f" [{', '.join(reverse_indicators)}]" if reverse_indicators else ""
    fig.suptitle(f'位相解析 ({cutoff_freq}Hz LPF) - {folder_type.upper()} Session: {session_id} [{data_source}]{reverse_suffix}', fontsize=16)

    # 1. フィルタ済み信号
    if 'angle_change_filtered' in df.columns:
        axes[0].plot(df['psychopy_time'], df['angle_change_filtered'], label='角度変化(正規化済み)', color='orange', linewidth=1.5)
        if 'audio_angle_change_filtered' in df.columns:
            axes[0].plot(df['psychopy_time'], df['audio_angle_change_filtered'], label='音響角度変化(正規化済み)', color='blue', alpha=0.7)
        if 'gvs_dac_output_filtered' in df.columns:
            axes[0].plot(df['psychopy_time'], df['gvs_dac_output_filtered'], label='GVS出力(正規化済み)', color='blue', alpha=0.7)
        if 'red_dot_x_change_filtered' in df.columns:
            axes[0].plot(df['psychopy_time'], df['red_dot_x_change_filtered'], label='赤ドット変化(正規化済み)', color='red', alpha=0.7)
        if 'green_dot_x_change_filtered' in df.columns:
            axes[0].plot(df['psychopy_time'], df['green_dot_x_change_filtered'], label='緑ドット変化(正規化済み)', color='green', alpha=0.7)

    axes[0].set_title(f'フィルタ済み信号 ({cutoff_freq}Hz LPF, 正規化済み)')
    axes[0].set_ylabel('正規化振幅 (-1~1)')
    axes[0].set_ylim(-1.2, 1.2)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 位相データ
    if 'angle_change_phase' in df.columns:
        axes[1].plot(df['psychopy_time'], np.degrees(df['angle_change_phase']), label='角度変化位相', color='orange', linewidth=1.5)
        if 'audio_phase' in df.columns:
            axes[1].plot(df['psychopy_time'], np.degrees(df['audio_phase']), label='音響位相', color='blue', alpha=0.7)
        if 'gvs_phase' in df.columns:
            axes[1].plot(df['psychopy_time'], np.degrees(df['gvs_phase']), label='GVS位相', color='blue', alpha=0.7)
        if 'red_dot_phase' in df.columns:
            axes[1].plot(df['psychopy_time'], np.degrees(df['red_dot_phase']), label='赤ドット位相', color='red', alpha=0.7)
        if 'green_dot_phase' in df.columns:
            axes[1].plot(df['psychopy_time'], np.degrees(df['green_dot_phase']), label='緑ドット位相', color='green', alpha=0.7)

    axes[1].set_title('位相データ')
    axes[1].set_ylabel('位相 (度)')
    axes[1].set_ylim(-180, 180)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 位相差
    phase_diff_cols = [
        ('audio_phase_diff', '音響位相差', 'blue'),
        ('gvs_phase_diff', 'GVS位相差', 'blue'),
        ('red_dot_phase_diff', '赤ドット位相差', 'red'),
        ('green_dot_phase_diff', '緑ドット位相差', 'green')
    ]

    for diff_col, label, color in phase_diff_cols:
        if diff_col in df.columns:
            axes[2].plot(df['psychopy_time'], np.degrees(df[diff_col]), label=label, color=color, alpha=0.7)

    axes[2].set_title('位相差（角度変化を基準）')
    axes[2].set_ylabel('位相差 (度)')
    axes[2].set_ylim(-180, 180)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 4. 窓位相相関
    window_corr_cols = [
        ('phase_correlation_angle_audio', '音響窓位相相関', 'blue'),
        ('phase_correlation_angle_gvs', 'GVS窓位相相関', 'blue'),
        ('phase_correlation_angle_red_dot', '赤ドット窓位相相関', 'red'),
        ('phase_correlation_angle_green_dot', '緑ドット窓位相相関', 'green')
    ]

    for corr_col, label, color in window_corr_cols:
        if corr_col in df.columns:
            axes[3].plot(df['psychopy_time'], df[corr_col], label=label, color=color, alpha=0.7)

    axes[3].set_title('窓位相相関 (10秒窓)')
    axes[3].set_ylabel('位相相関係数')
    axes[3].set_ylim(-1.2, 1.2)
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[3].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    axes[3].axhline(y=-0.5, color='gray', linestyle='--', alpha=0.3)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # 5. 全体位相相関の棒グラフ
    overall_correlations = []
    overall_labels = []
    overall_colors = []

    if 'angle_change_phase' in df.columns:
        angle_phase = df['angle_change_phase'].values

        # 各刺激との全体位相相関を計算
        correlation_items = [
            ('audio_phase', '音響', 'blue'),
            ('gvs_phase', 'GVS', 'blue'),
            ('red_dot_phase', '赤ドット', 'red'),
            ('green_dot_phase', '緑ドット', 'green')
        ]

        for phase_col, label, color in correlation_items:
            if phase_col in df.columns:
                stimulus_phase = df[phase_col].values
                corr_val = calculate_circular_correlation(angle_phase, stimulus_phase)
                if not np.isnan(corr_val):
                    overall_correlations.append(corr_val)
                    overall_labels.append(label)
                    overall_colors.append(color)

    if overall_correlations:
        bars = axes[4].bar(overall_labels, overall_correlations, color=overall_colors, alpha=0.7, edgecolor='black', linewidth=1)

        # 相関係数の値をバーの上に表示
        for bar, corr_val in zip(bars, overall_correlations):
            height = bar.get_height()
            axes[4].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr_val:.3f}', ha='center', va='bottom', fontweight='bold')

        axes[4].set_ylabel('位相相関係数', fontsize=10)
        axes[4].set_title('全体位相相関係数', fontsize=12)
        axes[4].set_ylim(-1.2, 1.2)
        axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[4].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        axes[4].axhline(y=-0.5, color='gray', linestyle='--', alpha=0.3)
        axes[4].grid(True, alpha=0.3, axis='y')
    else:
        axes[4].text(0.5, 0.5, '位相相関データなし', ha='center', va='center', transform=axes[4].transAxes)
        axes[4].set_title('全体位相相関係数')

    # 6. 位相差のヒストグラム
    phase_diff_data = []
    phase_diff_labels = []
    phase_diff_colors = []

    for diff_col, label, color in phase_diff_cols:
        if diff_col in df.columns:
            clean_diff = df[diff_col].dropna().values
            if len(clean_diff) > 0:
                phase_diff_data.append(np.degrees(clean_diff))
                phase_diff_labels.append(label)
                phase_diff_colors.append(color)

    if phase_diff_data:
        for i, (data, label, color) in enumerate(zip(phase_diff_data, phase_diff_labels, phase_diff_colors)):
            axes[5].hist(data, bins=36, alpha=0.6, label=label, color=color, density=True)

        axes[5].set_title('位相差分布')
        axes[5].set_xlabel('位相差 (度)')  
        axes[5].set_ylabel('密度')
        axes[5].set_xlim(-180, 180)
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
    else:
        axes[5].text(0.5, 0.5, '位相差データなし', ha='center', va='center', transform=axes[5].transAxes)
        axes[5].set_title('位相差分布')

    plt.tight_layout()

    # グラフを保存
    base_name = os.path.splitext(session_id)[0] if '.' in session_id else session_id
    scope_suffix = "_scope" if is_scope_data else ""
    output_file = os.path.join(folder_path, f"{base_name}{scope_suffix}_phase_analysis_{cutoff_freq}Hz.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"位相解析グラフを保存: {os.path.basename(output_file)}")
    plt.close()


def process_integrated_analysis_file(filepath, cutoff_freq=3.0):
    """
    単一のintegrated_analysisファイルを処理
    
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
    print(f"フォルダ: {folder_path}")

    # フォルダタイプを判定
    try:
        folder_type = get_folder_type(folder_path)
    except NameError:
        # get_folder_typeが利用できない場合のフォールバック
        folder_name = os.path.basename(folder_path).lower()
        if 'audio' in folder_name:
            folder_type = 'audio'
        elif 'gvs' in folder_name:
            folder_type = 'gvs'
        elif 'vis' in folder_name:
            folder_type = 'visual'
        else:
            folder_type = 'both'

    print(f"フォルダタイプ: {folder_type}")

    # 実験設定を取得
    experiment_settings = {'single_color_dot': False, 'visual_reverse': False, 'audio_reverse': False, 'gvs_reverse': False}
    condition = 'red'

    # experiment_logファイルを探索
    experiment_log_file = None
    try:
        log_files = list(Path(folder_path).glob('*experiment_log.csv'))
        if log_files:
            experiment_log_file = str(log_files[0])
            experiment_settings = get_experiment_settings_from_log(experiment_log_file)
            condition = get_condition_from_experiment_log(experiment_log_file)
            print(f"実験設定: {experiment_settings}")
            print(f"条件: {condition}")
    except Exception as e:
        print(f"実験設定取得エラー: {e}")

    # データを読み込み・位相解析処理
    filtered_df = load_and_filter_integrated_data(filepath, cutoff_freq)

    if filtered_df is not None:
        # 位相解析データを保存
        phase_data_file = save_phase_data(filtered_df, filepath, cutoff_freq)
        
        # 位相解析グラフを作成
        plot_phase_analysis(filtered_df, session_id, folder_path, folder_type, cutoff_freq, is_scope_data, experiment_settings, condition)
        
        # 位相相関係数を計算
        correlation_data = calculate_file_phase_correlations(filtered_df, session_id, experiment_settings, condition, cutoff_freq)
        
        print(f"位相解析完了: {session_id}")
        return correlation_data
    else:
        print(f"位相解析失敗: {session_id}")
        return None


def main():
    """メイン処理関数"""
    print("位相相関解析プログラム")
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

    # フォルダ別にファイルをグループ化
    folder_groups = {}
    for filepath in analysis_files:
        folder_path = os.path.dirname(filepath)
        if folder_path not in folder_groups:
            folder_groups[folder_path] = []
        folder_groups[folder_path].append(filepath)

    # フォルダごとに処理
    total_success_count = 0
    total_file_count = 0

    for folder_path, files in folder_groups.items():
        print(f"\n{'='*80}")
        print(f"フォルダ処理: {folder_path}")
        print(f"{'='*80}")

        correlation_data_list = []

        for filepath in files:
            total_file_count += 1
            correlation_data = process_integrated_analysis_file(filepath, cutoff_freq)
            
            if correlation_data:
                correlation_data_list.append(correlation_data)
                total_success_count += 1

        # フォルダ内の統合位相相関サマリーを保存
        if correlation_data_list:
            save_integrated_phase_correlation_summary(correlation_data_list, folder_path, cutoff_freq)

    print(f"\n{'='*80}")
    print(f"全体処理完了")
    print(f"総ファイル数: {total_file_count}")
    print(f"成功: {total_success_count}")
    print(f"失敗: {total_file_count - total_success_count}")
    print(f"処理フォルダ数: {len(folder_groups)}")
    print(f"ローパスフィルタ周波数: {cutoff_freq}Hz")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
