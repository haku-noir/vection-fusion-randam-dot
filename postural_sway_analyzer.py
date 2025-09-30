#!/usr/bin/env python3
"""
身体動揺解析プログラム (postural_sway_analyzer.py)

integrated_analysisファイルを読み込み、3Hzローパスフィルタを適用して
身体動揺成分のみを抽出し、解析・可視化を行う

機能:
1. integrated_analysis.csvファイルを再帰的に検索・読み込み
2. 加速度、視覚刺激、音響、GVSデータに3Hzローパスフィルタ適用
3. 身体動揺成分の抽出とCSV出力
4. グラフ作成（analyze_datasと同様の形式）

使用例:
    python postural_sway_analyzer.py hatano/
    python postural_sway_analyzer.py .
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
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
        save_angle_data_to_csv
    )
    print("analyze_datasから共通関数をインポートしました")
except ImportError as e:
    print(f"警告: analyze_datasからのインポートに失敗: {e}")
    print("独立実行モードで動作します")

ENABLE_AUDIO_FILTER = False

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

    技術的背景:
    - 身体動揺の主要成分: 0.1-3.0Hz
    - 3Hzローパス: 意図的な動きや高周波ノイズを除去
    - 4次バターワースフィルタ: 急峻な減衰特性で適切な分離
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


def calculate_windowed_correlation(signal1, signal2, window_sec=10.0, fs=60.0):
    """
    窓相関（移動窓相関）を計算

    Args:
        signal1 (array): 信号1（例: 角度変化）
        signal2 (array): 信号2（例: 視覚刺激、音響、GVS）
        window_sec (float): 窓のサイズ（秒） - デフォルト10秒
        fs (float): サンプリング周波数 (Hz) - デフォルト60Hz

    Returns:
        array: 窓相関係数の時系列データ

    技術的背景:
    - 窓相関: 一定時間窓での相関係数を時間的に移動計算
    - 10秒窓: 身体動揺応答の典型的な時定数を考慮
    - 刺激-応答の時間的変化を捉える
    """
    try:
        # 窓サイズをサンプル数に変換
        window_samples = int(window_sec * fs)

        if window_samples >= len(signal1):
            print(f"警告: 窓サイズ({window_samples})がデータ長({len(signal1)})以上です")
            return np.full(len(signal1), np.nan)

        # 結果配列の初期化
        correlations = np.full(len(signal1), np.nan)

        # 移動窓で相関を計算
        for i in range(window_samples // 2, len(signal1) - window_samples // 2):
            start_idx = i - window_samples // 2
            end_idx = i + window_samples // 2

            # 窓内のデータを抽出
            window_signal1 = signal1[start_idx:end_idx]
            window_signal2 = signal2[start_idx:end_idx]

            # NaNをチェック
            if np.any(np.isnan(window_signal1)) or np.any(np.isnan(window_signal2)):
                continue

            # 相関係数を計算
            if np.std(window_signal1) > 1e-10 and np.std(window_signal2) > 1e-10:
                correlation = np.corrcoef(window_signal1, window_signal2)[0, 1]
                correlations[i] = correlation

        return correlations

    except Exception as e:
        print(f"窓相関計算エラー: {e}")
        return np.full(len(signal1), np.nan)


def calculate_overall_correlation(signal1, signal2):
    """
    全体相関係数を計算
    
    Args:
        signal1 (array): 信号1（例: 角度変化）
        signal2 (array): 信号2（例: 音響角度変化、GVS DAC出力）
    
    Returns:
        float: 相関係数 (NaNの場合は計算不可)
    """
    try:
        # NaNを除去
        mask = ~(np.isnan(signal1) | np.isnan(signal2))
        if np.sum(mask) < 10:  # 最低10サンプル必要
            return np.nan
            
        clean_signal1 = signal1[mask]
        clean_signal2 = signal2[mask]
        
        # 標準偏差チェック
        if np.std(clean_signal1) < 1e-10 or np.std(clean_signal2) < 1e-10:
            return np.nan
            
        # 相関係数計算
        correlation = np.corrcoef(clean_signal1, clean_signal2)[0, 1]
        return correlation
        
    except Exception as e:
        print(f"全体相関計算エラー: {e}")
        return np.nan


def find_integrated_analysis_files(input_path):
    """
    integrated_analysis.csvファイルを再帰的に検索
    analyze_datasとanalyze_scope_datasの両方の出力ファイルに対応

    Args:
        input_path (str): 検索開始パス

    Returns:
        list: 見つかったファイルパスのリスト
    """
    analysis_files = []

    if os.path.isfile(input_path) and ('integrated_analysis.csv' in input_path or 'integrated_analysis_scope.csv' in input_path):
        return [input_path]

    # 再帰的にintegrated_analysis関連ファイルを検索
    # analyze_datasの出力: *integrated_analysis.csv
    # analyze_scope_datasの出力: *integrated_analysis_scope.csv
    search_patterns = [
        '**/*integrated_analysis.csv',
        '**/*integrated_analysis_scope.csv'
    ]
    
    for pattern in search_patterns:
        files = list(Path(input_path).glob(pattern))
        analysis_files.extend(files)

    return [str(f) for f in sorted(analysis_files)]


def load_and_filter_integrated_data(filepath, cutoff_freq=3.0,
                                   enable_audio_0_3hz_filter=ENABLE_AUDIO_FILTER,
                                   enable_gvs_sine_internal=True):
    """
    integrated_analysisファイルを読み込み、身体動揺成分を抽出

    Args:
        filepath (str): integrated_analysis.csvファイルのパス
        cutoff_freq (float): 基本ローパスフィルタのカットオフ周波数 (Hz)
        enable_audio_0_3hz_filter (bool): audio相関で0.3Hzフィルタを使用するかどうか
        enable_gvs_sine_internal (bool): GVS相関でsine_value_internal_swayを使用するかどうか

    Returns:
        pd.DataFrame: フィルタ処理済みデータフレーム

    技術的背景:
    - 基本フィルタ: 3Hzローパスで身体動揺成分を抽出
    - Audio専用フィルタ: 0.3Hzローパスで低周波数の応答を捉える
    - GVS sine_internal: より直接的な刺激信号として使用
    """
    try:
        # データ読み込み
        df = pd.read_csv(filepath)
        print(f"\n統合データを読み込み: {os.path.basename(filepath)}")
        print(f"  - 元データ: {len(df)} samples, {len(df.columns)} columns")

        # サンプリング周波数の推定（60Hzと仮定、時間間隔から計算）
        if len(df) > 1:
            time_interval = df['psychopy_time'].iloc[1] - df['psychopy_time'].iloc[0]
            estimated_fs = 1.0 / time_interval
            print(f"  - 推定サンプリング周波数: {estimated_fs:.1f}Hz")
        else:
            estimated_fs = 60.0
            print(f"  - デフォルトサンプリング周波数: {estimated_fs}Hz")

        # フィルタ処理結果を格納するデータフレーム
        filtered_df = df.copy()

        print(f"  - {cutoff_freq}Hzローパスフィルタ適用:")

        # 加速度データのフィルタ処理
        accel_cols = ['accel_x', 'accel_y', 'accel_z']
        for col in accel_cols:
            if col in df.columns:
                filtered_df[f'{col}_sway'] = apply_lowpass_filter(
                    df[col].values, cutoff_freq, estimated_fs
                )
                print(f"    {col} → {col}_sway")

        # 角度データのフィルタ処理
        angle_cols = ['roll', 'pitch', 'angle_change', 'roll_change', 'pitch_change']
        for col in angle_cols:
            if col in df.columns:
                filtered_df[f'{col}_sway'] = apply_lowpass_filter(
                    df[col].values, cutoff_freq, estimated_fs
                )
                print(f"    {col} → {col}_sway")

        # 視覚刺激データのフィルタ処理
        visual_cols = ['red_dot_mean_x', 'red_dot_mean_y', 'green_dot_mean_x', 'green_dot_mean_y',
                      'red_dot_x_change', 'green_dot_x_change']
        for col in visual_cols:
            if col in df.columns:
                filtered_df[f'{col}_sway'] = apply_lowpass_filter(
                    df[col].values, cutoff_freq, estimated_fs
                )
                print(f"    {col} → {col}_sway")

        # GVSデータのフィルタ処理
        gvs_cols = ['gvs_dac_output']
        for col in gvs_cols:
            if col in df.columns:
                filtered_df[f'{col}_sway'] = apply_lowpass_filter(
                    df[col].values, cutoff_freq, estimated_fs
                )
                print(f"    {col} → {col}_sway")

        # 音響データのフィルタ処理
        audio_cols = ['audio_angle_change']
        for col in audio_cols:
            if col in df.columns:
                filtered_df[f'{col}_sway'] = apply_lowpass_filter(
                    df[col].values, cutoff_freq, estimated_fs
                )
                print(f"    {col} → {col}_sway")

        print(f"  - フィルタ処理完了: {len(filtered_df)} samples, {len(filtered_df.columns)} columns")

        # 窓相関の計算
        print(f"  - 窓相関計算 (窓サイズ: 10秒):")

        if 'angle_change_sway' in filtered_df.columns:
            angle_sway = filtered_df['angle_change_sway'].values

            # 視覚刺激との相関（赤ドット・緑ドット）
            if 'red_dot_mean_x_sway' in filtered_df.columns:
                corr_red = calculate_windowed_correlation(
                    angle_sway, filtered_df['red_dot_mean_x_sway'].values, 10.0, estimated_fs
                )
                filtered_df['correlation_angle_red_dot'] = corr_red
                print(f"    角度変化 vs 赤ドットX座標")

            if 'green_dot_mean_x_sway' in filtered_df.columns:
                corr_green = calculate_windowed_correlation(
                    angle_sway, filtered_df['green_dot_mean_x_sway'].values, 10.0, estimated_fs
                )
                filtered_df['correlation_angle_green_dot'] = corr_green
                print(f"    角度変化 vs 緑ドットX座標")

            # GVSデータとの相関（sine_value_internalオプション付き）
            if enable_gvs_sine_internal and 'sine_value_internal_sway' in filtered_df.columns:
                corr_gvs_sine = calculate_windowed_correlation(
                    angle_sway, filtered_df['sine_value_internal_sway'].values, 10.0, estimated_fs
                )
                filtered_df['correlation_angle_gvs_sine'] = corr_gvs_sine
                print(f"    角度変化 vs GVS sine_value_internal")

            elif not enable_gvs_sine_internal:
                # 通常のDAC出力での相関計算
                if 'gvs_dac_output_sway' in filtered_df.columns:
                    corr_gvs = calculate_windowed_correlation(
                        angle_sway, filtered_df['gvs_dac_output_sway'].values, 10.0, estimated_fs
                    )
                    filtered_df['correlation_angle_gvs'] = corr_gvs
                    print(f"    角度変化 vs GVS DAC出力")

            # 音響データとの相関（0.3Hzフィルタオプション付き）
            if ENABLE_AUDIO_FILTER and enable_audio_0_3hz_filter and 'audio_angle_change_sway' in filtered_df.columns:
                print(f"    音響データに0.3Hzローパスフィルタを追加適用:")

                # 0.3Hzフィルタを適用した音響データ
                audio_0_3hz = apply_lowpass_filter(
                    filtered_df['audio_angle_change_sway'].values, 0.3, estimated_fs
                )
                filtered_df['audio_angle_change_0_3hz'] = audio_0_3hz  # グラフ用に保存
                corr_audio = calculate_windowed_correlation(
                    angle_sway, audio_0_3hz, 10.0, estimated_fs
                )
                filtered_df['correlation_angle_audio'] = corr_audio
                print(f"      角度変化 vs 音響角度変化 (0.3Hz LPF)")

            else:
                # 3Hz音響データとの相関（ENABLE_AUDIO_FILTERに関係なく常に実行）
                if 'audio_angle_change_sway' in filtered_df.columns:
                    corr_audio = calculate_windowed_correlation(
                        angle_sway, filtered_df['audio_angle_change_sway'].values, 10.0, estimated_fs
                    )
                    filtered_df['correlation_angle_audio'] = corr_audio
                    print(f"    角度変化 vs 音響角度変化 (3Hz LPF)")

        print(f"  - 窓相関計算完了: {len(filtered_df)} samples, {len(filtered_df.columns)} columns")

        return filtered_df

    except Exception as e:
        print(f"エラー: データ読み込み・フィルタ処理に失敗: {e}")
        return None


def save_sway_data(df, output_path, cutoff_freq=3.0):
    """
    身体動揺データをCSVファイルに保存

    Args:
        df (pd.DataFrame): フィルタ処理済みデータフレーム
        output_path (str): 出力ファイルパス
        cutoff_freq (float): 使用したカットオフ周波数
    """
    try:
        # ファイル名に周波数情報を含める
        base_name = os.path.splitext(output_path)[0]
        # analyze_scope_datasの出力ファイル(_scope.csv)の場合も適切に処理
        if base_name.endswith('_scope'):
            sway_output_path = f"{base_name}_sway_{cutoff_freq}Hz.csv"
        else:
            sway_output_path = f"{base_name}_sway_{cutoff_freq}Hz.csv"

        # 身体動揺関連の列のみを選択
        sway_columns = ['psychopy_time']

        # _swayサフィックスがついた列を抽出
        for col in df.columns:
            if col.endswith('_sway'):
                sway_columns.append(col)

        # 0.3Hz音響データの列を追加（ENABLE_AUDIO_FILTERがTrueの場合のみ）
        if ENABLE_AUDIO_FILTER:
            for col in df.columns:
                if col.endswith('_0_3hz'):
                    sway_columns.append(col)

        # 窓相関の列を追加
        for col in df.columns:
            if col.startswith('correlation_'):
                sway_columns.append(col)

        # 元の列も比較用に一部保持
        original_cols_to_keep = ['accel_x', 'accel_y', 'accel_z', 'angle_change', 
                               'red_dot_mean_x', 'green_dot_mean_x']
        for col in original_cols_to_keep:
            if col in df.columns:
                sway_columns.append(col)

        # データフレームを作成
        sway_df = df[sway_columns].copy()

        # 保存
        sway_df.to_csv(sway_output_path, index=False)

        print(f"身体動揺データを保存: {os.path.basename(sway_output_path)}")
        print(f"  - 保存列数: {len(sway_columns)}")
        print(f"  - サンプル数: {len(sway_df)}")
        print(f"  - カットオフ周波数: {cutoff_freq}Hz")
        print(f"  - 窓相関データを含む")

        return sway_output_path

    except Exception as e:
        print(f"エラー: ファイル保存に失敗: {e}")
        return None


def save_correlation_summary(df, output_path, cutoff_freq=3.0):
    """
    相関係数サマリーをCSVファイルに保存
    
    Args:
        df (pd.DataFrame): フィルタ処理済みデータフレーム
        output_path (str): 出力ファイルパス
        cutoff_freq (float): 使用したカットオフ周波数
    """
    try:
        # ファイル名の準備
        base_name = os.path.splitext(output_path)[0]
        if base_name.endswith('_scope'):
            corr_output_path = f"{base_name}_correlation_summary_{cutoff_freq}Hz.csv"
        else:
            corr_output_path = f"{base_name}_correlation_summary_{cutoff_freq}Hz.csv"
        
        correlations = []
        
        # ロール変化（angle_change_sway）との相関を計算
        if 'angle_change_sway' in df.columns:
            angle_sway = df['angle_change_sway'].values
            
            # 音響角度変化との相関
            if 'audio_angle_change_sway' in df.columns:
                audio_corr = calculate_overall_correlation(angle_sway, df['audio_angle_change_sway'].values)
                correlations.append({
                    'stimulus_type': 'audio_angle_change',
                    'filter_type': f'{cutoff_freq}Hz_LPF',
                    'correlation_coefficient': audio_corr,
                    'data_points': len(df),
                    'description': f'ロール変化 vs 音響角度変化 ({cutoff_freq}Hz LPF)'
                })
                
            # 0.3Hz音響データとの相関（利用可能な場合）
            if 'audio_angle_change_0_3hz' in df.columns:
                audio_0_3hz_corr = calculate_overall_correlation(angle_sway, df['audio_angle_change_0_3hz'].values)
                correlations.append({
                    'stimulus_type': 'audio_angle_change',
                    'filter_type': '0.3Hz_LPF',
                    'correlation_coefficient': audio_0_3hz_corr,
                    'data_points': len(df),
                    'description': 'ロール変化 vs 音響角度変化 (0.3Hz LPF)'
                })
                
            # GVS DACとの相関
            if 'gvs_dac_output_sway' in df.columns:
                gvs_corr = calculate_overall_correlation(angle_sway, df['gvs_dac_output_sway'].values)
                correlations.append({
                    'stimulus_type': 'gvs_dac_output',
                    'filter_type': f'{cutoff_freq}Hz_LPF',
                    'correlation_coefficient': gvs_corr,
                    'data_points': len(df),
                    'description': f'ロール変化 vs GVS DAC出力 ({cutoff_freq}Hz LPF)'
                })
                
            # 視覚刺激との相関
            if 'red_dot_mean_x_sway' in df.columns:
                red_corr = calculate_overall_correlation(angle_sway, df['red_dot_mean_x_sway'].values)
                correlations.append({
                    'stimulus_type': 'red_dot_x',
                    'filter_type': f'{cutoff_freq}Hz_LPF',
                    'correlation_coefficient': red_corr,
                    'data_points': len(df),
                    'description': f'ロール変化 vs 赤ドットX座標 ({cutoff_freq}Hz LPF)'
                })
                
            if 'green_dot_mean_x_sway' in df.columns:
                green_corr = calculate_overall_correlation(angle_sway, df['green_dot_mean_x_sway'].values)
                correlations.append({
                    'stimulus_type': 'green_dot_x',
                    'filter_type': f'{cutoff_freq}Hz_LPF',
                    'correlation_coefficient': green_corr,
                    'data_points': len(df),
                    'description': f'ロール変化 vs 緑ドットX座標 ({cutoff_freq}Hz LPF)'
                })
        
        # データフレーム作成・保存
        if correlations:
            corr_df = pd.DataFrame(correlations)
            corr_df.to_csv(corr_output_path, index=False)
            
            print(f"相関係数サマリーを保存: {os.path.basename(corr_output_path)}")
            print("相関係数:")
            for corr in correlations:
                if not np.isnan(corr['correlation_coefficient']):
                    print(f"  - {corr['description']}: {corr['correlation_coefficient']:.3f}")
                else:
                    print(f"  - {corr['description']}: N/A (計算不可)")
            
            return corr_output_path
        else:
            print("相関係数計算に必要なデータが見つかりませんでした")
            return None
            
    except Exception as e:
        print(f"エラー: 相関係数サマリー保存に失敗: {e}")
        return None


def plot_sway_data(df, session_id, folder_path, folder_type, cutoff_freq=3.0, is_scope_data=False):
    """
    身体動揺データのグラフを作成

    Args:
        df (pd.DataFrame): フィルタ処理済みデータフレーム
        session_id (str): セッションID
        folder_path (str): フォルダパス
        folder_type (str): フォルダタイプ
        cutoff_freq (float): 使用したカットオフ周波数
        is_scope_data (bool): オシロスコープデータかどうか
    """
    if df is None or df.empty:
        print("グラフ作成用のデータがありません")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

    # フォルダタイプに応じてサブプロット数を決定
    is_visual_only = folder_type not in ['gvs', 'audio'] or (
        folder_type == 'gvs' and 'gvs_dac_output_sway' not in df.columns
    ) or (
        folder_type == 'audio' and 'audio_angle_change_sway' not in df.columns
    )

    # audioの場合は0.3Hzプロット用に+1、相関プロット用に+1
    if folder_type == 'audio' and not is_visual_only and ENABLE_AUDIO_FILTER:
        subplot_count = 5  # 加速度、視覚、音響(3Hz)、音響(0.3Hz)、相関
    elif not is_visual_only:
        subplot_count = 4  # GVSの場合
    else:
        subplot_count = 3  # 視覚のみの場合

    fig, axes = plt.subplots(subplot_count, 1, figsize=(15, 12 if folder_type == 'audio' and not is_visual_only else (16 if not is_visual_only else 10)))

    # axesを常にリストとして扱う
    if subplot_count == 1:
        axes = [axes]
    else:
        axes = list(axes)

    data_source = "SCOPE" if is_scope_data else "SYNTH"
    fig.suptitle(f'身体動揺解析 ({cutoff_freq}Hz LPF) - {folder_type.upper()} Session: {session_id} [{data_source}]', fontsize=16)

    # サブプロット1: 加速度データ（身体動揺成分）
    if 'accel_x_sway' in df.columns:
        axes[0].plot(df['psychopy_time'], df['accel_x_sway'], label='X軸加速度(身体動揺)', alpha=0.7, linewidth=1.5)
        axes[0].plot(df['psychopy_time'], df['accel_y_sway'], label='Y軸加速度(身体動揺)', alpha=0.7, linewidth=1.5)
        axes[0].plot(df['psychopy_time'], df['accel_z_sway'], label='Z軸加速度(身体動揺)', alpha=0.7, linewidth=1.5)
    else:
        # フォールバック: 元のデータを表示
        axes[0].plot(df['psychopy_time'], df['accel_x'], label='X軸加速度', alpha=0.7)
        axes[0].plot(df['psychopy_time'], df['accel_y'], label='Y軸加速度', alpha=0.7)
        axes[0].plot(df['psychopy_time'], df['accel_z'], label='Z軸加速度', alpha=0.7)

    axes[0].set_title(f'加速度データ (身体動揺成分: {cutoff_freq}Hz LPF)')
    axes[0].set_ylabel('加速度 (m/s²)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # サブプロット2: 視覚刺激と角度変化の重ね合わせ（身体動揺成分）
    if ('red_dot_mean_x_sway' in df.columns and 'green_dot_mean_x_sway' in df.columns and 
        'angle_change_sway' in df.columns):

        ax2_1 = axes[1]
        # ドット位置（身体動揺成分、左軸）
        line1 = ax2_1.plot(df['psychopy_time'], df['red_dot_mean_x_sway'], 
                          color='red', alpha=0.7, label='赤ドットX座標', linewidth=1.5)
        line2 = ax2_1.plot(df['psychopy_time'], df['green_dot_mean_x_sway'], 
                          color='green', alpha=0.7, label='緑ドットX座標', linewidth=1.5)
        ax2_1.set_ylabel('X座標 (pixel)', color='black')
        ax2_1.tick_params(axis='y', labelcolor='black')

        # 角度変化（身体動揺成分、右軸）
        ax2_2 = ax2_1.twinx()
        line3 = ax2_2.plot(df['psychopy_time'], df['angle_change_sway'], 
                          color='orange', linewidth=2, alpha=0.8, label='ロール変化(身体動揺)')
        ax2_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
        ax2_2.set_ylabel('角度変化 (度)', color='orange')
        ax2_2.tick_params(axis='y', labelcolor='orange')

        # 凡例を結合
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2_1.legend(lines, labels, loc='upper left')
        ax2_1.set_title(f'視覚刺激と角度変化 (身体動揺成分: {cutoff_freq}Hz LPF)')
        ax2_1.grid(True, alpha=0.3)

    # サブプロット3: 刺激データと角度変化の重ね合わせ（身体動揺成分）
    if not is_visual_only:
        if folder_type == 'gvs' and 'gvs_dac_output_sway' in df.columns and 'angle_change_sway' in df.columns:
            ax3_1 = axes[2]
            # GVS DAC出力（身体動揺成分、左軸）
            line1 = ax3_1.plot(df['psychopy_time'], df['gvs_dac_output_sway'],
                              color='blue', alpha=0.7, label='GVS DAC出力', linewidth=1.5)

            if 'sine_value_internal_sway' in df.columns:
                line3 = ax3_1.plot(df['psychopy_time'], df['sine_value_internal_sway'], 
                                  color='purple', alpha=0.5, label='内部sin値(身体動揺)', linewidth=1.5)
            else:
                line3 = []

            ax3_1.set_ylabel('GVS出力値 (身体動揺成分)', color='blue')
            ax3_1.tick_params(axis='y', labelcolor='blue')

            # 角度変化（身体動揺成分、右軸）
            ax3_2 = ax3_1.twinx()
            line4 = ax3_2.plot(df['psychopy_time'], df['angle_change_sway'], 
                              color='orange', linewidth=2, alpha=0.8, label='ロール変化(身体動揺)')
            ax3_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
            ax3_2.set_ylabel('角度変化 (度)', color='orange')
            ax3_2.tick_params(axis='y', labelcolor='orange')

            # 凡例を結合
            all_lines = line1 + line2 + line3 + line4
            labels = [l.get_label() for l in all_lines]
            ax3_1.legend(all_lines, labels, loc='upper left')
            ax3_1.set_title(f'GVS刺激と角度変化 (身体動揺成分: {cutoff_freq}Hz LPF)')

        elif folder_type == 'audio' and 'angle_change_sway' in df.columns:
            ax3_1 = axes[2]

            # 音響データがある場合はプロットする
            if 'audio_angle_change_sway' in df.columns:
                line1 = ax3_1.plot(df['psychopy_time'], df['audio_angle_change_sway'], 
                                  color='blue', alpha=0.6, label='音響ロール変化(身体動揺)', linewidth=1.5)

                ax3_1.set_ylabel('音響角度変化 (身体動揺成分)', color='blue')
                ax3_1.tick_params(axis='y', labelcolor='blue')

                # 角度変化（身体動揺成分、右軸）
                ax3_2 = ax3_1.twinx()
                line_angle = ax3_2.plot(df['psychopy_time'], df['angle_change_sway'], 
                                       color='orange', linewidth=2, alpha=0.8, label='ロール変化(身体動揺)')
                ax3_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
                ax3_2.set_ylabel('角度変化 (度)', color='orange')
                ax3_2.tick_params(axis='y', labelcolor='orange')

                # 凡例を結合
                all_lines = line1 + line_angle
                labels = [l.get_label() for l in all_lines]
                ax3_1.legend(all_lines, labels, loc='upper left')
                ax3_1.set_title(f'音響刺激と角度変化 (身体動揺成分: {cutoff_freq}Hz LPF)')
            else:
                # 音響データがない場合は角度変化のみ
                axes[2].plot(df['psychopy_time'], df['angle_change_sway'], 
                           color='orange', linewidth=2, label='ロール変化(身体動揺)')
                axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[2].set_title(f'音響刺激と角度変化 (身体動揺成分: {cutoff_freq}Hz LPF)\\n（音響データなし）')
                axes[2].set_ylabel('角度変化 (度)')
                axes[2].legend()

        # 3番目のサブプロットがある場合のX軸ラベル設定
        if not is_visual_only:
            if folder_type == 'audio':
                # 音響の場合は常に3番目のサブプロット（3Hz音響プロット）にX軸とグリッドを設定
                axes[2].set_xlabel('時間 (秒)')
                axes[2].grid(True, alpha=0.3)
            elif folder_type != 'audio':
                # GVSの場合
                axes[2].set_xlabel('時間 (秒)')
                axes[2].grid(True, alpha=0.3)

    # 音響フォルダ用: 0.3Hzフィルタ適用音響プロット（相関プロットの1つ上）
    if ENABLE_AUDIO_FILTER and folder_type == 'audio' and not is_visual_only and 'audio_angle_change_0_3hz' in df.columns:
        audio_0_3hz_plot_idx = subplot_count - 2  # 相関プロットの1つ上

        ax4_1 = axes[audio_0_3hz_plot_idx]
        lines_0_3hz = []        # 0.3Hzフィルタ適用音響データ（左軸）
        line1 = ax4_1.plot(df['psychopy_time'], df['audio_angle_change_0_3hz'], 
                          color='darkblue', alpha=0.8, label='音響角度変化 (0.3Hz LPF)', linewidth=2)
        lines_0_3hz = line1

        ax4_1.set_ylabel('音響角度変化 (0.3Hz LPF)', color='darkblue')
        ax4_1.tick_params(axis='y', labelcolor='darkblue')

        # 角度変化（右軸）
        ax4_2 = ax4_1.twinx()
        line3 = ax4_2.plot(df['psychopy_time'], df['angle_change_sway'], 
                          color='orange', linewidth=2, alpha=0.8, label='角度変化(身体動揺)')
        ax4_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
        ax4_2.set_ylabel('角度変化 (度)', color='orange')
        ax4_2.tick_params(axis='y', labelcolor='orange')

        # 凡例を結合
        all_lines = lines_0_3hz + line3
        labels = [l.get_label() for l in all_lines]
        ax4_1.legend(all_lines, labels, loc='upper left')
        ax4_1.set_title('音響刺激（0.3Hz LPF）と角度変化の重ね合わせ')
        ax4_1.grid(True, alpha=0.3)

    # 相関プロット（最後のサブプロット）
    corr_plot_idx = subplot_count - 1

    # 視覚刺激との相関をプロット
    has_visual_corr = False
    if 'correlation_angle_red_dot' in df.columns:
        axes[corr_plot_idx].plot(df['psychopy_time'], df['correlation_angle_red_dot'], 
                                color='red', alpha=0.7, linewidth=1.5, label='角度変化 vs 赤ドット')
        has_visual_corr = True

    if 'correlation_angle_green_dot' in df.columns:
        axes[corr_plot_idx].plot(df['psychopy_time'], df['correlation_angle_green_dot'], 
                                color='green', alpha=0.7, linewidth=1.5, label='角度変化 vs 緑ドット')
        has_visual_corr = True

    # GVS/音響との相関をプロット
    has_stimulus_corr = False
    if folder_type == 'gvs':
        if 'correlation_angle_gvs_sine' in df.columns:
            axes[corr_plot_idx].plot(df['psychopy_time'], df['correlation_angle_gvs_sine'], 
                                    color='blue', alpha=0.7, linewidth=1.5, label='角度変化 vs GVS sine_internal')
            has_stimulus_corr = True
        elif 'correlation_angle_gvs' in df.columns:
            axes[corr_plot_idx].plot(df['psychopy_time'], df['correlation_angle_gvs'], 
                                    color='blue', alpha=0.7, linewidth=1.5, label='角度変化 vs GVS DAC')
            has_stimulus_corr = True

    elif folder_type == 'audio':
        if 'correlation_angle_audio' in df.columns:
            axes[corr_plot_idx].plot(df['psychopy_time'], df['correlation_angle_audio'], 
                                    color='blue', alpha=0.7, linewidth=1.5, label='角度変化 vs 音響角度変化')
            has_stimulus_corr = True

    # 相関プロットの設定
    axes[corr_plot_idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[corr_plot_idx].set_ylim(-1, 1)
    axes[corr_plot_idx].set_ylabel('相関係数')
    axes[corr_plot_idx].set_xlabel('時間 (秒)')

    if has_stimulus_corr and folder_type == 'gvs':
        title_suffix = 'vs 視覚刺激・GVS刺激'
    elif has_stimulus_corr and folder_type == 'audio':
        title_suffix = 'vs 視覚刺激・音刺激'
    elif has_visual_corr:
        title_suffix = 'vs 視覚刺激'
    else:
        title_suffix = '（データなし）'

    axes[corr_plot_idx].set_title(f'窓相関 (10秒窓): 角度変化 {title_suffix}')
    axes[corr_plot_idx].legend()
    axes[corr_plot_idx].grid(True, alpha=0.3)

    plt.tight_layout()

    # グラフを保存（周波数情報を含む）
    base_name = os.path.splitext(session_id)[0] if '.' in session_id else session_id
    scope_suffix = "_scope" if is_scope_data else ""
    output_file = os.path.join(folder_path, f"{base_name}{scope_suffix}_sway_analysis_{cutoff_freq}Hz.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"身体動揺解析グラフを保存: {os.path.basename(output_file)}")
    plt.close()


def plot_correlation_summary(df, session_id, folder_path, folder_type, cutoff_freq=3.0, is_scope_data=False):
    """
    相関係数サマリーの棒グラフを作成
    
    Args:
        df (pd.DataFrame): フィルタ処理済みデータフレーム
        session_id (str): セッションID
        folder_path (str): 出力フォルダパス
        folder_type (str): フォルダタイプ（'audio', 'gvs', 'visual', 'both'）
        cutoff_freq (float): カットオフ周波数
        is_scope_data (bool): scope解析データかどうか
    """
    try:
        correlations = []
        labels = []
        colors = []
        
        # ロール変化（angle_change_sway）との相関を計算
        if 'angle_change_sway' in df.columns:
            angle_sway = df['angle_change_sway'].values
            
            # 音響角度変化との相関
            if 'audio_angle_change_sway' in df.columns:
                audio_corr = calculate_overall_correlation(angle_sway, df['audio_angle_change_sway'].values)
                if not np.isnan(audio_corr):
                    correlations.append(audio_corr)
                    labels.append(f'音響角度変化\n({cutoff_freq}Hz LPF)')
                    colors.append('blue')
                    
            # 0.3Hz音響データとの相関
            if 'audio_angle_change_0_3hz' in df.columns:
                audio_0_3hz_corr = calculate_overall_correlation(angle_sway, df['audio_angle_change_0_3hz'].values)
                if not np.isnan(audio_0_3hz_corr):
                    correlations.append(audio_0_3hz_corr)
                    labels.append('音響角度変化\n(0.3Hz LPF)')
                    colors.append('darkblue')
                    
            # GVS DACとの相関
            if 'gvs_dac_output_sway' in df.columns:
                gvs_corr = calculate_overall_correlation(angle_sway, df['gvs_dac_output_sway'].values)
                if not np.isnan(gvs_corr):
                    correlations.append(gvs_corr)
                    labels.append(f'GVS DAC出力\n({cutoff_freq}Hz LPF)')
                    colors.append('red')
                    
            # 視覚刺激との相関
            if 'red_dot_mean_x_sway' in df.columns:
                red_corr = calculate_overall_correlation(angle_sway, df['red_dot_mean_x_sway'].values)
                if not np.isnan(red_corr):
                    correlations.append(red_corr)
                    labels.append(f'赤ドットX座標\n({cutoff_freq}Hz LPF)')
                    colors.append('red')
                    
            if 'green_dot_mean_x_sway' in df.columns:
                green_corr = calculate_overall_correlation(angle_sway, df['green_dot_mean_x_sway'].values)
                if not np.isnan(green_corr):
                    correlations.append(green_corr)
                    labels.append(f'緑ドットX座標\n({cutoff_freq}Hz LPF)')
                    colors.append('green')
        
        # グラフ作成
        if correlations:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bars = ax.bar(labels, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # 相関係数の値をバーの上に表示
            for bar, corr in zip(bars, correlations):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                       f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
            
            ax.set_ylabel('相関係数', fontsize=12)
            ax.set_title(f'ロール変化との相関係数 - {folder_type.upper()} Session: {session_id}', fontsize=14)
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 有意性のガイドライン
            ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='弱い相関 (±0.3)')
            ax.axhline(y=-0.3, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='中程度の相関 (±0.5)')
            ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
            ax.legend()
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # グラフを保存
            base_name = os.path.splitext(session_id)[0] if '.' in session_id else session_id
            scope_suffix = "_scope" if is_scope_data else ""
            output_file = os.path.join(folder_path, f"{base_name}{scope_suffix}_correlation_summary_{cutoff_freq}Hz.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"相関係数サマリーグラフを保存: {os.path.basename(output_file)}")
            plt.close()
            
            return output_file
        else:
            print("相関係数グラフ作成に必要なデータが見つかりませんでした")
            return None
            
    except Exception as e:
        print(f"エラー: 相関係数グラフ作成に失敗: {e}")
        return None


def process_integrated_analysis_file(filepath, cutoff_freq=3.0,
                                   enable_audio_0_3hz_filter=True,
                                   enable_gvs_sine_internal=True):
    """
    単一のintegrated_analysisファイルを処理
    analyze_datasとanalyze_scope_datasの両方の出力ファイルに対応

    Args:
        filepath (str): integrated_analysis.csvまたはintegrated_analysis_scope.csvファイルのパス
        cutoff_freq (float): ローパスフィルタのカットオフ周波数
        enable_audio_0_3hz_filter (bool): audio相関で0.3Hzフィルタを使用するかどうか
        enable_gvs_sine_internal (bool): GVS相関でsine_value_internal_swayを使用するかどうか

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

    # セッションIDを抽出（ファイル名から）
    session_match = re.search(r'(\d{8}_\d{6})', filename)
    if session_match:
        session_id = session_match.group(1)
    else:
        session_id = os.path.splitext(filename)[0].replace('_integrated_analysis_scope', '').replace('_integrated_analysis', '')

    print(f"セッションID: {session_id}")
    print(f"フォルダ: {folder_path}")

    # フォルダタイプを判定
    try:
        folder_type = get_folder_type(folder_path)
    except NameError:
        # analyze_datasをインポートできなかった場合のフォールバック
        folder_name = os.path.basename(folder_path).lower()
        if 'gvs' in folder_name:
            folder_type = 'gvs'
        elif 'audio' in folder_name:
            folder_type = 'audio'
        elif 'vis' in folder_name:
            folder_type = 'vis'
        else:
            folder_type = 'unknown'

    print(f"フォルダタイプ: {folder_type}")

    # データを読み込み・フィルタ処理
    filtered_df = load_and_filter_integrated_data(filepath, cutoff_freq,
                                                 enable_audio_0_3hz_filter,
                                                 enable_gvs_sine_internal)

    if filtered_df is not None:
        # 身体動揺データを保存
        sway_file_path = save_sway_data(filtered_df, filepath, cutoff_freq)

        # 相関係数サマリーを保存
        print("相関係数サマリーを計算・保存中...")
        corr_file_path = save_correlation_summary(filtered_df, filepath, cutoff_freq)
        
        # グラフを作成・保存
        if sway_file_path:
            print("身体動揺解析グラフを作成中...")
            plot_sway_data(filtered_df, session_id, folder_path, folder_type, cutoff_freq, is_scope_data)
            
            print("相関係数サマリーグラフを作成中...")
            plot_correlation_summary(filtered_df, session_id, folder_path, folder_type, cutoff_freq, is_scope_data)

        return True
    else:
        print("データ処理に失敗しました")
        return False


def main():
    """メイン処理関数"""
    print("身体動揺解析プログラム")
    print("=" * 50)

    # コマンドライン引数の処理
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = '.'
        print("引数が指定されていません。現在のディレクトリを検索します。")

    # カットオフ周波数の設定（コマンドライン引数で変更可能）
    cutoff_freq = 3.0
    if len(sys.argv) > 2:
        try:
            cutoff_freq = float(sys.argv[2])
        except ValueError:
            print(f"警告: 無効な周波数指定 '{sys.argv[2]}'。デフォルト3Hzを使用します。")

    # ON/OFFオプションの設定
    enable_audio_0_3hz_filter = True  # デフォルトON
    enable_gvs_sine_internal = True   # デフォルトON

    # コマンドライン引数でのON/OFF制御
    if '--no-audio-0-3hz' in sys.argv:
        enable_audio_0_3hz_filter = False
        print("Audio 0.3Hz フィルタ: OFF")
    else:
        print("Audio 0.3Hz フィルタ: ON")

    if '--no-gvs-sine-internal' in sys.argv:
        enable_gvs_sine_internal = False
        print("GVS sine_value_internal: OFF (DAC出力を使用)")
    else:
        print("GVS sine_value_internal: ON")

    print(f"入力パス: {input_path}")
    print(f"ローパスフィルタ周波数: {cutoff_freq}Hz")
    print()

    # integrated_analysis.csvファイルを検索
    analysis_files = find_integrated_analysis_files(input_path)

    if not analysis_files:
        print("integrated_analysis.csvまたはintegrated_analysis_scope.csvファイルが見つかりませんでした。")
        print("まずanalyze_datasまたはanalyze_scope_dataを実行して統合解析を行ってください。")
        return

    print(f"見つかったファイル数: {len(analysis_files)}")

    # 各ファイルを処理
    success_count = 0
    for filepath in analysis_files:
        try:
            if process_integrated_analysis_file(filepath, cutoff_freq,
                                               enable_audio_0_3hz_filter,
                                               enable_gvs_sine_internal):
                success_count += 1
        except Exception as e:
            print(f"エラー: {filepath} の処理に失敗: {e}")

    print(f"\n{'='*80}")
    print(f"処理完了")
    print(f"総ファイル数: {len(analysis_files)}")
    print(f"成功: {success_count}")
    print(f"失敗: {len(analysis_files) - success_count}")
    print(f"ローパスフィルタ周波数: {cutoff_freq}Hz")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
