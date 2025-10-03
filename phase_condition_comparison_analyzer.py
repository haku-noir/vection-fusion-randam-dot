#!/usr/bin/env python3
"""
位相相関条件間比較解析プログラム (phase_condition_comparison_analyzer.py)

単一被験者フォルダから位相相関データを読み込み、
vis/audio/gvs/all条件間での赤ドット・緑ドットの位相相関係数の占有時間を比較分析・可視化する

機能:
1. 単一被験者フォルダからvis/audio/gvs/all条件のphase/correlation_visualizations内のCSVファイルを検索
2. 全セッションから位相相関係数の占有割合計算（赤≥0.5、緑≥0.5、両方<0.5）
3. vis条件: 全conditionの平均占有時間を計算
4. audio/gvs/all条件: condition別（赤/緑）の平均占有時間を計算
5. 横向き積み上げ棒グラフ作成（赤、非定常状態、緑の順）

使用例:
    python ph        # ALL用のグラフは後で実装
        # fig_all, ax_all = plt.subplots(1, 1, figsize=(10, 6))
        # fig_all.suptitle(f'位相相関条件間比較 - {subject_name} (ALL)', fontsize=16, y=0.95)_condition_comparison_analyzer.py hatano
    python phase_condition_comparison_analyzer.py sugihara --output results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import re
from pathlib import Path
import argparse



def read_normalization_info_csv(csv_file):
    """
    規格化情報CSVファイルを読み込み
    
    Args:
        csv_file (str): 規格化情報CSVファイルのパス
        
    Returns:
        dict: 信号名をキーとした振幅情報辞書
    """
    try:
        if not os.path.exists(csv_file):
            return {}
            
        df = pd.read_csv(csv_file)
        norm_info = {}
        
        for _, row in df.iterrows():
            signal_name = row.get('signal_name', '')
            original_amplitude = row.get('original_mean_amplitude', np.nan)
            
            # 信号名を整理（_changeを除去）
            clean_name = signal_name.replace('_change', '')
            norm_info[clean_name] = {
                'original_mean_amplitude': original_amplitude,
                'scaling_factor': row.get('scaling_factor', np.nan),
                'data_mean': row.get('data_mean', np.nan),
                'target_mean_amplitude': row.get('target_mean_amplitude', np.nan)
            }
            
        return norm_info
    except Exception as e:
        print(f"規格化情報読み込みエラー: {e} for {csv_file}")
        return {}


def find_window_correlation_files(subject_folder, condition, cutoff_freq=3.0):
    """
    被験者フォルダ内の指定条件からwindow_correlationsファイルを検索

    Args:
        subject_folder (str): 被験者フォルダパス
        condition (str): 実験条件 ('vis', 'audio', 'gvs')
        cutoff_freq (float): カットオフ周波数

    Returns:
        tuple: (window_correlationsファイルリスト, 規格化情報辞書リスト)
    """
    try:
        # phase/correlation_visualizationsフォルダのパス
        correlation_path = os.path.join(subject_folder, condition, 'phase', 'correlation_visualizations')

        if not os.path.exists(correlation_path):
            print(f"警告: パスが存在しません - {correlation_path}")
            return []

        # window_correlationsファイルを検索
        pattern = f"*_window_correlations_{cutoff_freq}Hz.csv"
        search_path = os.path.join(correlation_path, pattern)
        files = glob.glob(search_path)
        
        # 対応する規格化情報ファイルを検索・読み込み
        normalization_info_list = []
        for csv_file in sorted(files):
            # 規格化情報CSVファイルのパスを推定
            base_name = os.path.basename(csv_file).replace('_window_correlations_', '_window_correlations_normalization_info_')
            norm_csv_path = os.path.join(correlation_path, base_name)
            
            norm_info = read_normalization_info_csv(norm_csv_path)
            normalization_info_list.append(norm_info)

        print(f"  {condition}: {len(files)}ファイル見つかりました")
        return sorted(files), normalization_info_list

    except Exception as e:
        print(f"エラー: ファイル検索に失敗 - {e}")
        return []


def get_condition_from_filename(csv_file):
    """
    CSVファイル名から実験条件（赤/緑）を推定

    Args:
        csv_file (str): CSVファイルパス

    Returns:
        str: 条件 ('red' または 'green' または 'unknown')
    """
    try:
        # 同じディレクトリのexperiment_logファイルを探す
        base_dir = os.path.dirname(csv_file)
        parent_dir = os.path.dirname(base_dir)  # phase/correlation_visualizationsの親
        parent_dir = os.path.dirname(parent_dir)  # phaseの親

        # セッションIDを抽出
        filename = os.path.basename(csv_file)
        session_match = re.search(r'(\d{8}_\d{6})', filename)
        if not session_match:
            return 'unknown'

        session_id = session_match.group(1)

        # experiment_logファイルを探す
        log_pattern = f"{session_id}_experiment_log.csv"
        log_file = os.path.join(parent_dir, log_pattern)

        if os.path.exists(log_file):
            try:
                df_log = pd.read_csv(log_file)
                if 'condition' in df_log.columns and len(df_log) > 0:
                    condition = df_log['condition'].iloc[0]
                    return condition if condition in ['red', 'green'] else 'unknown'
            except:
                pass

        return 'unknown'

    except Exception as e:
        print(f"条件推定エラー: {e}")
        return 'unknown'


def calculate_average_amplitudes(normalization_info_list):
    """
    規格化情報リストから平均振幅比を計算
    
    Args:
        normalization_info_list (list): 規格化情報辞書のリスト
        
    Returns:
        dict: 信号名をキーとした平均振幅比辞書
    """
    try:
        # 各信号の振幅データを集約
        signal_amplitudes = {}
        
        for norm_info in normalization_info_list:
            for signal_name, info in norm_info.items():
                amplitude = info.get('original_mean_amplitude', np.nan)
                if not np.isnan(amplitude):
                    if signal_name not in signal_amplitudes:
                        signal_amplitudes[signal_name] = []
                    signal_amplitudes[signal_name].append(amplitude)
        
        # 平均を計算
        average_amplitudes = {}
        for signal_name, amplitudes in signal_amplitudes.items():
            if amplitudes:
                average_amplitudes[signal_name] = np.mean(amplitudes)
            else:
                average_amplitudes[signal_name] = np.nan
                
        return average_amplitudes
        
    except Exception as e:
        print(f"平均振幅比計算エラー: {e}")
        return {}


def calculate_phase_correlation_occupancy(csv_file):
    """
    window_correlations CSVファイルから位相相関の占有時間を計算

    Args:
        csv_file (str): window_correlations CSVファイルのパス

    Returns:
        dict: 占有時間の割合 {'red_high': %, 'green_high': %, 'both_low': %}
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(csv_file)

        # 必要な列があるかチェック
        required_cols = ['phase_correlation_angle_red_dot', 'phase_correlation_angle_green_dot']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"警告: 必要な列が見つかりません - {missing_cols} in {os.path.basename(csv_file)}")
            return {'red_high': 0, 'green_high': 0, 'both_low': 0, 'total_samples': 0}

        # データを取得
        red_corr = df['phase_correlation_angle_red_dot'].values
        green_corr = df['phase_correlation_angle_green_dot'].values

        # NaNを除去
        valid_mask = ~(np.isnan(red_corr) | np.isnan(green_corr))
        red_corr_clean = red_corr[valid_mask]
        green_corr_clean = green_corr[valid_mask]

        if len(red_corr_clean) == 0:
            print(f"警告: 有効なデータがありません - {os.path.basename(csv_file)}")
            return {'red_high': 0, 'green_high': 0, 'both_low': 0, 'total_samples': 0}

        total_samples = len(red_corr_clean)

        # 条件別のサンプル数を計算
        red_high = np.sum(red_corr_clean >= 0.5)  # 赤ドット相関 ≥ 0.5
        green_high = np.sum(green_corr_clean >= 0.5)  # 緑ドット相関 ≥ 0.5
        both_high = np.sum((red_corr_clean >= 0.5) & (green_corr_clean >= 0.5))  # 両方 ≥ 0.5
        both_low = np.sum((red_corr_clean < 0.5) & (green_corr_clean < 0.5))  # 両方 < 0.5

        # 排他的な条件で計算
        red_only_high = red_high - both_high  # 赤のみ高い
        green_only_high = green_high - both_high  # 緑のみ高い

        # 割合に変換
        occupancy = {
            'red_only_high': (red_only_high / total_samples) * 100,
            'green_only_high': (green_only_high / total_samples) * 100,
            'both_high': (both_high / total_samples) * 100,
            'both_low': (both_low / total_samples) * 100,
            'total_samples': total_samples
        }

        return occupancy

    except Exception as e:
        print(f"エラー: 占有時間計算に失敗 - {e} for {csv_file}")
        return {'red_only_high': 0, 'green_only_high': 0, 'both_high': 0, 'both_low': 0, 'total_samples': 0}


def process_vis_condition(subject_folder, cutoff_freq=3.0):
    """
    vis条件の占有時間を処理（全conditionの平均）

    Args:
        subject_folder (str): 被験者フォルダパス
        cutoff_freq (float): カットオフ周波数

    Returns:
        tuple: (平均占有時間, 平均振幅比情報)
    """
    print("\nVIS条件の処理:")

    files, normalization_info_list = find_window_correlation_files(subject_folder, 'vis', cutoff_freq)

    if not files:
        print("  警告: visファイルが見つかりません")
        return {'red_only_high': 0, 'green_only_high': 0, 'both_low': 0}, {}

    all_occupancies = []

    for csv_file in files:
        filename = os.path.basename(csv_file)
        occupancy = calculate_phase_correlation_occupancy(csv_file)

        if occupancy['total_samples'] > 0:
            all_occupancies.append(occupancy)
            print(f"    {filename}: 赤{occupancy['red_only_high']:.1f}%, 緑{occupancy['green_only_high']:.1f}%, 非定常{occupancy['both_low']:.1f}%")

    if not all_occupancies:
        print("  有効なデータがありません")
        return {'red_only_high': 0, 'green_only_high': 0, 'both_low': 0}, {}

    # 平均を計算
    avg_occupancy = {
        'red_only_high': np.mean([occ['red_only_high'] for occ in all_occupancies]),
        'green_only_high': np.mean([occ['green_only_high'] for occ in all_occupancies]),
        'both_low': np.mean([occ['both_low'] for occ in all_occupancies])
    }
    
    # 平均振幅比を計算
    avg_amplitudes = calculate_average_amplitudes(normalization_info_list)

    print(f"  平均: 赤{avg_occupancy['red_only_high']:.1f}%, 緑{avg_occupancy['green_only_high']:.1f}%, 非定常{avg_occupancy['both_low']:.1f}%")

    return avg_occupancy, avg_amplitudes


def process_audio_gvs_condition(subject_folder, condition, cutoff_freq=3.0):
    """
    audio/gvs条件の占有時間を処理（condition別の平均）

    Args:
        subject_folder (str): 被験者フォルダパス
        condition (str): 実験条件 ('audio' または 'gvs')
        cutoff_freq (float): カットオフ周波数

    Returns:
        tuple: (condition別平均占有時間, condition別平均振幅比情報)
    """
    print(f"\n{condition.upper()}条件の処理:")

    files, normalization_info_list = find_window_correlation_files(subject_folder, condition, cutoff_freq)

    if not files:
        print(f"  警告: {condition}ファイルが見つかりません")
        return ({'red': {'red_only_high': 0, 'green_only_high': 0, 'both_low': 0},
                'green': {'red_only_high': 0, 'green_only_high': 0, 'both_low': 0}}, 
                {'red': {}, 'green': {}})

    red_occupancies = []
    green_occupancies = []
    red_norm_info = []
    green_norm_info = []

    for i, csv_file in enumerate(files):
        filename = os.path.basename(csv_file)
        file_condition = get_condition_from_filename(csv_file)
        occupancy = calculate_phase_correlation_occupancy(csv_file)
        norm_info = normalization_info_list[i] if i < len(normalization_info_list) else {}

        if occupancy['total_samples'] > 0:
            if file_condition == 'red':
                red_occupancies.append(occupancy)
                red_norm_info.append(norm_info)
                print(f"    {filename} (赤条件): 赤{occupancy['red_only_high']:.1f}%, 緑{occupancy['green_only_high']:.1f}%, 非定常{occupancy['both_low']:.1f}%")
            elif file_condition == 'green':
                green_occupancies.append(occupancy)
                green_norm_info.append(norm_info)
                print(f"    {filename} (緑条件): 赤{occupancy['red_only_high']:.1f}%, 緑{occupancy['green_only_high']:.1f}%, 非定常{occupancy['both_low']:.1f}%")
            else:
                print(f"    {filename} (条件不明): スキップ")

    # 条件別平均を計算
    result = {}
    amplitude_result = {}

    # 赤条件の平均
    if red_occupancies:
        result['red'] = {
            'red_only_high': np.mean([occ['red_only_high'] for occ in red_occupancies]),
            'green_only_high': np.mean([occ['green_only_high'] for occ in red_occupancies]),
            'both_low': np.mean([occ['both_low'] for occ in red_occupancies])
        }
        amplitude_result['red'] = calculate_average_amplitudes(red_norm_info)
        print(f"  赤条件平均: 赤{result['red']['red_only_high']:.1f}%, 緑{result['red']['green_only_high']:.1f}%, 非定常{result['red']['both_low']:.1f}%")
    else:
        result['red'] = {'red_only_high': 0, 'green_only_high': 0, 'both_low': 0}
        amplitude_result['red'] = {}
        print("  赤条件: データなし")

    # 緑条件の平均
    if green_occupancies:
        result['green'] = {
            'red_only_high': np.mean([occ['red_only_high'] for occ in green_occupancies]),
            'green_only_high': np.mean([occ['green_only_high'] for occ in green_occupancies]),
            'both_low': np.mean([occ['both_low'] for occ in green_occupancies])
        }
        amplitude_result['green'] = calculate_average_amplitudes(green_norm_info)
        print(f"  緑条件平均: 赤{result['green']['red_only_high']:.1f}%, 緑{result['green']['green_only_high']:.1f}%, 非定常{result['green']['both_low']:.1f}%")
    else:
        result['green'] = {'red_only_high': 0, 'green_only_high': 0, 'both_low': 0}
        amplitude_result['green'] = {}
        print("  緑条件: データなし")



    return result, amplitude_result


def create_condition_comparison_chart(vis_data, audio_data, gvs_data, all_data, vis_amplitudes, audio_amplitudes, gvs_amplitudes, all_amplitudes, subject_name, output_dir):
    """
    条件間比較の横向き積み上げ棒グラフを作成（2列×3行レイアウト）

    Args:
        vis_data (dict): vis条件の占有時間データ
        audio_data (dict): audio条件の占有時間データ（condition別）
        gvs_data (dict): gvs条件の占有時間データ（condition別）
        vis_amplitudes (dict): vis条件の平均振幅比データ
        audio_amplitudes (dict): audio条件の平均振幅比データ（condition別）
        gvs_amplitudes (dict): gvs条件の平均振幅比データ（condition別）
        subject_name (str): 被験者名
        output_dir (str): 出力ディレクトリ
    """
    try:
        plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
        plt.rcParams["font.size"] = 15

        # 色設定（赤、非定常状態、緑の順）
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        labels = ['赤ドット', '非定常状態', '緑ドット']

        # 条件データをまとめる（占有時間と振幅情報）
        conditions_data = [
            ('赤ドット同期　', audio_data.get('red', {}), gvs_data.get('red', {}), audio_amplitudes.get('red', {}), gvs_amplitudes.get('red', {})),
            ('視覚刺激のみ　', vis_data, vis_data, vis_amplitudes, vis_amplitudes),  # visは同じデータを使用
            ('緑ドット同期　', audio_data.get('green', {}), gvs_data.get('green', {}), audio_amplitudes.get('green', {}), gvs_amplitudes.get('green', {}))
        ]

        # AUDIO用のグラフを作成（1つのサブプロットに3条件）
        fig_audio, ax_audio = plt.subplots(1, 1, figsize=(8, 4))
        # fig_audio.suptitle(f'位相相関条件間比較 - {subject_name} (AUDIO)', fontsize=16, y=0.95)

        # GVS用のグラフを作成（1つのサブプロットに3条件）
        fig_gvs, ax_gvs = plt.subplots(1, 1, figsize=(8, 4))
        # fig_gvs.suptitle(f'位相相関条件間比較 - {subject_name} (GVS)', fontsize=16, y=0.95)

        # ALL用のグラフを作成（１つのサブプロットに３条件）
        fig_all, ax_all = plt.subplots(1, 1, figsize=(8, 4))
        # fig_all.suptitle(f'位相相関条件間比較 - {subject_name} (ALL)', fontsize=16, y=0.95)

        # y位置を設定（3つの条件）
        y_positions = [0, 1, 2]  # 赤刺激条件、視覚のみ条件、緑刺激条件
        condition_labels = []

        for row, (condition_name, audio_occ, gvs_occ, audio_amp, gvs_amp) in enumerate(conditions_data):
            condition_labels.append(condition_name)

            # AUDIOデータ
            audio_red = audio_occ.get('red_only_high', 0)
            audio_nonstationary = audio_occ.get('both_low', 0)
            audio_green = audio_occ.get('green_only_high', 0)

            # ALLデータを取得（条件に応じて）
            if condition_name == '赤ドット同期　':
                all_occ = all_data.get('red', {})
                all_amp = all_amplitudes.get('red', {})
            elif condition_name == '緑ドット同期　':
                all_occ = all_data.get('green', {})
                all_amp = all_amplitudes.get('green', {})
            else:  # 視覚刺激のみ - vis条件と同じデータを使用
                all_occ = vis_data
                all_amp = vis_amplitudes

            all_red = all_occ.get('red_only_high', 0)
            all_nonstationary = all_occ.get('both_low', 0)
            all_green = all_occ.get('green_only_high', 0)

            # AUDIO用横向き積み上げ棒グラフ
            bars1_audio = ax_audio.barh([y_positions[row]], [audio_red], color=colors[0], label=labels[0] if row == 0 else "")
            bars2_audio = ax_audio.barh([y_positions[row]], [audio_nonstationary], left=[audio_red], color=colors[1], label=labels[1] if row == 0 else "")
            bars3_audio = ax_audio.barh([y_positions[row]], [audio_green], left=[audio_red + audio_nonstationary], color=colors[2], label=labels[2] if row == 0 else "")

            # AUDIO用パーセンテージ表示（0.1%以上の場合のみ）
            if audio_red >= 0.1:
                ax_audio.text(audio_red/2, y_positions[row], f'{audio_red:.1f}%', ha='center', va='center', 
                             fontsize=15, fontweight='bold', color='black')
            if audio_nonstationary >= 0.1:
                ax_audio.text(audio_red + audio_nonstationary/2, y_positions[row], f'{audio_nonstationary:.1f}%', ha='center', va='center', 
                             fontsize=15, fontweight='bold', color='black')
            if audio_green >= 0.1:
                ax_audio.text(audio_red + audio_nonstationary + audio_green/2, y_positions[row], f'{audio_green:.1f}%', ha='center', va='center', 
                             fontsize=15, fontweight='bold', color='black')
            
            # AUDIO用振幅情報を右側に表示（angleのみ）
            amplitude_text_audio = []
            for signal_name, amp_value in audio_amp.items():
                if not np.isnan(amp_value) and signal_name == 'angle':
                    amplitude_text_audio.append(f"平均振幅比: {amp_value:.1f}")
            if amplitude_text_audio:
                ax_audio.text(105, y_positions[row], '\n'.join(amplitude_text_audio), ha='left', va='center', 
                             fontsize=12, color='darkblue', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))

            # GVSデータ
            gvs_red = gvs_occ.get('red_only_high', 0)
            gvs_nonstationary = gvs_occ.get('both_low', 0)
            gvs_green = gvs_occ.get('green_only_high', 0)

            # GVS用横向き積み上げ棒グラフ
            bars1_gvs = ax_gvs.barh([y_positions[row]], [gvs_red], color=colors[0], label=labels[0] if row == 0 else "")
            bars2_gvs = ax_gvs.barh([y_positions[row]], [gvs_nonstationary], left=[gvs_red], color=colors[1], label=labels[1] if row == 0 else "")
            bars3_gvs = ax_gvs.barh([y_positions[row]], [gvs_green], left=[gvs_red + gvs_nonstationary], color=colors[2], label=labels[2] if row == 0 else "")

            # GVS用パーセンテージ表示（0.1%以上の場合のみ）
            if gvs_red >= 0.1:
                ax_gvs.text(gvs_red/2, y_positions[row], f'{gvs_red:.1f}%', ha='center', va='center', 
                           fontsize=15, fontweight='bold', color='black')
            if gvs_nonstationary >= 0.1:
                ax_gvs.text(gvs_red + gvs_nonstationary/2, y_positions[row], f'{gvs_nonstationary:.1f}%', ha='center', va='center', 
                           fontsize=15, fontweight='bold', color='black')
            if gvs_green >= 0.1:
                ax_gvs.text(gvs_red + gvs_nonstationary + gvs_green/2, y_positions[row], f'{gvs_green:.1f}%', ha='center', va='center', 
                           fontsize=15, fontweight='bold', color='black')

            # ALL用横向き積み上げ棒グラフ
            bars1_all = ax_all.barh([y_positions[row]], [all_red], color=colors[0], label=labels[0] if row == 0 else "")
            bars2_all = ax_all.barh([y_positions[row]], [all_nonstationary], left=[all_red], color=colors[1], label=labels[1] if row == 0 else "")
            bars3_all = ax_all.barh([y_positions[row]], [all_green], left=[all_red + all_nonstationary], color=colors[2], label=labels[2] if row == 0 else "")

            # ALL用パーセンテージ表示（0.1%以上の場合のみ）
            if all_red >= 0.1:
                ax_all.text(all_red/2, y_positions[row], f'{all_red:.1f}%', ha='center', va='center', 
                           fontsize=15, fontweight='bold', color='black')
            if all_nonstationary >= 0.1:
                ax_all.text(all_red + all_nonstationary/2, y_positions[row], f'{all_nonstationary:.1f}%', ha='center', va='center', 
                           fontsize=15, fontweight='bold', color='black')
            if all_green >= 0.1:
                ax_all.text(all_red + all_nonstationary + all_green/2, y_positions[row], f'{all_green:.1f}%', ha='center', va='center', 
                           fontsize=15, fontweight='bold', color='black')
            
            # GVS用振幅情報を右側に表示（angleのみ）
            amplitude_text_gvs = []
            for signal_name, amp_value in gvs_amp.items():
                if not np.isnan(amp_value) and signal_name == 'angle':
                    amplitude_text_gvs.append(f"平均振幅比: {amp_value:.1f}")
            if amplitude_text_gvs:
                ax_gvs.text(105, y_positions[row], '\n'.join(amplitude_text_gvs), ha='left', va='center', 
                           fontsize=12, color='darkblue', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))

            # ALL用振幅情報を右側に表示（angleのみ）
            amplitude_text_all = []
            for signal_name, amp_value in all_amp.items():
                if not np.isnan(amp_value) and signal_name == 'angle':
                    amplitude_text_all.append(f"平均振幅比: {amp_value:.1f}")
            if amplitude_text_all:
                ax_all.text(105, y_positions[row], '\n'.join(amplitude_text_all), ha='left', va='center', 
                           fontsize=12, color='darkblue', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))

        # AUDIO用軸設定
        ax_audio.set_yticks(y_positions)
        ax_audio.set_yticklabels(condition_labels, fontsize=18)
        ax_audio.set_xlim(0, 150)  # 振幅情報表示用に範囲を延長
        ax_audio.set_xlabel('ベクションが成立した時間割合 [%]', fontsize=18)
        # ax_audio.set_ylabel('実験条件', fontsize=18)
        ax_audio.tick_params(axis='x', labelsize=16)  # x軸のティックラベルサイズ
        ax_audio.tick_params(axis='y', labelsize=18)  # y軸のティックラベルサイズ
        ax_audio.grid(True, alpha=0.3, axis='x')
        # ax_audio.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)

        # GVS用軸設定
        ax_gvs.set_yticks(y_positions)
        ax_gvs.set_yticklabels(condition_labels, fontsize=18)
        ax_gvs.set_xlim(0, 150)  # 振幅情報表示用に範囲を延長
        ax_gvs.set_xlabel('ベクションが成立した時間割合 [%]', fontsize=18)
        # ax_gvs.set_ylabel('実験条件', fontsize=18)
        ax_gvs.tick_params(axis='x', labelsize=16)  # x軸のティックラベルサイズ
        ax_gvs.tick_params(axis='y', labelsize=18)  # y軸のティックラベルサイズ
        ax_gvs.grid(True, alpha=0.3, axis='x')
        # ax_gvs.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)

        # ALL用軸設定
        ax_all.set_yticks(y_positions)
        ax_all.set_yticklabels(condition_labels, fontsize=18)
        ax_all.set_xlim(0, 150)  # 振幅情報表示用に範囲を延長
        ax_all.set_xlabel('ベクションが成立した時間割合 [%]', fontsize=18)
        # ax_all.set_ylabel('実験条件', fontsize=18)
        ax_all.tick_params(axis='x', labelsize=16)  # x軸のティックラベルサイズ
        ax_all.tick_params(axis='y', labelsize=18)  # y軸のティックラベルサイズ
        ax_all.grid(True, alpha=0.3, axis='x')
        # ax_all.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)

        # レイアウト調整
        fig_audio.tight_layout()
        fig_gvs.tight_layout()
        fig_all.tight_layout()

        # AUDIOグラフを保存
        output_file_audio = os.path.join(output_dir, f'phase_condition_comparison_{subject_name}_AUDIO.png')
        fig_audio.savefig(output_file_audio, dpi=300, bbox_inches='tight')
        print(f"\nAUDIOグラフを保存: {os.path.basename(output_file_audio)}")
        plt.close(fig_audio)

        # GVSグラフを保存
        output_file_gvs = os.path.join(output_dir, f'phase_condition_comparison_{subject_name}_GVS.png')
        fig_gvs.savefig(output_file_gvs, dpi=300, bbox_inches='tight')
        print(f"GVSグラフを保存: {os.path.basename(output_file_gvs)}")
        plt.close(fig_gvs)

        # ALLグラフを保存
        output_file_all = os.path.join(output_dir, f'phase_condition_comparison_{subject_name}_ALL.png')
        fig_all.savefig(output_file_all, dpi=300, bbox_inches='tight')
        print(f"ALLグラフを保存: {os.path.basename(output_file_all)}")
        plt.close(fig_all)

    except Exception as e:
        print(f"エラー: グラフ作成に失敗 - {e}")


def save_comparison_csv(vis_data, audio_data, gvs_data, all_data, vis_amplitudes, audio_amplitudes, gvs_amplitudes, all_amplitudes, subject_name, output_dir):
    """
    条件間比較データをCSVファイルに保存

    Args:
        vis_data (dict): vis条件の占有時間データ
        audio_data (dict): audio条件の占有時間データ
        gvs_data (dict): gvs条件の占有時間データ
        all_data (dict): all条件の占有時間データ
        vis_amplitudes (dict): vis条件の平均振幅比データ
        audio_amplitudes (dict): audio条件の平均振幅比データ
        gvs_amplitudes (dict): gvs条件の平均振幅比データ
        all_amplitudes (dict): all条件の平均振幅比データ
        subject_name (str): 被験者名
        output_dir (str): 出力ディレクトリ
    """
    try:
        # データを整理
        data_rows = []

        # VIS条件
        vis_amp_str = ', '.join([f"{k}: {v:.1f}" for k, v in vis_amplitudes.items() if not np.isnan(v) and k == 'angle'])
        data_rows.append({
            'subject': subject_name,
            'stimulus_type': 'AUDIO',
            'condition': 'vis_only',
            'red_only_high_percent': vis_data.get('red_only_high', 0),
            'green_only_high_percent': vis_data.get('green_only_high', 0),
            'both_low_percent': vis_data.get('both_low', 0),
            'average_amplitudes': vis_amp_str
        })

        data_rows.append({
            'subject': subject_name,
            'stimulus_type': 'GVS',
            'condition': 'vis_only',
            'red_only_high_percent': vis_data.get('red_only_high', 0),
            'green_only_high_percent': vis_data.get('green_only_high', 0),
            'both_low_percent': vis_data.get('both_low', 0),
            'average_amplitudes': vis_amp_str
        })

        # AUDIO条件
        for cond in ['red', 'green']:
            audio_occ = audio_data.get(cond, {})
            audio_amp = audio_amplitudes.get(cond, {})
            audio_amp_str = ', '.join([f"{k}: {v:.1f}" for k, v in audio_amp.items() if not np.isnan(v) and k == 'angle'])
            data_rows.append({
                'subject': subject_name,
                'stimulus_type': 'AUDIO',
                'condition': cond,
                'red_only_high_percent': audio_occ.get('red_only_high', 0),
                'green_only_high_percent': audio_occ.get('green_only_high', 0),
                'both_low_percent': audio_occ.get('both_low', 0),
                'average_amplitudes': audio_amp_str
            })

        # GVS条件
        for cond in ['red', 'green']:
            gvs_occ = gvs_data.get(cond, {})
            gvs_amp = gvs_amplitudes.get(cond, {})
            gvs_amp_str = ', '.join([f"{k}: {v:.1f}" for k, v in gvs_amp.items() if not np.isnan(v) and k == 'angle'])
            data_rows.append({
                'subject': subject_name,
                'stimulus_type': 'GVS',
                'condition': cond,
                'red_only_high_percent': gvs_occ.get('red_only_high', 0),
                'green_only_high_percent': gvs_occ.get('green_only_high', 0),
                'both_low_percent': gvs_occ.get('both_low', 0),
                'average_amplitudes': gvs_amp_str
            })

        # ALL条件
        for cond in ['red', 'green']:
            all_occ = all_data.get(cond, {})
            all_amp = all_amplitudes.get(cond, {})
            all_amp_str = ', '.join([f"{k}: {v:.1f}" for k, v in all_amp.items() if not np.isnan(v) and k == 'angle'])
            data_rows.append({
                'subject': subject_name,
                'stimulus_type': 'ALL',
                'condition': cond,
                'red_only_high_percent': all_occ.get('red_only_high', 0),
                'green_only_high_percent': all_occ.get('green_only_high', 0),
                'both_low_percent': all_occ.get('both_low', 0),
                'average_amplitudes': all_amp_str
            })

        # データフレーム作成
        df = pd.DataFrame(data_rows)

        # CSV保存
        output_file = os.path.join(output_dir, f'phase_condition_comparison_{subject_name}.csv')
        df.to_csv(output_file, index=False)

        print(f"CSVを保存: {os.path.basename(output_file)} ({len(df)}行)")

    except Exception as e:
        print(f"エラー: CSV保存に失敗 - {e}")


def main():
    """メイン処理関数"""
    print("位相相関条件間比較解析プログラム")
    print("=" * 60)

    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description='位相相関条件間比較分析')
    parser.add_argument('subject', help='被験者フォルダ名')
    parser.add_argument('--freq', type=float, default=3.0, help='カットオフ周波数（デフォルト: 3.0Hz）')
    parser.add_argument('--output', type=str, default='.', help='出力ディレクトリ（デフォルト: 現在のディレクトリ）')

    args = parser.parse_args()

    subject_folder = args.subject
    cutoff_freq = args.freq
    output_dir = args.output

    subject_name = os.path.basename(subject_folder.rstrip('/'))

    print(f"被験者: {subject_name} ({subject_folder})")
    print(f"カットオフ周波数: {cutoff_freq}Hz")
    print(f"出力ディレクトリ: {output_dir}")
    print()

    # 被験者フォルダの存在確認
    if not os.path.exists(subject_folder):
        print(f"エラー: フォルダが存在しません - {subject_folder}")
        sys.exit(1)

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 各条件を処理
    print("="*60)
    print("条件別データ処理")
    print("="*60)

    # VIS条件の処理
    vis_data, vis_amplitudes = process_vis_condition(subject_folder, cutoff_freq)

    # AUDIO条件の処理
    audio_data, audio_amplitudes = process_audio_gvs_condition(subject_folder, 'audio', cutoff_freq)

    # GVS条件の処理
    gvs_data, gvs_amplitudes = process_audio_gvs_condition(subject_folder, 'gvs', cutoff_freq)

    # ALL条件の処理
    all_data, all_amplitudes = process_audio_gvs_condition(subject_folder, 'all', cutoff_freq)

    # 結果の表示
    print("\n" + "="*60)
    print("処理結果サマリー")
    print("="*60)

    print(f"\nVIS条件:")
    print(f"  赤ドット: {vis_data['red_only_high']:.1f}%")
    print(f"  緑ドット: {vis_data['green_only_high']:.1f}%")
    print(f"  非定常状態: {vis_data['both_low']:.1f}%")

    print(f"\nAUDIO条件:")
    for cond in ['red', 'green']:
        occ = audio_data.get(cond, {})
        print(f"  {cond}条件: 赤{occ.get('red_only_high', 0):.1f}%, 緑{occ.get('green_only_high', 0):.1f}%, 非定常{occ.get('both_low', 0):.1f}%")

    print(f"\nGVS条件:")
    for cond in ['red', 'green']:
        occ = gvs_data.get(cond, {})
        print(f"  {cond}条件: 赤{occ.get('red_only_high', 0):.1f}%, 緑{occ.get('green_only_high', 0):.1f}%, 非定常{occ.get('both_low', 0):.1f}%")

    print(f"\nALL条件:")
    for cond in ['red', 'green']:
        occ = all_data.get(cond, {})
        print(f"  {cond}条件: 赤{occ.get('red_only_high', 0):.1f}%, 緑{occ.get('green_only_high', 0):.1f}%, 非定常{occ.get('both_low', 0):.1f}%")

    # グラフを作成
    create_condition_comparison_chart(vis_data, audio_data, gvs_data, all_data, vis_amplitudes, audio_amplitudes, gvs_amplitudes, all_amplitudes, subject_name, output_dir)

    # CSVファイルを保存
    save_comparison_csv(vis_data, audio_data, gvs_data, all_data, vis_amplitudes, audio_amplitudes, gvs_amplitudes, all_amplitudes, subject_name, output_dir)

    print(f"\n{'='*60}")
    print("処理完了")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"生成されたファイル:")
    print(f"  - phase_condition_comparison_{subject_name}_AUDIO.png")
    print(f"  - phase_condition_comparison_{subject_name}_GVS.png")
    print(f"  - phase_condition_comparison_{subject_name}_ALL.png")
    print(f"  - phase_condition_comparison_{subject_name}.csv")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
