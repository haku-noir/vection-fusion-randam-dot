#!/usr/bin/env python3
"""
頭部変位位相相関パターン別振幅解析プログラム (displacement_pattern_analyzer.py)

displacement_visualization_extractor.pyが生成した位相相関データを読み込み、
頭部変位と赤・緑ドットのx座標の位相相関パターン別振幅を計算・可視化する

※ displacement_condition_comparison_analyzer.pyと同じデータソースを使用
   （displacement_phase/correlation_visualizations内のCSVファイル）

機能:
1. 指定フォルダから再帰的にdisplacement_window_correlations.csvファイルを検索
2. 対応する_head_displacement.csvファイルからdisplacement_x_cmを読み込み
3. 位相相関パターン別に区間を分類（排他的条件）:
   - 赤のみ高い: phase_correlation_displacement_red_dot >= 0.5 かつ green < 0.5
   - 緑のみ高い: phase_correlation_displacement_green_dot >= 0.5 かつ red < 0.5
   - 両方高い: 両方 >= 0.5
   - 両方低相関: 両方 < 0.5
4. 各パターンの区間でdisplacement_x_cm振幅平均を計算
5. 被験者別、条件別の結果をCSVファイルに出力
6. 結果をグラフで可視化

使用例:
    python displacement_pattern_analyzer.py /path/to/data
    python displacement_pattern_analyzer.py hatano --output results

前提条件:
    displacement_visualization_extractor.pyを事前に実行して、
    displacement_phase/correlation_visualizations/にCSVファイルを生成しておくこと
"""

import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 相関閾値
CORRELATION_THRESHOLD = 0.5

# 窓相関計算のパラメータ（参照用）
WINDOW_SEC = 10.0  # 窓のサイズ（秒）
SAMPLING_RATE = 60.0  # サンプリング周波数 (Hz)

# データ切り出し設定（参照用 - 実際はdisplacement_visualization_extractor.pyで適用済み）
DATA_START_TIME = 20  # 開始時刻（秒）
DATA_END_TIME = None  # 終了時刻（秒）


def find_sway_files(root_folder):
    """
    指定フォルダから再帰的にdisplacement_window_correlationsファイルを検索
    （displacement_visualization_extractor.pyが生成したファイル）

    Args:
        root_folder (str): 検索対象のルートフォルダパス

    Returns:
        list: 見つかったCSVファイルのパスリスト
    """
    # displacement_phase/correlation_visualizations内のwindow_correlationsファイルを検索
    pattern = os.path.join(root_folder, '**', 'displacement_phase', 'correlation_visualizations', '*_displacement_window_correlations_*.csv')
    files = glob.glob(pattern, recursive=True)

    # normalization_infoファイルを除外
    files = [f for f in files if 'normalization_info' not in f]

    print(f"検索パターン: {pattern}")
    print(f"見つかったファイル数: {len(files)}")

    for file in sorted(files):
        rel_path = os.path.relpath(file, root_folder)
        print(f"  - {rel_path}")

    return sorted(files)


def get_color_condition_from_experiment_log(file_path):
    """
    experiment_logファイルから色条件（red/green）を取得

    Args:
        file_path (str): displacement_window_correlations.csvファイルのパス

    Returns:
        str: 色条件（'red' or 'green' or 'unknown'）
    """
    try:
        # ファイルパス構造: stimulus_type/displacement_phase/correlation_visualizations/session_xxx.csv
        # experiment_logは stimulus_type/ 直下にある
        filename = os.path.basename(file_path)
        session_match = re.search(r'(\d{8}_\d{6})', filename)

        if not session_match:
            return 'unknown'

        session_id = session_match.group(1)

        # correlation_visualizations -> displacement_phase -> stimulus_type
        corr_dir = os.path.dirname(file_path)
        phase_dir = os.path.dirname(corr_dir)
        stimulus_dir = os.path.dirname(phase_dir)

        experiment_log_path = os.path.join(stimulus_dir, f"{session_id}_experiment_log.csv")

        if not os.path.exists(experiment_log_path):
            print(f"    警告: experiment_logが見つかりません - {experiment_log_path}")
            return 'unknown'

        # experiment_logを読み込み
        df_log = pd.read_csv(experiment_log_path)

        # conditionカラムから色条件を取得（最初の行）
        if 'condition' in df_log.columns and len(df_log) > 0:
            condition = df_log['condition'].iloc[0]
            return condition if condition in ['red', 'green'] else 'unknown'
        else:
            print(f"    警告: experiment_logにcondition列がありません")
            return 'unknown'

    except Exception as e:
        print(f"    警告: experiment_log読み込みエラー - {e}")
        return 'unknown'


def extract_metadata_from_path(file_path, root_folder):
    """
    ファイルパスから被験者名、刺激タイプ、セッション情報を抽出し、
    experiment_logから色条件を取得

    Args:
        file_path (str): displacement_window_correlations CSVファイルのパス
        root_folder (str): ルートフォルダパス

    Returns:
        dict: メタデータ情報
    """
    rel_path = os.path.relpath(file_path, root_folder)
    path_parts = rel_path.split(os.sep)

    # 被験者名をルートフォルダ名から取得
    subject = os.path.basename(root_folder.rstrip('/'))

    # パス構造: stimulus_type/displacement_phase/correlation_visualizations/session_xxx.csv
    # stimulus_type: vis, audio, gvs など
    stimulus_type = path_parts[0] if len(path_parts) > 0 else 'unknown'

    # ファイル名からセッションIDを抽出
    filename = os.path.basename(file_path)
    session_match = re.search(r'(\d{8}_\d{6})', filename)
    session_id = session_match.group(1) if session_match else 'unknown'

    # experiment_logから色条件を取得
    color_condition = get_color_condition_from_experiment_log(file_path)

    return {
        'subject': subject,
        'stimulus_type': stimulus_type,  # vis, audio, gvs
        'color_condition': color_condition,  # red, green
        'session_id': session_id,
        'file_path': file_path,
        'relative_path': rel_path
    }


def load_and_merge_displacement_data(correlation_filepath, cutoff_freq=3.0):
    """
    displacement_window_correlationsファイルと対応するhead_displacementファイルを読み込み、マージ
    （displacement_visualization_extractor.pyが生成したファイルを使用）

    Args:
        correlation_filepath (str): displacement_window_correlations.csvファイルのパス
        cutoff_freq (float): カットオフ周波数 (Hz)

    Returns:
        pd.DataFrame: マージ済みデータフレーム
    """
    try:
        # window_correlationsデータ読み込み
        df_corr = pd.read_csv(correlation_filepath)
        print(f"    - 位相相関データ: {len(df_corr)} samples")

        # 対応するhead_displacementファイルを検索
        corr_dir = os.path.dirname(correlation_filepath)
        corr_basename = os.path.basename(correlation_filepath)

        # セッションIDを抽出
        session_match = re.search(r'(\d{8}_\d{6})', corr_basename)
        if session_match:
            session_id = session_match.group(1)
        else:
            session_id = None
            print(f"    警告: セッションIDを抽出できません")
            return None

        # カットオフ周波数を抽出
        freq_match = re.search(r'_(\d+\.?\d*)Hz\.csv', corr_basename)
        if freq_match:
            file_cutoff = freq_match.group(1)
        else:
            file_cutoff = str(cutoff_freq)

        # 親ディレクトリ（stimulus_type/）に移動してhead_displacementファイルを検索
        # correlation_visualizations -> displacement_phase -> stimulus_type
        parent_dir = os.path.dirname(os.path.dirname(corr_dir))

        # head_displacementファイルを検索
        displacement_pattern = f"{session_id}*sway*{file_cutoff}*head_displacement.csv"
        displacement_files = glob.glob(os.path.join(parent_dir, displacement_pattern))

        df = df_corr.copy()

        if displacement_files:
            displacement_path = displacement_files[0]
            df_displacement = pd.read_csv(displacement_path)
            print(f"    - 変位データ: {len(df_displacement)} samples ({os.path.basename(displacement_path)})")

            # psychopy_timeでマージ
            # 同じ長さの場合はインデックスベースでマージ
            if len(df_corr) == len(df_displacement):
                print(f"    - 同一長さのためインデックスベースでマージ")
                for col in ['displacement_x_cm', 'displacement_y_cm', 'displacement_x_relative_cm', 'displacement_y_relative_cm']:
                    if col in df_displacement.columns:
                        df[col] = df_displacement[col].values
            else:
                # 長さが異なる場合はmerge_asofで近似マージ
                df_corr_sorted = df_corr.sort_values('psychopy_time').reset_index(drop=True)
                df_disp_sorted = df_displacement.sort_values('psychopy_time').reset_index(drop=True)

                df = pd.merge_asof(
                    df_corr_sorted, 
                    df_disp_sorted[['psychopy_time', 'displacement_x_cm', 'displacement_y_cm', 
                                    'displacement_x_relative_cm', 'displacement_y_relative_cm']],
                    on='psychopy_time',
                    direction='nearest',
                    tolerance=0.01  # 10ms以内の誤差を許容
                )

            print(f"    - マージ後: {len(df)} samples, displacement_x_cm有効: {df['displacement_x_cm'].notna().sum()}")
        else:
            print(f"    - 警告: 変位データが見つかりません: {displacement_pattern}")
            return None

        return df

    except Exception as e:
        print(f"    エラー: データ読み込み失敗 - {e}")
        import traceback
        traceback.print_exc()
        return None


def classify_correlation_patterns(df, threshold=CORRELATION_THRESHOLD):
    """
    窓位相相関データから相関パターンを分類
    （displacement_visualization_extractor.pyが計算した位相相関を使用）

    Args:
        df (pd.DataFrame): マージ済みデータフレーム
        threshold (float): 相関係数の閾値

    Returns:
        dict: パターン別のマスク配列
    """
    # displacement_x_cm列の存在確認
    if 'displacement_x_cm' not in df.columns:
        print(f"    警告: displacement_x_cm列が見つかりません")
        return None

    displacement = df['displacement_x_cm'].values

    # NaN値のチェック
    valid_displacement = ~np.isnan(displacement)
    if valid_displacement.sum() < 100:
        print(f"    警告: displacement_x_cmの有効サンプル数が少なすぎます ({valid_displacement.sum()})")
        return None

    # 位相相関列の存在確認（displacement_visualization_extractor.pyが計算済み）
    required_cols = ['phase_correlation_displacement_red_dot', 'phase_correlation_displacement_green_dot']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"    警告: 必要な位相相関列が見つかりません - {missing_cols}")
        return None

    red_corr = df['phase_correlation_displacement_red_dot'].values
    green_corr = df['phase_correlation_displacement_green_dot'].values
    print(f"    既存の位相相関データを使用（displacement_visualization_extractor.py出力）")

    # 有効なデータのマスク（NaNでない）
    valid_mask = (
        np.isfinite(red_corr) & 
        np.isfinite(green_corr) & 
        np.isfinite(displacement)
    )

    print(f"    有効サンプル数: {valid_mask.sum()}")

    # パターン分類（排他的条件 - displacement_condition_comparison_analyzer.pyと同じ）
    # 赤のみ高い: 赤 >= threshold かつ 緑 < threshold
    red_only_high = valid_mask & (red_corr >= threshold) & (green_corr < threshold)

    # 緑のみ高い: 緑 >= threshold かつ 赤 < threshold
    green_only_high = valid_mask & (green_corr >= threshold) & (red_corr < threshold)

    # 両方高い: 両方 >= threshold
    both_high = valid_mask & (red_corr >= threshold) & (green_corr >= threshold)

    # 両方低相関: 両方 < threshold
    both_low = valid_mask & (red_corr < threshold) & (green_corr < threshold)

    patterns = {
        'red_dominant': red_only_high,  # 排他的: 赤のみ高い
        'green_dominant': green_only_high,  # 排他的: 緑のみ高い
        'both_high': both_high,  # 両方高い
        'both_low': both_low,
        'valid_mask': valid_mask
    }

    # デバッグ出力
    print(f"    パターン分類: 赤のみ={red_only_high.sum()}, 緑のみ={green_only_high.sum()}, 両方高={both_high.sum()}, 両方低={both_low.sum()}")

    return patterns


def calculate_pattern_amplitudes(df, patterns):
    """
    各パターンの区間でdisplacement_x_cm振幅平均を計算

    Args:
        df (pd.DataFrame): マージ済みデータフレーム
        patterns (dict): パターン別のマスク配列

    Returns:
        dict: パターン別の振幅統計
    """
    if patterns is None:
        return {
            'red_dominant_mean': np.nan,
            'red_dominant_std': np.nan,
            'red_dominant_samples': 0,
            'green_dominant_mean': np.nan,
            'green_dominant_std': np.nan,
            'green_dominant_samples': 0,
            'both_high_mean': np.nan,
            'both_high_std': np.nan,
            'both_high_samples': 0,
            'both_low_mean': np.nan,
            'both_low_std': np.nan,
            'both_low_samples': 0,
            'total_valid_samples': 0
        }

    displacement = df['displacement_x_cm'].values
    results = {}

    for pattern_name in ['red_dominant', 'green_dominant', 'both_high', 'both_low']:
        mask = patterns.get(pattern_name, np.zeros(len(displacement), dtype=bool))
        pattern_data = displacement[mask]

        if len(pattern_data) > 0:
            # 振幅（絶対値）の統計
            amplitudes = np.abs(pattern_data)
            results[f'{pattern_name}_mean'] = np.mean(amplitudes)
            results[f'{pattern_name}_std'] = np.std(amplitudes)
            results[f'{pattern_name}_samples'] = len(pattern_data)
        else:
            results[f'{pattern_name}_mean'] = np.nan
            results[f'{pattern_name}_std'] = np.nan
            results[f'{pattern_name}_samples'] = 0

    results['total_valid_samples'] = np.sum(patterns['valid_mask'])

    return results


def analyze_file_by_pattern(file_path, metadata):
    """
    単一ファイルを相関パターン別に解析

    Args:
        file_path (str): CSVファイルのパス
        metadata (dict): ファイルのメタデータ

    Returns:
        dict: 解析結果
    """
    try:
        # データの読み込みとマージ
        df = load_and_merge_displacement_data(file_path)

        if df is None:
            print(f"    スキップ: データ読み込み失敗 - {os.path.basename(file_path)}")
            return None

        # パターン分類
        patterns = classify_correlation_patterns(df)

        if patterns is None:
            print(f"    スキップ: パターン分類不可 - {os.path.basename(file_path)}")
            return None

        # パターン別振幅計算
        amplitude_stats = calculate_pattern_amplitudes(df, patterns)

        # 結果をマージ
        result = {**metadata, **amplitude_stats}

        # サマリー表示
        print(f"    {os.path.basename(file_path)}:")
        print(f"      刺激タイプ: {metadata['stimulus_type']}, 色条件: {metadata['color_condition']}")
        print(f"      赤ドット優勢: 平均振幅={amplitude_stats['red_dominant_mean']:.3f}cm, "
              f"サンプル数={amplitude_stats['red_dominant_samples']}")
        print(f"      緑ドット優勢: 平均振幅={amplitude_stats['green_dominant_mean']:.3f}cm, "
              f"サンプル数={amplitude_stats['green_dominant_samples']}")
        print(f"      両方低相関: 平均振幅={amplitude_stats['both_low_mean']:.3f}cm, "
              f"サンプル数={amplitude_stats['both_low_samples']}")

        return result

    except Exception as e:
        print(f"    エラー: {file_path} の処理に失敗 - {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_all_files(files, root_folder):
    """
    全ファイルを解析して相関パターン別の振幅を計算

    Args:
        files (list): CSVファイルのパスリスト
        root_folder (str): ルートフォルダパス

    Returns:
        list: 解析結果のリスト
    """
    results = []

    print(f"\n{'='*80}")
    print("相関パターン別 頭部変位(displacement_x_cm)振幅解析")
    print(f"相関閾値: {CORRELATION_THRESHOLD}")
    print(f"{'='*80}")

    for file_path in files:
        print(f"\n処理中: {os.path.relpath(file_path, root_folder)}")

        # メタデータを抽出
        metadata = extract_metadata_from_path(file_path, root_folder)

        # パターン別解析
        result = analyze_file_by_pattern(file_path, metadata)

        if result is not None:
            results.append(result)

    return results


def create_summary_statistics(results):
    """
    被験者別・刺激タイプ別・色条件別・パターン別の要約統計を作成

    Args:
        results (list): 解析結果のリスト

    Returns:
        pandas.DataFrame: 要約統計のデータフレーム
    """
    df = pd.DataFrame(results)

    if len(df) == 0:
        print("警告: 有効なデータが見つかりません")
        return pd.DataFrame()

    # 被験者別・刺激タイプ別・色条件別の要約統計
    summary = df.groupby(['subject', 'stimulus_type', 'color_condition']).agg({
        'red_dominant_mean': ['mean', 'std', 'count'],
        'red_dominant_samples': 'sum',
        'green_dominant_mean': ['mean', 'std'],
        'green_dominant_samples': 'sum',
        'both_high_mean': ['mean', 'std'],
        'both_high_samples': 'sum',
        'both_low_mean': ['mean', 'std'],
        'both_low_samples': 'sum',
        'total_valid_samples': 'sum'
    }).round(3)

    # 列名を平坦化
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    return summary


def save_results(results, summary, output_dir):
    """
    結果をCSVファイルに保存

    Args:
        results (list): 詳細解析結果
        summary (pandas.DataFrame): 要約統計
        output_dir (str): 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # 詳細結果の保存
    df_detailed = pd.DataFrame(results)
    detailed_file = os.path.join(output_dir, 'displacement_pattern_detailed.csv')
    df_detailed.to_csv(detailed_file, index=False, encoding='utf-8-sig')
    print(f"\n詳細結果を保存: {detailed_file}")

    # 要約統計の保存
    if not summary.empty:
        summary_file = os.path.join(output_dir, 'displacement_pattern_summary.csv')
        summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"要約統計を保存: {summary_file}")


def create_stacked_bar_visualization(summary, output_dir):
    """
    被験者別・刺激タイプ別・色条件別の積み上げ棒グラフを作成
    各パターンの割合（%）と平均振幅を表示

    Args:
        summary (pandas.DataFrame): 要約統計のデータフレーム
        output_dir (str): 出力ディレクトリ
    """
    if summary.empty:
        print("警告: 可視化するデータがありません")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 24

    # 被験者ごとに図を作成
    subjects = summary['subject'].unique()

    for subject in subjects:
        subject_data = summary[summary['subject'] == subject]

        # 刺激タイプと色条件の組み合わせを作成
        stimulus_types = sorted(subject_data['stimulus_type'].unique())
        color_conditions = sorted(subject_data['color_condition'].unique())

        fig, ax = plt.subplots(figsize=(10, 10))

        # バーの位置とラベル
        x_positions = []
        x_labels = []
        bar_idx = 0

        # データの準備
        red_dominant_data = []
        green_dominant_data = []
        both_low_data = []
        red_amplitude_data = []  # 赤ドット優勢の平均振幅
        green_amplitude_data = []  # 緑ドット優勢の平均振幅
        both_amplitude_data = []  # 両方低相関の平均振幅

        # 表示順序を定義：Gvs green, vis, Gvs red, Audio green, vis, Audio red
        display_order = [
            ('gvs', 'green'),
            ('vis', None),  # visは色条件なし
            ('gvs', 'red'),
            ('audio', 'green'),
            ('vis', None),  # visは色条件なし（2回目）
            ('audio', 'red')
        ]

        # 表示順序に従ってデータを処理
        for stim, color in display_order:
            # visタイプの場合は色条件をまとめる
            if stim == 'vis':
                # vis全体のデータを集約
                vis_data = subject_data[subject_data['stimulus_type'] == 'vis']

                if len(vis_data) > 0:
                    # 各パターンのサンプル数と振幅の加重平均を計算
                    total_samples = vis_data['total_valid_samples_sum'].sum()
                    red_samples = vis_data['red_dominant_samples_sum'].sum()
                    green_samples = vis_data['green_dominant_samples_sum'].sum()
                    both_samples = vis_data['both_low_samples_sum'].sum()

                    if total_samples > 0:
                        red_pct = (red_samples / total_samples) * 100
                        green_pct = (green_samples / total_samples) * 100
                        both_pct = (both_samples / total_samples) * 100

                        red_dominant_data.append(red_pct)
                        green_dominant_data.append(green_pct)
                        both_low_data.append(both_pct)

                        # 加重平均振幅を計算
                        red_mean_weighted = 0
                        green_mean_weighted = 0
                        both_mean_weighted = 0
                        red_count = 0
                        green_count = 0
                        both_count = 0

                        for _, row in vis_data.iterrows():
                            if not pd.isna(row['red_dominant_mean_mean']) and row['red_dominant_samples_sum'] > 0:
                                red_mean_weighted += row['red_dominant_mean_mean'] * row['red_dominant_samples_sum']
                                red_count += row['red_dominant_samples_sum']
                            if not pd.isna(row['green_dominant_mean_mean']) and row['green_dominant_samples_sum'] > 0:
                                green_mean_weighted += row['green_dominant_mean_mean'] * row['green_dominant_samples_sum']
                                green_count += row['green_dominant_samples_sum']
                            if not pd.isna(row['both_low_mean_mean']) and row['both_low_samples_sum'] > 0:
                                both_mean_weighted += row['both_low_mean_mean'] * row['both_low_samples_sum']
                                both_count += row['both_low_samples_sum']

                        red_amplitude_data.append(red_mean_weighted / red_count if red_count > 0 else np.nan)
                        green_amplitude_data.append(green_mean_weighted / green_count if green_count > 0 else np.nan)
                        both_amplitude_data.append(both_mean_weighted / both_count if both_count > 0 else np.nan)
                    else:
                        red_dominant_data.append(0)
                        green_dominant_data.append(0)
                        both_low_data.append(0)
                        red_amplitude_data.append(np.nan)
                        green_amplitude_data.append(np.nan)
                        both_amplitude_data.append(np.nan)
                else:
                    red_dominant_data.append(0)
                    green_dominant_data.append(0)
                    both_low_data.append(0)
                    red_amplitude_data.append(np.nan)
                    green_amplitude_data.append(np.nan)
                    both_amplitude_data.append(np.nan)

                x_positions.append(bar_idx)
                x_labels.append(f"{stim}")
                bar_idx += 1
            else:
                # vis以外は色条件ごとに処理
                group_data = subject_data[
                    (subject_data['stimulus_type'] == stim) & 
                    (subject_data['color_condition'] == color)
                ]

                if len(group_data) > 0:
                    row = group_data.iloc[0]
                    total_samples = row['total_valid_samples_sum']

                    if total_samples > 0:
                        # 各パターンのサンプル数と割合を計算
                        red_samples = row['red_dominant_samples_sum']
                        green_samples = row['green_dominant_samples_sum']
                        both_samples = row['both_low_samples_sum']

                        red_pct = (red_samples / total_samples) * 100
                        green_pct = (green_samples / total_samples) * 100
                        both_pct = (both_samples / total_samples) * 100

                        red_dominant_data.append(red_pct)
                        green_dominant_data.append(green_pct)
                        both_low_data.append(both_pct)

                        # 各パターンの平均振幅
                        red_mean = row['red_dominant_mean_mean'] if not pd.isna(row['red_dominant_mean_mean']) else np.nan
                        green_mean = row['green_dominant_mean_mean'] if not pd.isna(row['green_dominant_mean_mean']) else np.nan
                        both_mean = row['both_low_mean_mean'] if not pd.isna(row['both_low_mean_mean']) else np.nan

                        red_amplitude_data.append(red_mean)
                        green_amplitude_data.append(green_mean)
                        both_amplitude_data.append(both_mean)
                    else:
                        red_dominant_data.append(0)
                        green_dominant_data.append(0)
                        both_low_data.append(0)
                        red_amplitude_data.append(np.nan)
                        green_amplitude_data.append(np.nan)
                        both_amplitude_data.append(np.nan)
                else:
                    red_dominant_data.append(0)
                    green_dominant_data.append(0)
                    both_low_data.append(0)
                    red_amplitude_data.append(np.nan)
                    green_amplitude_data.append(np.nan)
                    both_amplitude_data.append(np.nan)

                x_positions.append(bar_idx)
                x_labels.append(f"{stim}\n{color}")
                bar_idx += 1

        # 積み上げ棒グラフを作成（順序：下から赤、灰色、緑）
        width = 0.7

        # 色設定
        color_red = '#ff6b6b'
        color_green = '#51cf66'
        color_both = '#808080'  # 灰色に変更

        # 棒グラフ描画（左から：赤、灰色、緑の順）
        # 1. 赤ドット優勢（最左層）
        p1 = ax.barh(x_positions, red_dominant_data, width, 
                    label='赤ドット優勢 (≥0.5)', color=color_red, alpha=0.8)

        # 2. 両方低相関（中間層）
        p2 = ax.barh(x_positions, both_low_data, width, 
                    left=red_dominant_data, 
                    label='両方低相関 (<0.5)', color=color_both, alpha=0.8)

        # 3. 緑ドット優勢（最右層）
        green_left = [r + b for r, b in zip(red_dominant_data, both_low_data)]
        p3 = ax.barh(x_positions, green_dominant_data, width, 
                    left=green_left, 
                    label='緑ドット優勢 (≥0.5)', color=color_green, alpha=0.8)

        # グラフの設定
        ax.set_title(f'被験者: {subject} - 頭部変位相関パターン別割合と平均振幅\n（閾値={CORRELATION_THRESHOLD}）', 
                    fontsize=14, fontweight='bold')
        ax.set_yticklabels(["", "", "", ""], rotation=0, ha='right')
        ax.set_xlim(0, 100)
        ax.set_xticklabels([0, 20, 40, 60, 80, 100], fontsize=24)
        ax.grid(True, alpha=0.3, axis='x')

        # 各セグメント内にパーセンテージと平均振幅を表示
        for i, y in enumerate(x_positions):
            # 赤ドット優勢（最左層）
            if red_dominant_data[i] > 1.0:  # 1%以上の場合のみ表示
                x_pos = red_dominant_data[i] / 2
                pct_text = f'{red_dominant_data[i]:.1f}[%]'
                # 平均振幅を追加（NaNでない場合）- 単位をcmに変更
                if not np.isnan(red_amplitude_data[i]):
                    pct_text += f'\n{red_amplitude_data[i]:.2f}[cm]'

                text_color = 'black'
                ax.text(x_pos, y, pct_text, 
                       ha='center', va='center', fontsize=30, color=text_color, fontweight='bold')

            # 両方低相関（中間層）
            if both_low_data[i] > 1.0:  # 1%以上の場合のみ表示
                x_pos = red_dominant_data[i] + both_low_data[i] / 2
                pct_text = f'{both_low_data[i]:.1f}[%]'
                # 平均振幅を追加（NaNでない場合）- 単位をcmに変更
                if not np.isnan(both_amplitude_data[i]):
                    pct_text += f'\n{both_amplitude_data[i]:.2f}[cm]'

                text_color = 'black'
                ax.text(x_pos, y, pct_text, 
                       ha='center', va='center', fontsize=30, color=text_color, fontweight='bold')

            # 緑ドット優勢（最右層）
            if green_dominant_data[i] > 1.0:  # 1%以上の場合のみ表示
                x_pos = green_left[i] + green_dominant_data[i] / 2
                pct_text = f'{green_dominant_data[i]:.1f}[%]'
                # 平均振幅を追加（NaNでない場合）- 単位をcmに変更
                if not np.isnan(green_amplitude_data[i]):
                    pct_text += f'\n{green_amplitude_data[i]:.2f}[cm]'

                text_color = 'black'
                ax.text(x_pos, y, pct_text, 
                       ha='center', va='center', fontsize=30, color=text_color, fontweight='bold')

        plt.tight_layout()

        # グラフを保存
        output_file = os.path.join(output_dir, f'displacement_pattern_percentage_stacked_{subject}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"積み上げ棒グラフを保存: {output_file}")
        plt.close(fig)


def create_visualization(results, summary, output_dir):
    """
    結果を可視化（新しい積み上げグラフを作成）

    Args:
        results (list): 詳細解析結果
        summary (pandas.DataFrame): 要約統計
        output_dir (str): 出力ディレクトリ
    """
    if summary.empty:
        return

    # 新しい積み上げ棒グラフを作成（標準）
    create_stacked_bar_visualization(summary, output_dir)
    
    # 新しい積み上げ棒グラフを作成（Mixed条件）
    create_stacked_bar_visualization_mixed(summary, output_dir)
    
    # 新しい積み上げ棒グラフを作成（Singles条件）
    create_stacked_bar_visualization_singles(summary, output_dir)
    
    # 新しい積み上げ棒グラフを作成（Audio Expanded条件）
    create_stacked_bar_visualization_audio_expanded(summary, output_dir)

    # 新しい積み上げ棒グラフを作成（GVS Expanded条件）
    create_stacked_bar_visualization_gvs_expanded(summary, output_dir)

    # 従来のグラフも作成（参考用）
    create_legacy_visualization(results, summary, output_dir)


def create_stacked_bar_visualization_mixed(summary, output_dir):
    """
    Mixed条件（vis+aud, aud only, vis+ves, ves only）の積み上げ棒グラフを作成
    上から: 
    - vis+aud(red)
    - aud only
    - vis+aud(green)
    - vis+ves(red)
    - ves only
    - vis+ves(green)
    """
    if summary.empty:
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 24

    subjects = summary['subject'].unique()

    for subject in subjects:
        subject_data = summary[summary['subject'] == subject]
        fig, ax = plt.subplots(figsize=(10, 10))

        # 表示順序 (下から上へ)
        # 1. vis+ves(green) -> 'gvs', 'green'
        # 2. ves only -> 'only_gvs' (aggregated)
        # 3. vis+ves(red) -> 'gvs', 'red'
        # 4. vis+aud(green) -> 'audio', 'green'
        # 5. aud only -> 'only_audio' (aggregated)
        # 6. vis+aud(red) -> 'audio', 'red'
        
        display_order = [
            ('gvs', 'green', 'Vis. + Ves.'),
            ('only_gvs', None, 'Ves. only'),
            ('gvs', 'red', 'Vis. + Ves.'),
            ('audio', 'green', 'Vis. + Aud.'),
            ('only_audio', None, 'Aud. only'),
            ('audio', 'red', 'Vis. + Aud.')
        ]

        # 描画用データ
        y_positions = []
        y_labels = []
        label_colors = []  # ラベルの色を保持

        
        red_dominant_data = []
        green_dominant_data = []
        both_low_data = []
        
        red_amplitude_data = []
        green_amplitude_data = []
        both_amplitude_data = []

        for idx, (stim, color, label) in enumerate(display_order):
            
            y_positions.append(idx)
            y_labels.append(label)
            
            # 色の決定
            if color == 'red':
                label_colors.append('red')
            elif color == 'green':
                label_colors.append('green')
            else:
                label_colors.append('black')

            # データ取得
            if color is None:
                 # Aggregated types (only_audio, only_gvs)
                data_subset = subject_data[subject_data['stimulus_type'] == stim]
                
                # 集約処理
                total_samples_sum = data_subset['total_valid_samples_sum'].sum() if not data_subset.empty else 0
                
                if total_samples_sum > 0:
                     # 加重平均処理
                    total_red_samples = data_subset['red_dominant_samples_sum'].sum()
                    total_green_samples = data_subset['green_dominant_samples_sum'].sum()
                    total_both_samples = data_subset['both_low_samples_sum'].sum()
                    
                    red_pct = (total_red_samples / total_samples_sum) * 100
                    green_pct = (total_green_samples / total_samples_sum) * 100
                    both_pct = (total_both_samples / total_samples_sum) * 100
                    
                    # 振幅加重平均
                    w_red_amp = 0
                    w_green_amp = 0
                    w_both_amp = 0
                    
                    for _, row in data_subset.iterrows():
                        if pd.notna(row['red_dominant_mean_mean']): w_red_amp += row['red_dominant_mean_mean'] * row['red_dominant_samples_sum']
                        if pd.notna(row['green_dominant_mean_mean']): w_green_amp += row['green_dominant_mean_mean'] * row['green_dominant_samples_sum']
                        if pd.notna(row['both_low_mean_mean']): w_both_amp += row['both_low_mean_mean'] * row['both_low_samples_sum']
                    
                    red_amp = w_red_amp / total_red_samples if total_red_samples > 0 else np.nan
                    green_amp = w_green_amp / total_green_samples if total_green_samples > 0 else np.nan
                    both_amp = w_both_amp / total_both_samples if total_both_samples > 0 else np.nan
                    
                    red_dominant_data.append(red_pct)
                    green_dominant_data.append(green_pct)
                    both_low_data.append(both_pct)
                    red_amplitude_data.append(red_amp)
                    green_amplitude_data.append(green_amp)
                    both_amplitude_data.append(both_amp)

                else:
                    # データなし
                    red_dominant_data.append(0)
                    green_dominant_data.append(0)
                    both_low_data.append(0)
                    red_amplitude_data.append(np.nan)
                    green_amplitude_data.append(np.nan)
                    both_amplitude_data.append(np.nan)

            else:
                # Specific color condition
                row = subject_data[(subject_data['stimulus_type'] == stim) & (subject_data['color_condition'] == color)]
                
                if not row.empty:
                    r = row.iloc[0]
                    total_samples = r['total_valid_samples_sum']
                    if total_samples > 0:
                        red_dominant_data.append((r['red_dominant_samples_sum'] / total_samples) * 100)
                        green_dominant_data.append((r['green_dominant_samples_sum'] / total_samples) * 100)
                        both_low_data.append((r['both_low_samples_sum'] / total_samples) * 100)
                        
                        red_amplitude_data.append(r['red_dominant_mean_mean'])
                        green_amplitude_data.append(r['green_dominant_mean_mean'])
                        both_amplitude_data.append(r['both_low_mean_mean'])
                    else:
                         red_dominant_data.append(0); green_dominant_data.append(0); both_low_data.append(0)
                         red_amplitude_data.append(np.nan); green_amplitude_data.append(np.nan); both_amplitude_data.append(np.nan)
                else:
                    red_dominant_data.append(0); green_dominant_data.append(0); both_low_data.append(0)
                    red_amplitude_data.append(np.nan); green_amplitude_data.append(np.nan); both_amplitude_data.append(np.nan)

        # Plotting
        width = 0.7
        color_red = '#ff6b6b'
        color_green = '#51cf66'
        color_both = '#808080'

        p1 = ax.barh(y_positions, red_dominant_data, width, label='赤ドット優勢', color=color_red, alpha=0.8)
        p2 = ax.barh(y_positions, both_low_data, width, left=red_dominant_data, label='両方低相関', color=color_both, alpha=0.8)
        
        green_left = [r + b for r, b in zip(red_dominant_data, both_low_data)]
        p3 = ax.barh(y_positions, green_dominant_data, width, left=green_left, label='緑ドット優勢', color=color_green, alpha=0.8)

        ax.set_title(f'被験者: {subject} - Mixed Condition Comparison', fontsize=14, fontweight='bold')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        
        # y軸ラベルの色を設定
        for i, tick_label in enumerate(ax.get_yticklabels()):
            tick_label.set_color(label_colors[i])
            
        ax.set_xlim(0, 100)
        ax.set_xticklabels([0, 20, 40, 60, 80, 100], fontsize=24)
        ax.grid(True, alpha=0.3, axis='x')

        # Annotations
        for i, y in enumerate(y_positions):
            # Red
            if red_dominant_data[i] > 1.0:
                x_pos = red_dominant_data[i] / 2
                label_text = f'{red_dominant_data[i]:.1f}[%]'
                if not np.isnan(red_amplitude_data[i]):
                     label_text += f'\n{red_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')
            
            # Both
            if both_low_data[i] > 1.0:
                x_pos = red_dominant_data[i] + both_low_data[i] / 2
                label_text = f'{both_low_data[i]:.1f}[%]'
                if not np.isnan(both_amplitude_data[i]):
                     label_text += f'\n{both_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')
                
            # Green
            if green_dominant_data[i] > 1.0:
                x_pos = green_left[i] + green_dominant_data[i] / 2
                label_text = f'{green_dominant_data[i]:.1f}[%]'
                if not np.isnan(green_amplitude_data[i]):
                     label_text += f'\n{green_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'displacement_pattern_percentage_stacked_mixed_{subject}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Mixed条件積み上げ棒グラフを保存: {output_file}")
        plt.close(fig)


def create_stacked_bar_visualization_singles(summary, output_dir):
    """
    Single条件（aud only, vis only, ves only）の積み上げ棒グラフを作成
    上から:
    - aud only
    - vis only
    - ves only
    """
    if summary.empty:
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 24

    subjects = summary['subject'].unique()

    for subject in subjects:
        subject_data = summary[summary['subject'] == subject]
        fig, ax = plt.subplots(figsize=(10, 6)) # 高さ調整

        # 表示順序 (下から上へ)
        # 1. ves only -> 'only_gvs'
        # 2. vis only -> 'vis'
        # 3. aud only -> 'only_audio'
        
        display_order = [
            ('only_gvs', 'Ves. only'),
            ('vis', 'Vis. only'),
            ('only_audio', 'Aud. only')
        ]

        # 描画用データ
        y_positions = []
        y_labels = []
        
        red_dominant_data = []
        green_dominant_data = []
        both_low_data = []
        
        red_amplitude_data = []
        green_amplitude_data = []
        both_amplitude_data = []

        for idx, (stim, label) in enumerate(display_order):
            
            y_positions.append(idx)
            y_labels.append(label)

            # Aggregated types processing
            data_subset = subject_data[subject_data['stimulus_type'] == stim]
            
            # 集約処理
            total_samples_sum = data_subset['total_valid_samples_sum'].sum() if not data_subset.empty else 0
            
            if total_samples_sum > 0:
                    # 加重平均処理
                total_red_samples = data_subset['red_dominant_samples_sum'].sum()
                total_green_samples = data_subset['green_dominant_samples_sum'].sum()
                total_both_samples = data_subset['both_low_samples_sum'].sum()
                
                red_pct = (total_red_samples / total_samples_sum) * 100
                green_pct = (total_green_samples / total_samples_sum) * 100
                both_pct = (total_both_samples / total_samples_sum) * 100
                
                # 振幅加重平均
                w_red_amp = 0
                w_green_amp = 0
                w_both_amp = 0
                
                for _, row in data_subset.iterrows():
                    if pd.notna(row['red_dominant_mean_mean']): w_red_amp += row['red_dominant_mean_mean'] * row['red_dominant_samples_sum']
                    if pd.notna(row['green_dominant_mean_mean']): w_green_amp += row['green_dominant_mean_mean'] * row['green_dominant_samples_sum']
                    if pd.notna(row['both_low_mean_mean']): w_both_amp += row['both_low_mean_mean'] * row['both_low_samples_sum']
                
                red_amp = w_red_amp / total_red_samples if total_red_samples > 0 else np.nan
                green_amp = w_green_amp / total_green_samples if total_green_samples > 0 else np.nan
                both_amp = w_both_amp / total_both_samples if total_both_samples > 0 else np.nan
                
                red_dominant_data.append(red_pct)
                green_dominant_data.append(green_pct)
                both_low_data.append(both_pct)
                red_amplitude_data.append(red_amp)
                green_amplitude_data.append(green_amp)
                both_amplitude_data.append(both_amp)

            else:
                # データなし
                red_dominant_data.append(0)
                green_dominant_data.append(0)
                both_low_data.append(0)
                red_amplitude_data.append(np.nan)
                green_amplitude_data.append(np.nan)
                both_amplitude_data.append(np.nan)

        # Plotting
        width = 0.7
        color_red = '#ff6b6b'
        color_green = '#51cf66'
        color_both = '#808080'

        p1 = ax.barh(y_positions, red_dominant_data, width, label='赤ドット優勢', color=color_red, alpha=0.8)
        p2 = ax.barh(y_positions, both_low_data, width, left=red_dominant_data, label='両方低相関', color=color_both, alpha=0.8)
        
        green_left = [r + b for r, b in zip(red_dominant_data, both_low_data)]
        p3 = ax.barh(y_positions, green_dominant_data, width, left=green_left, label='緑ドット優勢', color=color_green, alpha=0.8)

        ax.set_title(f'被験者: {subject} - Single Condition Comparison', fontsize=14, fontweight='bold')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlim(0, 100)
        ax.set_xticklabels([0, 20, 40, 60, 80, 100], fontsize=24)
        ax.grid(True, alpha=0.3, axis='x')

        # Annotations
        for i, y in enumerate(y_positions):
            # Red
            if red_dominant_data[i] > 1.0:
                x_pos = red_dominant_data[i] / 2
                label_text = f'{red_dominant_data[i]:.1f}[%]'
                if not np.isnan(red_amplitude_data[i]):
                     label_text += f'\n{red_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')
            
            # Both
            if both_low_data[i] > 1.0:
                x_pos = red_dominant_data[i] + both_low_data[i] / 2
                label_text = f'{both_low_data[i]:.1f}[%]'
                if not np.isnan(both_amplitude_data[i]):
                     label_text += f'\n{both_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')
                
            # Green
            if green_dominant_data[i] > 1.0:
                x_pos = green_left[i] + green_dominant_data[i] / 2
                label_text = f'{green_dominant_data[i]:.1f}[%]'
                if not np.isnan(green_amplitude_data[i]):
                     label_text += f'\n{green_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'displacement_pattern_percentage_singles_{subject}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Single条件積み上げ棒グラフを保存: {output_file}")
        plt.close(fig)


def create_stacked_bar_visualization_audio_expanded(summary, output_dir):
    """
    Audio Expanded条件の積み上げ棒グラフを作成
    上から:
    - Aud. only (Text color: Red) -> 'only_audio', 'red'
    - Vis. + Aud. (Text color: Red) -> 'audio', 'red'
    - Vis. only (Text color: Black) -> 'vis', (aggregated)
    - Vis. + Aud. (Text color: Green) -> 'audio', 'green'
    - Aud. only (Text color: Green) -> 'only_audio', 'green'
    """
    if summary.empty:
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 24

    subjects = summary['subject'].unique()

    for subject in subjects:
        subject_data = summary[summary['subject'] == subject]
        fig, ax = plt.subplots(figsize=(10, 8)) 

        display_order = [
            ('only_audio', 'green', 'Aud. only'),
            ('audio', 'green', 'Vis. + Aud.'),
            ('vis', None, 'Vis. only'),
            ('audio', 'red', 'Vis. + Aud.'),
            ('only_audio', 'red', 'Aud. only')
        ]

        # 描画用データ
        y_positions = []
        y_labels = []
        label_colors = []

        red_dominant_data = []
        green_dominant_data = []
        both_low_data = []
        red_amplitude_data = []
        green_amplitude_data = []
        both_amplitude_data = []

        for idx, (stim, color, label) in enumerate(display_order):
            y_positions.append(idx)
            y_labels.append(label)

            # ラベル色
            if color == 'red':
                label_colors.append('red')
            elif color == 'green':
                label_colors.append('green')
            else:
                label_colors.append('black')

            if stim == 'vis':
                # Vis aggregated
                vis_data = subject_data[subject_data['stimulus_type'] == 'vis']
                if not vis_data.empty:
                    # 集約 (加重平均)
                    total_samples = vis_data['total_valid_samples_sum'].sum()
                    if total_samples > 0:
                        total_red = vis_data['red_dominant_samples_sum'].sum()
                        total_green = vis_data['green_dominant_samples_sum'].sum()
                        total_both = vis_data['both_low_samples_sum'].sum()
                        
                        red_dominant_data.append((total_red / total_samples) * 100)
                        green_dominant_data.append((total_green / total_samples) * 100)
                        both_low_data.append((total_both / total_samples) * 100)
                        
                        # 振幅平均
                        w_r_a, w_g_a, w_b_a = 0, 0, 0
                        for _, row in vis_data.iterrows():
                             if pd.notna(row['red_dominant_mean_mean']): w_r_a += row['red_dominant_mean_mean'] * row['red_dominant_samples_sum']
                             if pd.notna(row['green_dominant_mean_mean']): w_g_a += row['green_dominant_mean_mean'] * row['green_dominant_samples_sum']
                             if pd.notna(row['both_low_mean_mean']): w_b_a += row['both_low_mean_mean'] * row['both_low_samples_sum']
                        
                        red_amplitude_data.append(w_r_a/total_red if total_red > 0 else np.nan)
                        green_amplitude_data.append(w_g_a/total_green if total_green > 0 else np.nan)
                        both_amplitude_data.append(w_b_a/total_both if total_both > 0 else np.nan)
                    else:
                        red_dominant_data.append(0); green_dominant_data.append(0); both_low_data.append(0)
                        red_amplitude_data.append(np.nan); green_amplitude_data.append(np.nan); both_amplitude_data.append(np.nan)
                else:
                    red_dominant_data.append(0); green_dominant_data.append(0); both_low_data.append(0)
                    red_amplitude_data.append(np.nan); green_amplitude_data.append(np.nan); both_amplitude_data.append(np.nan)
            
            else:
                # Specific color condition
                row = subject_data[(subject_data['stimulus_type'] == stim) & (subject_data['color_condition'] == color)]
                if not row.empty:
                    r = row.iloc[0]
                    total_samples = r['total_valid_samples_sum']
                    if total_samples > 0:
                        red_dominant_data.append((r['red_dominant_samples_sum'] / total_samples) * 100)
                        green_dominant_data.append((r['green_dominant_samples_sum'] / total_samples) * 100)
                        both_low_data.append((r['both_low_samples_sum'] / total_samples) * 100)
                        red_amplitude_data.append(r['red_dominant_mean_mean'])
                        green_amplitude_data.append(r['green_dominant_mean_mean'])
                        both_amplitude_data.append(r['both_low_mean_mean'])
                    else:
                        red_dominant_data.append(0); green_dominant_data.append(0); both_low_data.append(0)
                        red_amplitude_data.append(np.nan); green_amplitude_data.append(np.nan); both_amplitude_data.append(np.nan)
                else:
                    red_dominant_data.append(0); green_dominant_data.append(0); both_low_data.append(0)
                    red_amplitude_data.append(np.nan); green_amplitude_data.append(np.nan); both_amplitude_data.append(np.nan)

        # Plotting
        width = 0.7
        color_red = '#ff6b6b'
        color_green = '#51cf66'
        color_both = '#808080'

        p1 = ax.barh(y_positions, red_dominant_data, width, label='赤ドット優勢', color=color_red, alpha=0.8)
        p2 = ax.barh(y_positions, both_low_data, width, left=red_dominant_data, label='両方低相関', color=color_both, alpha=0.8)
        green_left = [r + b for r, b in zip(red_dominant_data, both_low_data)]
        p3 = ax.barh(y_positions, green_dominant_data, width, left=green_left, label='緑ドット優勢', color=color_green, alpha=0.8)

        ax.set_title(f'被験者: {subject} - Audio Expanded Comparison', fontsize=14, fontweight='bold')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        
        # y軸ラベルの色を設定
        for i, tick_label in enumerate(ax.get_yticklabels()):
            tick_label.set_color(label_colors[i])

        ax.set_xlim(0, 100)
        ax.set_xticklabels([0, 20, 40, 60, 80, 100], fontsize=24)
        ax.grid(True, alpha=0.3, axis='x')

        # Annotations
        for i, y in enumerate(y_positions):
            if red_dominant_data[i] > 1.0:
                x_pos = red_dominant_data[i] / 2
                label_text = f'{red_dominant_data[i]:.1f}[%]'
                if not np.isnan(red_amplitude_data[i]): label_text += f'\n{red_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')
            
            if both_low_data[i] > 1.0:
                x_pos = red_dominant_data[i] + both_low_data[i] / 2
                label_text = f'{both_low_data[i]:.1f}[%]'
                if not np.isnan(both_amplitude_data[i]): label_text += f'\n{both_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')
                
            if green_dominant_data[i] > 1.0:
                x_pos = green_left[i] + green_dominant_data[i] / 2
                label_text = f'{green_dominant_data[i]:.1f}[%]'
                if not np.isnan(green_amplitude_data[i]): label_text += f'\n{green_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'displacement_pattern_percentage_stacked_audio_expanded_{subject}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Audio Expanded条件積み上げ棒グラフを保存: {output_file}")
        plt.close(fig)


def create_stacked_bar_visualization_gvs_expanded(summary, output_dir):
    """
    GVS Expanded条件の積み上げ棒グラフを作成
    上から:
    - Ves. only (Text color: Red) -> 'only_gvs', 'red'
    - Vis. + Ves. (Text color: Red) -> 'gvs', 'red'
    - Vis. only (Text color: Black) -> 'vis', (aggregated)
    - Vis. + Ves. (Text color: Green) -> 'gvs', 'green'
    - Ves. only (Text color: Green) -> 'only_gvs', 'green'
    """
    if summary.empty:
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 24

    subjects = summary['subject'].unique()

    for subject in subjects:
        subject_data = summary[summary['subject'] == subject]
        fig, ax = plt.subplots(figsize=(10, 8)) 

        display_order = [
            ('only_gvs', 'green', 'Ves. only'),
            ('gvs', 'green', 'Vis. + Ves.'),
            ('vis', None, 'Vis. only'),
            ('gvs', 'red', 'Vis. + Ves.'),
            ('only_gvs', 'red', 'Ves. only')
        ]

        # 描画用データ
        y_positions = []
        y_labels = []
        label_colors = []

        red_dominant_data = []
        green_dominant_data = []
        both_low_data = []
        red_amplitude_data = []
        green_amplitude_data = []
        both_amplitude_data = []

        for idx, (stim, color, label) in enumerate(display_order):
            y_positions.append(idx)
            y_labels.append(label)

            # ラベル色
            if color == 'red':
                label_colors.append('red')
            elif color == 'green':
                label_colors.append('green')
            else:
                label_colors.append('black')

            if stim == 'vis':
                # Vis aggregated
                vis_data = subject_data[subject_data['stimulus_type'] == 'vis']
                if not vis_data.empty:
                    # 集約 (加重平均)
                    total_samples = vis_data['total_valid_samples_sum'].sum()
                    if total_samples > 0:
                        total_red = vis_data['red_dominant_samples_sum'].sum()
                        total_green = vis_data['green_dominant_samples_sum'].sum()
                        total_both = vis_data['both_low_samples_sum'].sum()
                        
                        red_dominant_data.append((total_red / total_samples) * 100)
                        green_dominant_data.append((total_green / total_samples) * 100)
                        both_low_data.append((total_both / total_samples) * 100)
                        
                        # 振幅平均
                        w_r_a, w_g_a, w_b_a = 0, 0, 0
                        for _, row in vis_data.iterrows():
                             if pd.notna(row['red_dominant_mean_mean']): w_r_a += row['red_dominant_mean_mean'] * row['red_dominant_samples_sum']
                             if pd.notna(row['green_dominant_mean_mean']): w_g_a += row['green_dominant_mean_mean'] * row['green_dominant_samples_sum']
                             if pd.notna(row['both_low_mean_mean']): w_b_a += row['both_low_mean_mean'] * row['both_low_samples_sum']
                        
                        red_amplitude_data.append(w_r_a/total_red if total_red > 0 else np.nan)
                        green_amplitude_data.append(w_g_a/total_green if total_green > 0 else np.nan)
                        both_amplitude_data.append(w_b_a/total_both if total_both > 0 else np.nan)
                    else:
                        red_dominant_data.append(0); green_dominant_data.append(0); both_low_data.append(0)
                        red_amplitude_data.append(np.nan); green_amplitude_data.append(np.nan); both_amplitude_data.append(np.nan)
                else:
                    red_dominant_data.append(0); green_dominant_data.append(0); both_low_data.append(0)
                    red_amplitude_data.append(np.nan); green_amplitude_data.append(np.nan); both_amplitude_data.append(np.nan)
            
            else:
                # Specific color condition
                row = subject_data[(subject_data['stimulus_type'] == stim) & (subject_data['color_condition'] == color)]
                if not row.empty:
                    r = row.iloc[0]
                    total_samples = r['total_valid_samples_sum']
                    if total_samples > 0:
                        red_dominant_data.append((r['red_dominant_samples_sum'] / total_samples) * 100)
                        green_dominant_data.append((r['green_dominant_samples_sum'] / total_samples) * 100)
                        both_low_data.append((r['both_low_samples_sum'] / total_samples) * 100)
                        red_amplitude_data.append(r['red_dominant_mean_mean'])
                        green_amplitude_data.append(r['green_dominant_mean_mean'])
                        both_amplitude_data.append(r['both_low_mean_mean'])
                    else:
                        red_dominant_data.append(0); green_dominant_data.append(0); both_low_data.append(0)
                        red_amplitude_data.append(np.nan); green_amplitude_data.append(np.nan); both_amplitude_data.append(np.nan)
                else:
                    red_dominant_data.append(0); green_dominant_data.append(0); both_low_data.append(0)
                    red_amplitude_data.append(np.nan); green_amplitude_data.append(np.nan); both_amplitude_data.append(np.nan)

        # Plotting
        width = 0.7
        color_red = '#ff6b6b'
        color_green = '#51cf66'
        color_both = '#808080'

        p1 = ax.barh(y_positions, red_dominant_data, width, label='赤ドット優勢', color=color_red, alpha=0.8)
        p2 = ax.barh(y_positions, both_low_data, width, left=red_dominant_data, label='両方低相関', color=color_both, alpha=0.8)
        green_left = [r + b for r, b in zip(red_dominant_data, both_low_data)]
        p3 = ax.barh(y_positions, green_dominant_data, width, left=green_left, label='緑ドット優勢', color=color_green, alpha=0.8)

        ax.set_title(f'被験者: {subject} - GVS Expanded Comparison', fontsize=14, fontweight='bold')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        
        # y軸ラベルの色を設定
        for i, tick_label in enumerate(ax.get_yticklabels()):
            tick_label.set_color(label_colors[i])

        ax.set_xlim(0, 100)
        ax.set_xticklabels([0, 20, 40, 60, 80, 100], fontsize=24)
        ax.grid(True, alpha=0.3, axis='x')

        # Annotations
        for i, y in enumerate(y_positions):
            if red_dominant_data[i] > 1.0:
                x_pos = red_dominant_data[i] / 2
                label_text = f'{red_dominant_data[i]:.1f}[%]'
                if not np.isnan(red_amplitude_data[i]): label_text += f'\n{red_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')
            
            if both_low_data[i] > 1.0:
                x_pos = red_dominant_data[i] + both_low_data[i] / 2
                label_text = f'{both_low_data[i]:.1f}[%]'
                if not np.isnan(both_amplitude_data[i]): label_text += f'\n{both_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')
                
            if green_dominant_data[i] > 1.0:
                x_pos = green_left[i] + green_dominant_data[i] / 2
                label_text = f'{green_dominant_data[i]:.1f}[%]'
                if not np.isnan(green_amplitude_data[i]): label_text += f'\n{green_amplitude_data[i]:.2f}[cm]'
                ax.text(x_pos, y, label_text, ha='center', va='center', fontsize=20, color='black', fontweight='bold')

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'displacement_pattern_percentage_stacked_gvs_expanded_{subject}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"GVS Expanded条件積み上げ棒グラフを保存: {output_file}")
        plt.close(fig)


def create_legacy_visualization(results, summary, output_dir):
    """
    従来の可視化（参考用）

    Args:
        results (list): 詳細解析結果
        summary (pandas.DataFrame): 要約統計
        output_dir (str): 出力ディレクトリ
    """
    if summary.empty:
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 10

    # stimulus_typeとcolor_conditionを使用
    stimulus_types = sorted(summary['stimulus_type'].unique())
    color_conditions = sorted(summary['color_condition'].unique())
    subjects = sorted(summary['subject'].unique())

    pattern_names = ['red_dominant', 'green_dominant', 'both_low']
    pattern_labels = ['赤ドット優勢\n(≥0.5)', '緑ドット優勢\n(≥0.5)', '両方低相関\n(<0.5)']
    n_patterns = len(pattern_names)

    # stimulus_typeから表示名への変換マップ
    stim_display_names = {
        'vis': 'Vis. Only',
        'audio': 'Vis.+Aud.',
        'gvs': 'Vis.+Ves.',
        'only_audio': 'Aud. Only',
        'only_gvs': 'Ves. Only',
        'all': 'All'
    }

    # 条件リストを作成（visは平均のみ、他はred/greenに分ける）
    condition_labels = []  # 内部処理用（stim_color形式）
    condition_display_labels = []  # 表示用
    condition_label_colors = []  # 軸ラベルの色
    for stim in stimulus_types:
        display_name = stim_display_names.get(stim, stim)
        if stim == 'vis':
            condition_labels.append('vis')
            condition_display_labels.append(display_name)
            condition_label_colors.append('black')  # 黒
        else:
            condition_labels.append(f'{stim}_red')
            condition_display_labels.append(display_name)
            condition_label_colors.append('#d62728')  # 赤
            condition_labels.append(f'{stim}_green')
            condition_display_labels.append(display_name)
            condition_label_colors.append('#2ca02c')  # 緑

    n_conditions = len(condition_labels)

    # 全データの最大値を計算して縦軸を統一
    y_max = 0
    for subj in subjects:
        for pattern_name in pattern_names:
            for stim in stimulus_types:
                if stim == 'vis':
                    # visは両方の平均
                    vals = []
                    errs = []
                    for color in color_conditions:
                        condition_data = summary[
                            (summary['stimulus_type'] == stim) & 
                            (summary['color_condition'] == color) &
                            (summary['subject'] == subj)
                        ]
                        if len(condition_data) > 0:
                            mean_col = f'{pattern_name}_mean_mean'
                            std_col = f'{pattern_name}_mean_std'
                            val = condition_data[mean_col].values[0]
                            err = condition_data[std_col].values[0] if not pd.isna(condition_data[std_col].values[0]) else 0
                            if not pd.isna(val):
                                vals.append(val)
                                errs.append(err)
                    if vals:
                        avg_val = np.mean(vals)
                        avg_err = np.mean(errs)
                        y_max = max(y_max, avg_val + avg_err)
                else:
                    for color in color_conditions:
                        condition_data = summary[
                            (summary['stimulus_type'] == stim) & 
                            (summary['color_condition'] == color) &
                            (summary['subject'] == subj)
                        ]
                        if len(condition_data) > 0:
                            mean_col = f'{pattern_name}_mean_mean'
                            std_col = f'{pattern_name}_mean_std'
                            val = condition_data[mean_col].values[0]
                            err = condition_data[std_col].values[0] if not pd.isna(condition_data[std_col].values[0]) else 0
                            if not pd.isna(val):
                                y_max = max(y_max, val + err)

    # 余白を追加
    y_max = y_max * 1.1 if y_max > 0 else 1.0

    # 被験者ごとに別のグラフを作成
    # x軸: 条件（vis, audio_red, audio_green, ...）
    # 凡例: パターン（赤ドット優勢、緑ドット優勢、両方低相関）
    pattern_colors = {
        'red_dominant': '#ff6b6b',
        'green_dominant': '#51cf66',
        'both_low': '#868e96'
    }
    pattern_labels_short = ['赤ドット優勢', '緑ドット優勢', '両方低相関']

    for subj in subjects:
        fig, ax = plt.subplots(figsize=(14, 6))

        x_pos = np.arange(n_conditions)
        width = 0.8 / n_patterns

        # 各パターンのバーを描画し、その位置と値を保存
        bar_positions_all = []  # [(x, y, val, pct), ...]

        for pattern_idx, (pattern_name, pattern_label) in enumerate(zip(pattern_names, pattern_labels_short)):
            values = []
            errors = []
            percentages = []  # 時間割合を格納

            for cond_label in condition_labels:
                if cond_label == 'vis':
                    # visは両方の色の平均
                    vals = []
                    errs = []
                    pcts = []
                    for color in color_conditions:
                        condition_data = summary[
                            (summary['stimulus_type'] == 'vis') & 
                            (summary['color_condition'] == color) &
                            (summary['subject'] == subj)
                        ]
                        if len(condition_data) > 0:
                            mean_col = f'{pattern_name}_mean_mean'
                            std_col = f'{pattern_name}_mean_std'
                            samples_col = f'{pattern_name}_samples_sum'
                            total_col = 'total_valid_samples_sum'

                            val = condition_data[mean_col].values[0]
                            err = condition_data[std_col].values[0] if not pd.isna(condition_data[std_col].values[0]) else 0
                            samples = condition_data[samples_col].values[0]
                            total = condition_data[total_col].values[0]

                            if not pd.isna(val):
                                vals.append(val)
                                errs.append(err)
                            if total > 0:
                                pcts.append((samples / total) * 100)
                    if vals:
                        values.append(np.mean(vals))
                        errors.append(np.mean(errs))
                    else:
                        values.append(0)
                        errors.append(0)
                    if pcts:
                        percentages.append(np.mean(pcts))
                    else:
                        percentages.append(0)
                else:
                    # 他の条件はred/greenに分ける
                    stim = cond_label.rsplit('_', 1)[0]
                    color = cond_label.rsplit('_', 1)[1]
                    condition_data = summary[
                        (summary['stimulus_type'] == stim) & 
                        (summary['color_condition'] == color) &
                        (summary['subject'] == subj)
                    ]
                    if len(condition_data) > 0:
                        mean_col = f'{pattern_name}_mean_mean'
                        std_col = f'{pattern_name}_mean_std'
                        samples_col = f'{pattern_name}_samples_sum'
                        total_col = 'total_valid_samples_sum'

                        val = condition_data[mean_col].values[0]
                        err = condition_data[std_col].values[0] if not pd.isna(condition_data[std_col].values[0]) else 0
                        samples = condition_data[samples_col].values[0]
                        total = condition_data[total_col].values[0]

                        values.append(val if not pd.isna(val) else 0)
                        errors.append(err if not pd.isna(err) else 0)
                        if total > 0:
                            percentages.append((samples / total) * 100)
                        else:
                            percentages.append(0)
                    else:
                        values.append(0)
                        errors.append(0)
                        percentages.append(0)

            bar_x = x_pos + pattern_idx * width - width * (n_patterns - 1) / 2
            bars = ax.bar(bar_x, values, width, label=pattern_label, 
                   color=pattern_colors[pattern_name], alpha=0.8, 
                   yerr=errors, capsize=3)

            # バーの上に値を表示するための情報を保存
            for i, (bx, by, err, pct) in enumerate(zip(bar_x, values, errors, percentages)):
                bar_positions_all.append((bx, by, err, pct, pattern_name))

        # 棒グラフに数値を表示
        for bx, by, err, pct, pattern_name in bar_positions_all:
            if by > 0:
                # 時間割合を棒グラフの中心に表示（単位付き）
                ax.text(bx, by / 2, f'{pct:.1f}%', ha='center', va='center', 
                       fontsize=12, fontweight='bold')
                # 平均振幅の数値を棒グラフの上に表示（単位付き）
                text_y = by + err + y_max * 0.02
                ax.text(bx, text_y, f'{by:.2f}cm', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')

        ax.set_xlabel('条件')
        ax.set_ylabel('頭部変位平均振幅 [cm]')
        ax.set_title(f'被験者: {subj} - 相関パターン別 頭部変位平均振幅（閾値={CORRELATION_THRESHOLD}）\n'
                     f'棒グラフ上: 平均振幅[cm]、棒グラフ内: 時間割合[%]　※軸ラベル色: 赤=red条件, 緑=green条件')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(condition_display_labels)
        # 軸ラベルに色を付ける
        for tick_label, color in zip(ax.get_xticklabels(), condition_label_colors):
            tick_label.set_color(color)
            tick_label.set_fontweight('bold')
        ax.set_ylim(0, y_max * 1.15)  # 上部にテキスト表示用の余白を確保
        ax.legend(loc='upper right', title='相関パターン')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # グラフを保存
        output_file = os.path.join(output_dir, f'displacement_pattern_by_subject_{subj}.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"被験者別パターングラフを保存: {output_file}")
        plt.close(fig)

    # 図2: 条件別・パターン別の箱ひげ図
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 8))

    # データを準備
    plot_data = []
    plot_labels = []
    plot_colors = []

    colors_dict = {
        'red_dominant': '#ff6b6b',
        'green_dominant': '#51cf66',
        'both_low': '#94d82d'
    }

    # 刺激タイプと色条件の組み合わせごとに処理
    for stim in sorted(stimulus_types):
        for color in sorted(color_conditions):
            condition_df = df[
                (df['stimulus_type'] == stim) & 
                (df['color_condition'] == color)
            ]

            for pattern_name, pattern_label in zip(pattern_names, pattern_labels):
                mean_col = f'{pattern_name}_mean'
                values = condition_df[mean_col].dropna().values

                if len(values) > 0:
                    plot_data.append(values)
                    plot_labels.append(f'{stim}_{color}\n{pattern_label}')
                    plot_colors.append(colors_dict[pattern_name])

    if len(plot_data) > 0:
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)

        # 色を設定
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 統計情報を表示
        for i, data in enumerate(plot_data):
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)

            stats_text = f"μ={mean_val:.2f}cm\nσ={std_val:.2f}cm\nM={median_val:.2f}cm"

            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(i+1, y_pos, stats_text, ha='center', va='top', 
                    fontsize=20, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_xlabel('条件 - パターン')
        ax.set_ylabel('頭部変位平均振幅 [cm]')
        ax.set_title(f'条件別・パターン別 頭部変位平均振幅分布（閾値={CORRELATION_THRESHOLD}）\n（μ=平均, σ=標準偏差, M=中央値）')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # グラフを保存
        output_file = os.path.join(output_dir, 'displacement_pattern_by_condition.png')
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"条件別パターングラフを保存: {output_file}")
        plt.close(fig)

    # 図3: サンプル数の比較
    fig, ax = plt.subplots(figsize=(12, 8))

    plot_data = []
    plot_labels = []

    # 刺激タイプと色条件の組み合わせごとに処理
    for stim in sorted(stimulus_types):
        for color in sorted(color_conditions):
            condition_summary = summary[
                (summary['stimulus_type'] == stim) & 
                (summary['color_condition'] == color)
            ]

            for pattern_name, pattern_label in zip(pattern_names, pattern_labels):
                samples_col = f'{pattern_name}_samples_sum'
                total_samples = condition_summary[samples_col].sum()

                plot_data.append(total_samples)
                plot_labels.append(f'{stim}_{color}\n{pattern_label}')

    x_pos = np.arange(len(plot_labels))
    bars = ax.bar(x_pos, plot_data, alpha=0.8)

    # 色を設定
    for i, bar in enumerate(bars):
        pattern_idx = i % len(pattern_names)
        bar.set_color(list(colors_dict.values())[pattern_idx])

    # 値を棒の上に表示
    for i, (x, y) in enumerate(zip(x_pos, plot_data)):
        ax.text(x, y, f'{int(y)}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('条件 - パターン')
    ax.set_ylabel('サンプル数')
    ax.set_title(f'相関パターン別サンプル数（閾値={CORRELATION_THRESHOLD}）')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_labels)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # グラフを保存
    output_file = os.path.join(output_dir, 'displacement_pattern_sample_counts.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"サンプル数グラフを保存: {output_file}")
    plt.close(fig)


def main():
    """メイン処理関数"""
    print("相関パターン別 頭部変位(displacement_x_cm)振幅解析プログラム")
    print("=" * 80)

    # コマンドライン引数の処理
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("使用法: python displacement_pattern_analyzer.py <入力フォルダ> [--output <出力フォルダ>]")
        sys.exit(1)

    # 出力ディレクトリの設定
    output_dir = 'displacement_pattern_results'
    if '--output' in sys.argv:
        output_idx = sys.argv.index('--output')
        if output_idx + 1 < len(sys.argv):
            output_dir = sys.argv[output_idx + 1]

    print(f"入力パス: {input_path}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"相関閾値: {CORRELATION_THRESHOLD}")
    print()

    # integrated_analysis_sway.csvファイルを検索
    analysis_files = find_sway_files(input_path)

    if not analysis_files:
        print("エラー: integrated_analysis_sway.csvファイルが見つかりません")
        sys.exit(1)

    # 全ファイルを解析
    results = analyze_all_files(analysis_files, input_path)

    if not results:
        print("エラー: 解析可能なデータが見つかりません")
        sys.exit(1)

    # 要約統計を作成
    summary = create_summary_statistics(results)

    # 結果を保存
    save_results(results, summary, output_dir)

    # 可視化
    create_visualization(results, summary, output_dir)

    print(f"\n{'='*80}")
    print("処理完了")
    print(f"総ファイル数: {len(analysis_files)}")
    print(f"解析成功: {len(results)}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
