#!/usr/bin/env python3
"""
angle_change平均振幅解析プログラム (angle_change_amplitude_analyzer.py)

指定フォルダから再帰的にintegrated_analysis.csvファイルを読み込み、
angle_changeの平均振幅を計算・可視化する

機能:
1. 指定フォルダから再帰的にintegrated_analysis.csvファイルを検索
2. 各ファイルからangle_changeデータを抽出
3. 平均振幅（絶対値の平均）を計算
4. 被験者別、条件別の結果をCSVファイルに出力
5. 結果をグラフで可視化

使用例:
    python angle_change_amplitude_analyzer.py /path/to/data
    python angle_change_amplitude_analyzer.py hatano --output results
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


def find_integrated_analysis_files(root_folder):
    """
    指定フォルダから再帰的にintegrated_analysis.csvファイルを検索

    Args:
        root_folder (str): 検索対象のルートフォルダパス

    Returns:
        list: 見つかったCSVファイルのパスリスト
    """
    pattern = os.path.join(root_folder, '**', '*integrated_analysis.csv')
    files = glob.glob(pattern, recursive=True)

    print(f"検索パターン: {pattern}")
    print(f"見つかったファイル数: {len(files)}")

    for file in sorted(files):
        rel_path = os.path.relpath(file, root_folder)
        print(f"  - {rel_path}")

    return sorted(files)


def extract_metadata_from_path(file_path, root_folder):
    """
    ファイルパスから被験者名、条件、セッション情報を抽出

    Args:
        file_path (str): CSVファイルのパス
        root_folder (str): ルートフォルダパス

    Returns:
        dict: メタデータ情報
    """
    rel_path = os.path.relpath(file_path, root_folder)
    path_parts = rel_path.split(os.sep)

    # 被験者名をルートフォルダ名から取得
    subject = os.path.basename(root_folder.rstrip('/'))

    # パス構造: condition/session_integrated_analysis.csv
    condition = path_parts[0] if len(path_parts) > 0 else 'unknown'

    # ファイル名からセッションIDを抽出
    filename = os.path.basename(file_path)
    session_match = re.search(r'(\d{8}_\d{6})', filename)
    session_id = session_match.group(1) if session_match else 'unknown'

    return {
        'subject': subject,
        'condition': condition,
        'session_id': session_id,
        'file_path': file_path,
        'relative_path': rel_path
    }


def calculate_angle_change_amplitude(csv_file):
    """
    CSVファイルからangle_changeの平均振幅を計算

    Args:
        csv_file (str): CSVファイルのパス

    Returns:
        dict: 振幅統計情報
    """
    try:
        df = pd.read_csv(csv_file)

        # angle_change列の存在確認
        if 'angle_change' not in df.columns:
            print(f"  警告: angle_change列が見つかりません - {os.path.basename(csv_file)}")
            return {
                'mean_amplitude': np.nan,
                'std_amplitude': np.nan,
                'max_amplitude': np.nan,
                'min_amplitude': np.nan,
                'total_samples': 0,
                'valid_samples': 0
            }

        angle_change = df['angle_change'].values

        # NaNや無限値を除外
        valid_mask = np.isfinite(angle_change)
        angle_change_clean = angle_change[valid_mask]

        if len(angle_change_clean) == 0:
            print(f"  警告: 有効なangle_changeデータがありません - {os.path.basename(csv_file)}")
            return {
                'mean_amplitude': np.nan,
                'std_amplitude': np.nan,
                'max_amplitude': np.nan,
                'min_amplitude': np.nan,
                'total_samples': len(angle_change),
                'valid_samples': 0
            }

        # 振幅統計を計算（絶対値の統計）
        amplitudes = np.abs(angle_change_clean)

        stats = {
            'mean_amplitude': np.mean(amplitudes),
            'std_amplitude': np.std(amplitudes),
            'max_amplitude': np.max(amplitudes),
            'min_amplitude': np.min(amplitudes),
            'total_samples': len(angle_change),
            'valid_samples': len(angle_change_clean)
        }

        print(f"  {os.path.basename(csv_file)}: 平均振幅={stats['mean_amplitude']:.3f}°, サンプル数={stats['valid_samples']}")

        return stats

    except Exception as e:
        print(f"  エラー: {csv_file} の処理に失敗 - {e}")
        return {
            'mean_amplitude': np.nan,
            'std_amplitude': np.nan,
            'max_amplitude': np.nan,
            'min_amplitude': np.nan,
            'total_samples': 0,
            'valid_samples': 0
        }


def analyze_all_files(files, root_folder):
    """
    全ファイルを解析してangle_changeの平均振幅を計算

    Args:
        files (list): CSVファイルのパスリスト
        root_folder (str): ルートフォルダパス

    Returns:
        list: 解析結果のリスト
    """
    results = []

    print(f"\n{'='*60}")
    print("angle_change振幅解析")
    print(f"{'='*60}")

    for file_path in files:
        print(f"\n処理中: {os.path.relpath(file_path, root_folder)}")

        # メタデータを抽出
        metadata = extract_metadata_from_path(file_path, root_folder)

        # angle_changeの振幅を計算
        amplitude_stats = calculate_angle_change_amplitude(file_path)

        # 結果をマージ
        result = {**metadata, **amplitude_stats}
        results.append(result)

    return results


def load_angle_change_data_generic(results, use_absolute=False):
    """
    各試行のangle_changeデータを読み込み（生値または絶対値）

    Args:
        results (list): 解析結果のリスト
        use_absolute (bool): 絶対値を使用するかどうか

    Returns:
        tuple: (試行別データ, 条件別データ)
    """
    trial_data = {}
    condition_data = {}

    print(f"\n{'='*60}")
    data_type = "絶対値" if use_absolute else ""
    print(f"angle_change{data_type}データ読み込み")
    print(f"{'='*60}")

    for result in results:
        if result['valid_samples'] == 0:
            continue

        try:
            df = pd.read_csv(result['file_path'])
            if 'angle_change' not in df.columns:
                continue

            angle_change = df['angle_change'].values
            valid_mask = np.isfinite(angle_change)
            angle_change_clean = angle_change[valid_mask]

            if len(angle_change_clean) == 0:
                continue

            # 絶対値を計算（必要に応じて）
            processed_data = np.abs(angle_change_clean) if use_absolute else angle_change_clean

            # 試行別データ
            trial_key = f"{result['condition']}_{result['session_id']}"
            trial_data[trial_key] = {
                'data': processed_data,
                'condition': result['condition'],
                'session_id': result['session_id'],
                'subject': result['subject']
            }

            # 条件別データの統合
            condition = result['condition']
            if condition not in condition_data:
                condition_data[condition] = []
            condition_data[condition].extend(processed_data.tolist())

            print(f"  読み込み: {trial_key} - {len(processed_data)}サンプル")

        except Exception as e:
            print(f"  エラー: {result['file_path']} - {e}")
            continue

    return trial_data, condition_data


def load_angle_change_data(results):
    """
    各試行のangle_changeデータを読み込み（生値）

    Args:
        results (list): 解析結果のリスト

    Returns:
        tuple: (試行別データ, 条件別データ)
    """
    return load_angle_change_data_generic(results, use_absolute=False)


def load_angle_change_abs_data(results):
    """
    各試行のangle_change絶対値データを読み込み

    Args:
        results (list): 解析結果のリスト

    Returns:
        tuple: (試行別データ, 条件別データ)
    """
    return load_angle_change_data_generic(results, use_absolute=True)


def create_trial_distribution_visualization_generic(trial_data, output_dir, output_filename, title_suffix=""):
    """
    試行別angle_change分布の可視化（汎用版）

    Args:
        trial_data (dict): 試行別のangle_changeデータ
        output_dir (str): 出力ディレクトリ
        output_filename (str): 出力ファイル名
        title_suffix (str): タイトルに追加する文字列
    """
    if not trial_data:
        print(f"警告: 試行別{title_suffix}分布を作成するデータがありません")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 10

    # 条件別に色を設定
    conditions = list(set([data['condition'] for data in trial_data.values()]))
    colors = plt.cm.Set3(np.linspace(0, 1, len(conditions)))
    condition_colors = dict(zip(conditions, colors))

    # 図のサイズを調整（試行数に応じて）
    n_trials = len(trial_data)
    fig_width = max(12, n_trials * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # データを準備
    plot_data = []
    labels = []
    colors_list = []

    # 条件でソートしてから表示
    sorted_trials = sorted(trial_data.items(), key=lambda x: (x[1]['condition'], x[1]['session_id']))

    for trial_key, trial_info in sorted_trials:
        plot_data.append(trial_info['data'])
        labels.append(f"{trial_info['condition']}\n{trial_info['session_id']}")
        colors_list.append(condition_colors[trial_info['condition']])

    # 箱ひげ図を作成
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)

    # 色を設定
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 各試行の統計情報をグラフ上に表示
    for i, (trial_key, trial_info) in enumerate(sorted_trials):
        data = trial_info['data']
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # 統計情報のテキストを作成
        stats_text = f"μ={mean_val:.2f}°\nσ={std_val:.2f}°"
        
        # グラフの上部に統計情報を表示
        y_pos = ax.get_ylim()[1] * 0.95  # グラフの上端から5%下
        ax.text(i+1, y_pos, stats_text, ha='center', va='top', 
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel('試行 (条件_セッションID)')
    ax.set_ylabel(f'angle_change{title_suffix} [°]')
    ax.set_title(f'試行別 angle_change{title_suffix} 分布比較（μ=平均, σ=標準偏差）')
    ax.grid(True, alpha=0.3)

    # x軸ラベルを回転
    plt.xticks(rotation=45, ha='right')

    # 凡例を追加
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=condition_colors[cond], alpha=0.7, label=cond)
                      for cond in conditions]
    ax.legend(handles=legend_elements, title='条件', loc='upper right')

    plt.tight_layout()

    # グラフを保存
    output_file = os.path.join(output_dir, output_filename)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"試行別{title_suffix}分布グラフを保存: {output_file}")
    plt.close(fig)


def load_angle_change_data(results):
    """
    各試行のangle_changeデータを読み込み（生値）

    Args:
        results (list): 解析結果のリスト

    Returns:
        tuple: (試行別データ, 条件別データ)
    """
    return load_angle_change_data_generic(results, use_absolute=False)


def load_angle_change_abs_data(results):
    """
    各試行のangle_change絶対値データを読み込み

    Args:
        results (list): 解析結果のリスト

    Returns:
        tuple: (試行別データ, 条件別データ)
    """
    return load_angle_change_data_generic(results, use_absolute=True)


def create_summary_statistics(results):
    """
    被験者別・条件別の要約統計を作成

    Args:
        results (list): 解析結果のリスト

    Returns:
        pandas.DataFrame: 要約統計のデータフレーム
    """
    df = pd.DataFrame(results)

    # 有効なデータのみでグループ化
    df_valid = df[df['valid_samples'] > 0].copy()

    if len(df_valid) == 0:
        print("警告: 有効なデータが見つかりません")
        return pd.DataFrame()

    # 被験者別・条件別の要約統計
    summary = df_valid.groupby(['subject', 'condition']).agg({
        'mean_amplitude': ['mean', 'std', 'count'],
        'max_amplitude': 'max',
        'valid_samples': 'sum'
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
    detailed_file = os.path.join(output_dir, 'angle_change_amplitude_detailed.csv')
    df_detailed.to_csv(detailed_file, index=False)
    print(f"\n詳細結果を保存: {detailed_file}")

    # 要約統計の保存
    if not summary.empty:
        summary_file = os.path.join(output_dir, 'angle_change_amplitude_summary.csv')
        summary.to_csv(summary_file, index=False)
        print(f"要約統計を保存: {summary_file}")


def create_trial_distribution_visualization(trial_data, output_dir):
    """
    試行別angle_change分布の可視化

    Args:
        trial_data (dict): 試行別のangle_changeデータ
        output_dir (str): 出力ディレクトリ
    """
    create_trial_distribution_visualization_generic(
        trial_data, output_dir, 'angle_change_trial_distributions.png'
    )


def create_trial_abs_distribution_visualization(trial_data, output_dir):
    """
    試行別angle_change絶対値分布の可視化

    Args:
        trial_data (dict): 試行別のangle_change絶対値データ
        output_dir (str): 出力ディレクトリ
    """
    create_trial_distribution_visualization_generic(
        trial_data, output_dir, 'angle_change_abs_trial_distributions.png', "絶対値"
    )


def create_condition_distribution_visualization_generic(condition_data, output_dir, output_filename, title_suffix="", stats_header=""):
    """
    条件別統合angle_change分布の可視化（汎用版）

    Args:
        condition_data (dict): 条件別の統合angle_changeデータ
        output_dir (str): 出力ディレクトリ
        output_filename (str): 出力ファイル名
        title_suffix (str): タイトルに追加する文字列
        stats_header (str): 統計表示のヘッダー
    """
    if not condition_data:
        print(f"警告: 条件別{title_suffix}分布を作成するデータがありません")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(12, 8))

    # 条件をソート
    sorted_conditions = sorted(condition_data.keys())

    # データを準備
    plot_data = []
    labels = []

    for condition in sorted_conditions:
        data = condition_data[condition]
        plot_data.append(data)
        labels.append(f"{condition}\n(n={len(data)})")

    # 箱ひげ図を作成
    bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)

    # 色を設定
    colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_conditions)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 各条件の統計情報をグラフ上に表示
    for i, condition in enumerate(sorted_conditions):
        data = np.array(condition_data[condition])
        mean_val = np.mean(data)
        std_val = np.std(data)
        median_val = np.median(data)
        
        # 統計情報のテキストを作成
        stats_text = f"μ={mean_val:.2f}°\nσ={std_val:.2f}°\nM={median_val:.2f}°"
        
        # グラフの上部に統計情報を表示
        y_pos = ax.get_ylim()[1] * 0.95  # グラフの上端から5%下
        ax.text(i+1, y_pos, stats_text, ha='center', va='top', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel('条件 (サンプル数)')
    ax.set_ylabel(f'angle_change{title_suffix} [°]')
    ax.set_title(f'条件別 angle_change{title_suffix} 統合分布比較（μ=平均, σ=標準偏差, M=中央値）')
    ax.grid(True, alpha=0.3)

    # 統計情報を表示
    print(f"\n{'='*60}")
    header = stats_header if stats_header else f"条件別統合{title_suffix}分布統計"
    print(header)
    print(f"{'='*60}")

    for condition in sorted_conditions:
        data = np.array(condition_data[condition])
        print(f"{condition:>10}: 平均={np.mean(data):6.3f}°, 標準偏差={np.std(data):6.3f}°, "
              f"中央値={np.median(data):6.3f}°, サンプル数={len(data):>6}")

    plt.tight_layout()

    # グラフを保存
    output_file = os.path.join(output_dir, output_filename)
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"条件別統合{title_suffix}分布グラフを保存: {output_file}")
    plt.close(fig)


def create_condition_distribution_visualization(condition_data, output_dir):
    """
    条件別統合angle_change分布の可視化

    Args:
        condition_data (dict): 条件別の統合angle_changeデータ
        output_dir (str): 出力ディレクトリ
    """
    create_condition_distribution_visualization_generic(
        condition_data, output_dir, 'angle_change_condition_distributions.png'
    )


def create_condition_abs_distribution_visualization(condition_data, output_dir):
    """
    条件別統合angle_change絶対値分布の可視化

    Args:
        condition_data (dict): 条件別の統合angle_change絶対値データ
        output_dir (str): 出力ディレクトリ
    """
    create_condition_distribution_visualization_generic(
        condition_data, output_dir, 'angle_change_abs_condition_distributions.png',
        "絶対値", "条件別統合絶対値分布統計"
    )


def create_visualization(summary, trial_data, condition_data, trial_abs_data, condition_abs_data, output_dir):
    """
    結果を可視化（既存の関数を拡張）

    Args:
        summary (pandas.DataFrame): 要約統計
        trial_data (dict): 試行別のangle_changeデータ
        condition_data (dict): 条件別の統合angle_changeデータ
        trial_abs_data (dict): 試行別のangle_change絶対値データ
        condition_abs_data (dict): 条件別の統合angle_change絶対値データ
        output_dir (str): 出力ディレクトリ
    """
    if summary.empty:
        print("警告: 可視化するデータがありません")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 12

    # 既存の平均振幅比較グラフ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 被験者別・条件別の棒グラフ
    conditions = summary['condition'].unique()
    subjects = summary['subject'].unique()

    x_pos = np.arange(len(subjects))
    width = 0.8 / len(conditions)

    for i, condition in enumerate(conditions):
        condition_data_summary = summary[summary['condition'] == condition]
        values = [condition_data_summary[condition_data_summary['subject'] == subj]['mean_amplitude_mean'].values[0]
                 if len(condition_data_summary[condition_data_summary['subject'] == subj]) > 0 else 0
                 for subj in subjects]
        errors = [condition_data_summary[condition_data_summary['subject'] == subj]['mean_amplitude_std'].values[0]
                 if len(condition_data_summary[condition_data_summary['subject'] == subj]) > 0 else 0
                 for subj in subjects]

        ax1.bar(x_pos + i * width, values, width, label=condition, alpha=0.8, yerr=errors, capsize=5)

    ax1.set_xlabel('被験者')
    ax1.set_ylabel('angle_change平均振幅 [°]')
    ax1.set_title('被験者別・条件別 angle_change平均振幅')
    ax1.set_xticks(x_pos + width * (len(conditions) - 1) / 2)
    ax1.set_xticklabels(subjects)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 条件別の箱ひげ図
    condition_values = []
    condition_labels = []
    for condition in conditions:
        values = summary[summary['condition'] == condition]['mean_amplitude_mean'].values
        condition_values.append(values)
        condition_labels.append(condition)

    bp2 = ax2.boxplot(condition_values, labels=condition_labels, patch_artist=True)
    
    # 箱ひげ図に色を設定
    colors = plt.cm.Set1(np.linspace(0, 1, len(conditions)))
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 各条件の統計情報をグラフ上に表示
    for i, condition in enumerate(conditions):
        values = summary[summary['condition'] == condition]['mean_amplitude_mean'].values
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # 統計情報のテキストを作成
            stats_text = f"μ={mean_val:.2f}°\nσ={std_val:.2f}°"
            
            # グラフの上部に統計情報を表示
            y_pos = ax2.get_ylim()[1] * 0.95  # グラフの上端から5%下
            ax2.text(i+1, y_pos, stats_text, ha='center', va='top', 
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='white', alpha=0.8, edgecolor='gray'))

    ax2.set_xlabel('条件')
    ax2.set_ylabel('angle_change平均振幅 [°]')
    ax2.set_title('条件別 angle_change平均振幅分布（μ=平均, σ=標準偏差）')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # グラフを保存
    output_file = os.path.join(output_dir, 'angle_change_amplitude_analysis.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"平均振幅グラフを保存: {output_file}")
    plt.close(fig)

    # 新しい分布解析グラフを作成
    create_trial_distribution_visualization(trial_data, output_dir)
    create_condition_distribution_visualization(condition_data, output_dir)

    # 絶対値分布解析グラフも作成
    create_trial_abs_distribution_visualization(trial_abs_data, output_dir)
    create_condition_abs_distribution_visualization(condition_abs_data, output_dir)
def main():
    """メイン関数"""
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description='angle_change平均振幅解析')
    parser.add_argument('folder', help='検索対象フォルダパス')
    parser.add_argument('--output', type=str, default='.', help='出力ディレクトリ（デフォルト: 現在のディレクトリ）')

    args = parser.parse_args()

    root_folder = args.folder
    output_dir = args.output

    print(f"検索対象フォルダ: {root_folder}")
    print(f"出力ディレクトリ: {output_dir}")
    print()

    # フォルダの存在確認
    if not os.path.exists(root_folder):
        print(f"エラー: フォルダが存在しません - {root_folder}")
        sys.exit(1)

    # integrated_analysis.csvファイルを検索
    files = find_integrated_analysis_files(root_folder)

    if not files:
        print("エラー: integrated_analysis.csvファイルが見つかりません")
        sys.exit(1)

    # 全ファイルを解析
    results = analyze_all_files(files, root_folder)

    # angle_changeデータを読み込み
    trial_data, condition_data = load_angle_change_data(results)

    # angle_change絶対値データを読み込み
    trial_abs_data, condition_abs_data = load_angle_change_abs_data(results)

    # 要約統計を作成
    summary = create_summary_statistics(results)

    # 結果を保存
    save_results(results, summary, output_dir)

    # 可視化（拡張版）
    create_visualization(summary, trial_data, condition_data, trial_abs_data, condition_abs_data, output_dir)

    # 結果の表示
    print(f"\n{'='*60}")
    print("解析結果サマリー")
    print(f"{'='*60}")

    if not summary.empty:
        print("\n被験者別・条件別の平均振幅:")
        for _, row in summary.iterrows():
            print(f"  {row['subject']} - {row['condition']}: "
                  f"{row['mean_amplitude_mean']:.3f}° ± {row['mean_amplitude_std']:.3f}° "
                  f"(n={int(row['mean_amplitude_count'])})")

    print(f"\n処理完了:")
    print(f"  解析ファイル数: {len(files)}")
    print(f"  試行データ数: {len(trial_data)}")
    print(f"  条件数: {len(condition_data)}")
    print(f"  出力ディレクトリ: {output_dir}")
    print(f"  生成ファイル:")
    print(f"    - angle_change_amplitude_detailed.csv")
    print(f"    - angle_change_amplitude_summary.csv")
    print(f"    - angle_change_amplitude_analysis.png")
    print(f"    - angle_change_trial_distributions.png")
    print(f"    - angle_change_condition_distributions.png")
    print(f"    - angle_change_abs_trial_distributions.png")
    print(f"    - angle_change_abs_condition_distributions.png")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
