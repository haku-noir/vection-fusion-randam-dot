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


def create_visualization(summary, output_dir):
    """
    結果を可視化

    Args:
        summary (pandas.DataFrame): 要約統計
        output_dir (str): 出力ディレクトリ
    """
    if summary.empty:
        print("警告: 可視化するデータがありません")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 12

    # 条件別の平均振幅比較
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 被験者別・条件別の棒グラフ
    conditions = summary['condition'].unique()
    subjects = summary['subject'].unique()

    x_pos = np.arange(len(subjects))
    width = 0.8 / len(conditions)

    for i, condition in enumerate(conditions):
        condition_data = summary[summary['condition'] == condition]
        values = [condition_data[condition_data['subject'] == subj]['mean_amplitude_mean'].values[0] 
                 if len(condition_data[condition_data['subject'] == subj]) > 0 else 0 
                 for subj in subjects]
        errors = [condition_data[condition_data['subject'] == subj]['mean_amplitude_std'].values[0] 
                 if len(condition_data[condition_data['subject'] == subj]) > 0 else 0 
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

    ax2.boxplot(condition_values, labels=condition_labels)
    ax2.set_xlabel('条件')
    ax2.set_ylabel('angle_change平均振幅 [°]')
    ax2.set_title('条件別 angle_change平均振幅分布')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # グラフを保存
    output_file = os.path.join(output_dir, 'angle_change_amplitude_analysis.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"グラフを保存: {output_file}")
    plt.close(fig)


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

    # 要約統計を作成
    summary = create_summary_statistics(results)

    # 結果を保存
    save_results(results, summary, output_dir)

    # 可視化
    create_visualization(summary, output_dir)

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
    print(f"  出力ディレクトリ: {output_dir}")
    print(f"  生成ファイル:")
    print(f"    - angle_change_amplitude_detailed.csv")
    print(f"    - angle_change_amplitude_summary.csv")
    print(f"    - angle_change_amplitude_analysis.png")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
