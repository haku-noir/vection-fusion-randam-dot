#!/usr/bin/env python3
"""
位相相関占有時間解析プログラム (phase_occupancy_analyzer.py)

複数の被験者フォルダから位相相関データを読み込み、
赤ドット・緑ドットの位相相関係数の占有時間を分析・可視化する

機能:
1. 複数被験者フォルダからphase/correlation_visualizations内のCSVファイルを検索
2. 各被験者からランダムサンプリング（デフォルト3ファイル）
3. 位相相関係数の占有割合計算（赤≥0.5、緑≥0.5、両方<0.5）
4. 実験条件別（all/audio/gvs/vis）の棒グラフ作成
5. 占有時間データのCSV出力

使用例:
    python phase_occupancy_analyzer.py hatano sugihara fujimoto
    python phase_occupancy_analyzer.py hatano sugihara --samples 5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import re
import random
from pathlib import Path
import argparse


def find_window_correlation_files(subject_folder, condition, cutoff_freq=3.0):
    """
    被験者フォルダ内の指定条件からwindow_correlationsファイルを検索

    Args:
        subject_folder (str): 被験者フォルダパス
        condition (str): 実験条件 ('all', 'audio', 'gvs', 'vis')
        cutoff_freq (float): カットオフ周波数

    Returns:
        list: 見つかったCSVファイルのパスリスト
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

        print(f"  {condition}: {len(files)}ファイル見つかりました")
        return sorted(files)

    except Exception as e:
        print(f"エラー: ファイル検索に失敗 - {e}")
        return []


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
            'total_samples': total_samples,
            'red_high_total': (red_high / total_samples) * 100,
            'green_high_total': (green_high / total_samples) * 100
        }

        return occupancy

    except Exception as e:
        print(f"エラー: 占有時間計算に失敗 - {e} for {csv_file}")
        return {'red_only_high': 0, 'green_only_high': 0, 'both_high': 0, 'both_low': 0, 'total_samples': 0}


def extract_session_id(csv_file):
    """
    CSVファイル名からセッションIDを抽出

    Args:
        csv_file (str): CSVファイルパス

    Returns:
        str: セッションID
    """
    try:
        filename = os.path.basename(csv_file)
        # パターン: {session}_window_correlations_{freq}Hz.csv
        match = re.search(r'(.+)_window_correlations_\d+\.?\d*Hz\.csv', filename)
        if match:
            return match.group(1)
        else:
            # フォールバック: 拡張子を除去
            return os.path.splitext(filename)[0]
    except:
        return "unknown"


def sample_files_randomly(files, num_samples=3):
    """
    ファイルリストからランダムサンプリング

    Args:
        files (list): ファイルパスのリスト
        num_samples (int): サンプル数

    Returns:
        list: サンプリングされたファイルパスのリスト
    """
    if len(files) <= num_samples:
        return files
    else:
        return random.sample(files, num_samples)


def process_subject_condition(subject_folder, subject_name, condition, num_samples=3, cutoff_freq=3.0):
    """
    被験者の特定条件について占有時間を処理

    Args:
        subject_folder (str): 被験者フォルダパス
        subject_name (str): 被験者名
        condition (str): 実験条件
        num_samples (int): サンプル数
        cutoff_freq (float): カットオフ周波数

    Returns:
        list: 各セッションの占有時間データのリスト
    """
    print(f"\n  {condition}条件の処理:")

    # window_correlationsファイルを検索
    files = find_window_correlation_files(subject_folder, condition, cutoff_freq)

    if not files:
        print(f"    警告: {condition}条件でファイルが見つかりません")
        return []

    # ランダムサンプリング
    sampled_files = sample_files_randomly(files, num_samples)
    print(f"    {len(files)}ファイル中{len(sampled_files)}ファイルをサンプリング")

    occupancy_data = []

    for csv_file in sampled_files:
        session_id = extract_session_id(csv_file)
        occupancy = calculate_phase_correlation_occupancy(csv_file)

        # データを整理
        data = {
            'subject': subject_name,
            'condition': condition,
            'session_id': session_id,
            'red_only_high_percent': occupancy.get('red_only_high', 0),
            'green_only_high_percent': occupancy.get('green_only_high', 0),
            'both_high_percent': occupancy.get('both_high', 0),
            'both_low_percent': occupancy.get('both_low', 0),
            'red_high_total_percent': occupancy.get('red_high_total', 0),
            'green_high_total_percent': occupancy.get('green_high_total', 0),
            'total_samples': occupancy.get('total_samples', 0)
        }

        occupancy_data.append(data)
        print(f"    {session_id}: 赤のみ{occupancy.get('red_only_high', 0):.1f}%, 緑のみ{occupancy.get('green_only_high', 0):.1f}%, 両方高{occupancy.get('both_high', 0):.1f}%, 両方低{occupancy.get('both_low', 0):.1f}%")

    return occupancy_data


def create_occupancy_bar_chart(occupancy_data_list, condition, output_dir):
    """
    占有時間の棒グラフを作成

    Args:
        occupancy_data_list (list): 全被験者の占有時間データリスト
        condition (str): 実験条件
        output_dir (str): 出力ディレクトリ
    """
    try:
        if not occupancy_data_list:
            print(f"警告: {condition}条件のデータがありません")
            return

        # データを被験者別に整理
        subjects = {}
        for data in occupancy_data_list:
            subject = data['subject']
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append(data)

        # 被験者名を匿名化（アルファベット順にA, B, C...）
        # subject_names = sorted(subjects.keys())
        # subject_mapping = {name: chr(65 + i) for i, name in enumerate(subject_names)}  # A=65 in ASCII
        subject_names = subjects.keys()
        subject_mapping = {name: chr(65 + i) for i, name in enumerate(subject_names)}  # A=65 in ASCII

        # 表示順序をA, B, Cの順番に統一
        ordered_subjects = [(name, subjects[name]) for name in subject_names]

        # グラフの設定
        plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

        num_subjects = len(subjects)
        max_samples = max(len(sessions) for sessions in subjects.values())

        fig, axes = plt.subplots(1, num_subjects, figsize=(4 * num_subjects, 6), sharey=True)
        if num_subjects == 1:
            axes = [axes]

        # fig.suptitle(f'位相相関占有時間分析 - {condition.upper()}条件', fontsize=16, y=0.98)

        colors = ['lightcoral', 'lightgreen', 'lightblue']  # 赤のみ、緑のみ、両方低
        labels = ['赤ドット', '緑ドット', '非定常状態']

        for i, (subject, sessions) in enumerate(ordered_subjects):
            ax = axes[i]

            # セッション別の棒グラフデータを準備
            x_positions = range(len(sessions))
            red_only = [s['red_only_high_percent'] for s in sessions]
            green_only = [s['green_only_high_percent'] for s in sessions]
            both_high = [s['both_high_percent'] for s in sessions]
            both_low = [s['both_low_percent'] for s in sessions]

            # 積み上げ棒グラフ（両方≥0.5は削除）
            bars1 = ax.bar(x_positions, red_only, color=colors[0], label=labels[0])
            bars2 = ax.bar(x_positions, green_only, bottom=red_only, color=colors[1], label=labels[1])
            bars3 = ax.bar(x_positions, both_low, bottom=np.array(red_only) + np.array(green_only), color=colors[2], label=labels[2])

            # 各セクションに％数値を表示
            for j, (pos, r_only, g_only, b_low) in enumerate(zip(x_positions, red_only, green_only, both_low)):
                # 赤のみ≥0.5の数値表示（5%以上の場合のみ表示）
                if r_only >= 5.0:
                    ax.text(pos, r_only/2, f'{r_only:.1f}%', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='black')

                # 緑のみ≥0.5の数値表示（5%以上の場合のみ表示）
                if g_only >= 5.0:
                    ax.text(pos, r_only + g_only/2, f'{g_only:.1f}%', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='black')

                # 両方<0.5の数値表示（5%以上の場合のみ表示）
                if b_low >= 5.0:
                    ax.text(pos, r_only + g_only + b_low/2, f'{b_low:.1f}%', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='black')

            # x軸ラベルを「試行1, 2, 3...」に設定
            trial_labels = [f'試行{j+1}' for j in range(len(sessions))]
            ax.set_xticks(x_positions)
            ax.set_xticklabels(trial_labels, rotation=0, ha='center')

            # 匿名化された被験者名を表示
            anonymous_name = subject_mapping[subject]
            ax.set_title(f'実験参加者{anonymous_name}', fontsize=14)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y')

            # 最初のサブプロットにのみy軸ラベルを設定
            if i == 0:
                ax.set_ylabel('占有時間 (%)', fontsize=12)
                ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)

        plt.tight_layout()

        # グラフを保存
        output_file = os.path.join(output_dir, f'phase_occupancy_{condition}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  グラフを保存: {os.path.basename(output_file)}")
        plt.close()

    except Exception as e:
        print(f"エラー: グラフ作成に失敗 - {e}")


def save_occupancy_csv(occupancy_data_list, condition, output_dir):
    """
    占有時間データをCSVファイルに保存

    Args:
        occupancy_data_list (list): 占有時間データリスト
        condition (str): 実験条件
        output_dir (str): 出力ディレクトリ
    """
    try:
        if not occupancy_data_list:
            print(f"警告: {condition}条件の保存データがありません")
            return

        # データフレームを作成
        df = pd.DataFrame(occupancy_data_list)

        # 列の順序を整理
        columns_order = [
            'subject', 'condition', 'session_id',
            'red_only_high_percent', 'green_only_high_percent', 
            'both_high_percent', 'both_low_percent',
            'red_high_total_percent', 'green_high_total_percent',
            'total_samples'
        ]

        df = df[columns_order]

        # CSV保存
        output_file = os.path.join(output_dir, f'phase_occupancy_{condition}.csv')
        df.to_csv(output_file, index=False)

        print(f"  CSVを保存: {os.path.basename(output_file)} ({len(df)}行)")

    except Exception as e:
        print(f"エラー: CSV保存に失敗 - {e}")


def main():
    """メイン処理関数"""
    print("位相相関占有時間解析プログラム")
    print("=" * 60)

    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description='位相相関占有時間分析')
    parser.add_argument('subjects', nargs='+', help='被験者フォルダ名（複数指定可能）')
    parser.add_argument('--samples', type=int, default=3, help='各被験者からのサンプル数（デフォルト: 3）')
    parser.add_argument('--freq', type=float, default=3.0, help='カットオフ周波数（デフォルト: 3.0Hz）')
    parser.add_argument('--output', type=str, default='.', help='出力ディレクトリ（デフォルト: 現在のディレクトリ）')

    args = parser.parse_args()

    subject_folders = args.subjects
    num_samples = args.samples
    cutoff_freq = args.freq
    output_dir = args.output

    print(f"被験者: {', '.join(subject_folders)}")
    print(f"サンプル数: {num_samples}")
    print(f"カットオフ周波数: {cutoff_freq}Hz")
    print(f"出力ディレクトリ: {output_dir}")
    print()

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 実験条件
    conditions = ['all', 'audio', 'gvs', 'vis']

    # 各条件について処理
    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"{condition.upper()}条件の処理")
        print(f"{'='*60}")

        all_occupancy_data = []

        # 被験者ごとに処理
        for subject_folder in subject_folders:
            subject_name = os.path.basename(subject_folder.rstrip('/'))
            print(f"\n被験者: {subject_name} ({subject_folder})")

            if not os.path.exists(subject_folder):
                print(f"  警告: フォルダが存在しません - {subject_folder}")
                continue

            # 被験者の条件データを処理
            occupancy_data = process_subject_condition(
                subject_folder, subject_name, condition, num_samples, cutoff_freq
            )

            all_occupancy_data.extend(occupancy_data)

        if all_occupancy_data:
            print(f"\n{condition}条件の結果:")
            print(f"  総データ数: {len(all_occupancy_data)}")

            # 棒グラフを作成
            create_occupancy_bar_chart(all_occupancy_data, condition, output_dir)

            # CSVファイルを保存
            save_occupancy_csv(all_occupancy_data, condition, output_dir)
        else:
            print(f"  {condition}条件: データがありませんでした")

    print(f"\n{'='*60}")
    print("全体処理完了")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"生成されたファイル:")
    for condition in conditions:
        print(f"  - phase_occupancy_{condition}.png")
        print(f"  - phase_occupancy_{condition}.csv")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
