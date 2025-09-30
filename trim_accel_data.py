#!/usr/bin/env python3
"""
加速度センサーデータトリミングプログラム (trim_accel_data.py)

accel_sensor_trial_1.csvファイルを読み込み、accel_timeが指定した闾値を初めて超えた
タイミングを0秒として、そこから指定した時間分のデータを切り出し、
トリミングされたデータを新しいCSVファイルに保存する。

機能:
1. accel_sensor_trial_1.csvファイルを再帰的に検索・読み込み
2. accel_timeが指定闾値を超える最初のタイミングを検出
3. そのタイミングを0秒として時間軸を調整
4. 0秒から指定時間のデータを抽出
5. トリミングされたデータを新しいCSVファイルに保存

使用例:
    python trim_accel_data.py volume/
    python trim_accel_data.py . 15.0 180.0
    python trim_accel_data.py volume/ 20.0 120.0
    python trim_accel_data.py volume/20250930_190306_accel_sensor_trial_1.csv 10.0 300.0

パラメータ:
    input_path: 検索パスまたはファイルパス
    start_threshold: 開始闾値（秒） - デフォルト 15.0秒
    duration: 抽出時間（秒） - デフォルト 180.0秒
"""

import pandas as pd
import numpy as np
import os
import sys
import re
from pathlib import Path

def find_accel_sensor_files(input_path):
    """
    accel_sensor_trial_1.csvファイルを再帰的に検索

    Args:
        input_path (str): 検索開始パス

    Returns:
        list: 見つかったファイルパスのリスト
    """
    accel_files = []

    if os.path.isfile(input_path):
        if 'accel_sensor_trial_1.csv' in input_path or 'accel_log_trial_1.csv' in input_path:
            return [input_path]
        else:
            print(f"指定されたファイルは加速度センサーファイルではありません: {input_path}")
            return []

    # 再帰的にaccel_sensor関連ファイルを検索
    search_patterns = [
        '**/*accel_sensor_trial_1.csv',
        '**/*accel_log_trial_1.csv',
        '**/*accel_log_serial_trial_1.csv'
    ]

    for pattern in search_patterns:
        files = list(Path(input_path).glob(pattern))
        accel_files.extend(files)

    return [str(f) for f in sorted(accel_files)]

def trim_accel_data(filepath, start_threshold=15.0, duration=180.0):
    """
    加速度センサーデータをトリミングする

    Args:
        filepath (str): 加速度センサーCSVファイルのパス
        start_threshold (float): 開始タイミングの閾値（秒） - デフォルト15秒
        duration (float): 抽出するデータの長さ（秒） - デフォルト180秒

    Returns:
        tuple: (トリミング済みデータフレーム, 開始時刻, 終了時刻, 元データ長)
    """
    try:
        # データ読み込み
        df = pd.read_csv(filepath)
        print(f"\n加速度データを読み込み: {os.path.basename(filepath)}")
        print(f"  - 元データ: {len(df)} samples, {len(df.columns)} columns")

        # accel_timeカラムの確認
        time_columns = [col for col in df.columns if 'time' in col.lower()]
        print(f"  - 時間カラム: {time_columns}")

        # 適切な時間カラムを選択
        if 'accel_time' in df.columns:
            time_col = 'accel_time'
        elif 'time' in df.columns:
            time_col = 'time'
        elif len(time_columns) > 0:
            time_col = time_columns[0]
            print(f"  - 時間カラムとして {time_col} を使用")
        else:
            raise ValueError("時間カラムが見つかりません")

        # データをソート
        df_sorted = df.sort_values(time_col).reset_index(drop=True)

        # 15秒を初めて超えるタイミングを検出
        start_mask = df_sorted[time_col] > start_threshold
        if not start_mask.any():
            raise ValueError(f"accel_timeが{start_threshold}秒を超えるデータが見つかりません")

        start_idx = start_mask.idxmax()  # 最初にTrueになるインデックス
        start_time = df_sorted[time_col].iloc[start_idx]

        print(f"  - {start_threshold}秒を超える最初のタイミング: {start_time:.3f}秒 (インデックス: {start_idx})")

        # 新しい時間軸を作成（開始時刻を0秒とする）
        df_trimmed = df_sorted.copy()
        df_trimmed[time_col] = df_sorted[time_col] - start_time

        # 0秒から180秒のデータを抽出
        end_time = duration
        time_mask = (df_trimmed[time_col] >= 0) & (df_trimmed[time_col] <= end_time)
        df_result = df_trimmed[time_mask].reset_index(drop=True)

        actual_start = df_result[time_col].min()
        actual_end = df_result[time_col].max()

        print(f"  - トリミング結果:")
        print(f"    元データ長: {len(df)} samples")
        print(f"    トリミング後: {len(df_result)} samples")
        print(f"    時間範囲: {actual_start:.3f}秒 ~ {actual_end:.3f}秒")
        print(f"    実際の長さ: {actual_end - actual_start:.3f}秒")

        return df_result, start_time, actual_end, len(df)

    except Exception as e:
        print(f"エラー: データトリミングに失敗: {e}")
        return None, None, None, None

def save_trimmed_data(df, original_filepath, start_time, end_time, original_length, start_threshold=15.0):
    """
    トリミングされたデータをCSVファイルに保存

    Args:
        df (pd.DataFrame): トリミング済みデータフレーム
        original_filepath (str): 元ファイルのパス
        start_time (float): 元データでの開始時刻
        end_time (float): トリミング後の終了時刻
        original_length (int): 元データの長さ
        start_threshold (float): 開始タイミングの閾値（秒）

    Returns:
        str: 保存されたファイルのパス
    """
    try:
        # 出力ファイル名を生成
        original_dir = os.path.dirname(original_filepath)
        original_filename = os.path.basename(original_filepath)

        # ファイル名からセッションIDを抽出
        session_match = re.search(r'(\d{8}_\d{6})', original_filename)
        if session_match:
            session_id = session_match.group(1)
            # ファイル名に開始秒数と合計秒数を含める
            base_name = f"{session_id}_accel_sensor_trial_1_trimmed_{start_threshold:.0f}s_{end_time:.0f}s"
        else:
            base_name = f"{original_filename.replace('.csv', '')}_trimmed_{start_threshold:.0f}s_{end_time:.0f}s"

        output_filepath = os.path.join(original_dir, f"{base_name}.csv")

        # メタデータを別ファイルに保存
        metadata_filepath = os.path.join(original_dir, f"{base_name}_metadata.txt")
        metadata_lines = [
            f"トリミング済み加速度センサーデータ - メタデータ",
            f"=" * 50,
            f"元ファイル: {original_filename}",
            f"出力ファイル: {base_name}.csv",
            f"元データ長: {original_length} samples",
            f"開始閾値: {start_threshold}秒を超えた時点 (元データ時刻: {start_time:.3f}秒)",
            f"トリミング範囲: 0.0秒 ~ {end_time:.3f}秒 ({end_time:.1f}秒間)",
            f"トリミング後サンプル数: {len(df)} samples",
            f"処理日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"パラメータ:",
            f"  - 開始閾値: {start_threshold}秒",
            f"  - 抽出時間: {end_time}秒",
            f"  - 実際の開始時刻: {start_time:.3f}秒",
            f"  - 実際の終了時刻: {start_time + end_time:.3f}秒",
            f"",
            f"処理概要:",
            f"  accel_timeが{start_threshold}秒を初めて超えたタイミングを0秒として調整し、",
            f"  そこから{end_time}秒間のデータを抽出しました。"
        ]

        # メタデータファイルを保存
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            for line in metadata_lines:
                f.write(line + '\n')

        # CSVファイルを標準形式で保存（ヘッダー付き）
        df.to_csv(output_filepath, index=False)

        print(f"トリミング済みデータを保存: {os.path.basename(output_filepath)}")
        print(f"  - CSVファイル: {output_filepath}")
        print(f"  - メタデータファイル: {metadata_filepath}")
        print(f"  - トリミング範囲: 0.0秒 ~ {end_time:.3f}秒")
        print(f"  - サンプル数: {len(df)}")

        return output_filepath

    except Exception as e:
        print(f"エラー: ファイル保存に失敗: {e}")
        return None

def process_accel_file(filepath, start_threshold=15.0, duration=180.0):
    """
    単一の加速度センサーファイルを処理

    Args:
        filepath (str): 加速度センサーCSVファイルのパス
        start_threshold (float): 開始タイミングの閾値（秒）
        duration (float): 抽出するデータの長さ（秒）

    Returns:
        bool: 処理成功時True、失敗時False
    """
    print(f"\n{'='*80}")
    print(f"処理中: {filepath}")
    print(f"{'='*80}")

    # データをトリミング
    trimmed_df, start_time, end_time, original_length = trim_accel_data(
        filepath, start_threshold, duration
    )

    if trimmed_df is not None:
        # トリミング済みデータを保存
        output_path = save_trimmed_data(
            trimmed_df, filepath, start_time, end_time, original_length, start_threshold
        )

        if output_path:
            return True

    return False

def main():
    """メイン処理関数"""
    print("加速度センサーデータトリミングプログラム")
    print("=" * 60)

    # コマンドライン引数の処理
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = '.'
        print("引数が指定されていません。現在のディレクトリを検索します。")

    # パラメータの設定（コマンドライン引数で変更可能）
    start_threshold = 15.0  # 開始閾値（秒）
    duration = 180.0        # 抽出時間（秒）

    if len(sys.argv) > 2:
        try:
            start_threshold = float(sys.argv[2])
        except ValueError:
            print(f"警告: 無効な開始閾値 '{sys.argv[2]}'。デフォルト15.0秒を使用します。")

    if len(sys.argv) > 3:
        try:
            duration = float(sys.argv[3])
        except ValueError:
            print(f"警告: 無効な抽出時間 '{sys.argv[3]}'。デフォルト180.0秒を使用します。")

    print(f"入力パス: {input_path}")
    print(f"開始閾値: {start_threshold}秒")
    print(f"抽出時間: {duration}秒")
    print()

    # 加速度センサーファイルを検索
    accel_files = find_accel_sensor_files(input_path)

    if not accel_files:
        print("accel_sensor_trial_1.csvファイルが見つかりませんでした。")
        print("対象ファイル: *accel_sensor_trial_1.csv, *accel_log_trial_1.csv")
        print()
        print("使用例:")
        print(f"  python {os.path.basename(__file__)} volume/")
        print(f"  python {os.path.basename(__file__)} . 15.0 180.0")
        print(f"  python {os.path.basename(__file__)} volume/ 20.0 120.0")
        print()
        print("パラメータ:")
        print("  第1引数: 検索パスまたはファイルパス")
        print("  第2引数: 開始閾値（秒） - デフォルト 15.0秒")
        print("  第3引数: 抽出時間（秒） - デフォルト 180.0秒")
        return

    print(f"見つかったファイル数: {len(accel_files)}")

    # 各ファイルを処理
    success_count = 0
    for filepath in accel_files:
        try:
            if process_accel_file(filepath, start_threshold, duration):
                success_count += 1
        except Exception as e:
            print(f"エラー: {filepath} の処理に失敗: {e}")

    print(f"\n{'='*80}")
    print(f"処理完了")
    print(f"総ファイル数: {len(accel_files)}")
    print(f"成功: {success_count}")
    print(f"失敗: {len(accel_files) - success_count}")
    print(f"開始閾値: {start_threshold}秒")
    print(f"抽出時間: {duration}秒")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
