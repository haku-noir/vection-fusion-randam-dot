import pandas as pd
import sys
import os
import re
from pathlib import Path

def find_header_row(filename):
    """CSVファイル内のヘッダー行（'TIME'から始まる行）を探す"""
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if line.strip().startswith('TIME'):
                print(f"ヘッダー行を発見: 行番号 {i}, 内容: {line.strip()}")
                return i
    print(f"警告: 'TIME'で始まるヘッダー行が見つかりませんでした: {filename}")
    return None

def find_target_scope_files(input_path):
    """
    指定されたパスからtek****ALL.csvファイルを再帰的に検索

    Args:
        input_path (str): ファイルまたはフォルダのパス

    Returns:
        list: 対象CSVファイルのパスリスト
    """
    target_files = []
    scope_pattern = r'tek\d{4}ALL\.csv$'

    if os.path.isfile(input_path):
        # 単一ファイルが指定された場合
        if re.search(scope_pattern, os.path.basename(input_path)):
            target_files.append(input_path)
    elif os.path.isdir(input_path):
        # ディレクトリが指定された場合、再帰的に検索
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if re.search(scope_pattern, file):
                    target_files.append(os.path.join(root, file))

    return sorted(target_files)

def find_experiment_logs_in_folder(folder_path):
    """
    フォルダ内から{timestamp}_experiment_log.csvファイルを検索

    Args:
        folder_path (str): 検索するフォルダのパス

    Returns:
        list: 見つかったexperiment_logファイルのタイムスタンプリスト（昇順）
    """
    exp_log_pattern = r'(\d{8}_\d{6})_experiment_log\.csv$'
    timestamps = []

    try:
        files = os.listdir(folder_path)
        for file in files:
            match = re.search(exp_log_pattern, file)
            if match:
                timestamps.append(match.group(1))
    except Exception as e:
        print(f"フォルダの読み込みエラー: {e}")
        return []

    return sorted(timestamps)

def generate_output_filename(scope_file_path, timestamps):
    """
    オシロスコープファイルに対応する出力ファイル名を生成

    Args:
        scope_file_path (str): オシロスコープファイルのパス
        timestamps (list): 対応するタイムスタンプのリスト

    Returns:
        str: 出力ファイル名、エラーの場合はNone
    """
    folder_path = os.path.dirname(scope_file_path)
    folder_name = os.path.basename(folder_path)
    filename = os.path.basename(scope_file_path)

    # tek番号を抽出
    tek_match = re.search(r'tek(\d{4})ALL\.csv$', filename)
    if not tek_match:
        print(f"エラー: ファイル名からtek番号を抽出できません: {filename}")
        return None

    tek_number = int(tek_match.group(1))

    # 同じフォルダ内のオシロスコープファイルを全て検索
    scope_files_in_folder = []
    try:
        files = os.listdir(folder_path)
        for file in files:
            if re.search(r'tek\d{4}ALL\.csv$', file):
                tek_match_inner = re.search(r'tek(\d{4})ALL\.csv$', file)
                if tek_match_inner:
                    scope_files_in_folder.append(int(tek_match_inner.group(1)))
    except Exception as e:
        print(f"フォルダの読み込みエラー: {e}")
        return None

    scope_files_in_folder.sort()

    # ファイル数のチェック
    if len(scope_files_in_folder) != len(timestamps):
        print(f"エラー: ファイル数が一致しません")
        print(f"  フォルダ: {folder_path}")
        print(f"  オシロスコープファイル数: {len(scope_files_in_folder)} (tek番号: {scope_files_in_folder})")
        print(f"  experiment_logファイル数: {len(timestamps)} (timestamps: {timestamps})")
        return None

    # tek番号のインデックスを見つける
    try:
        tek_index = scope_files_in_folder.index(tek_number)
        corresponding_timestamp = timestamps[tek_index]

        output_filename = f"{corresponding_timestamp}_scope_{folder_name}_trial_1.csv"
        output_path = os.path.join(folder_path, output_filename)

        print(f"ファイル対応:")
        print(f"  入力: {filename} (tek番号: {tek_number})")
        print(f"  出力: {output_filename} (timestamp: {corresponding_timestamp})")

        return output_path

    except ValueError:
        print(f"エラー: tek番号 {tek_number} がリストに見つかりません")
        return None

def process_oscilloscope_csv(input_filename, output_filename):
    """
    オシロスコープのCSVファイルを処理する関数

    Args:
        input_filename (str): 入力CSVファイル名
        output_filename (str): 出力CSVファイル名
    """
    # ヘッダー行を自動的に見つける
    header_row = find_header_row(input_filename)
    if header_row is None:
        print("エラー: CSVファイルにヘッダー行（'TIME'から始まる行）が見つかりません。")
        return

    # ヘッダー行を指定してCSVを読み込む
    try:
        df = pd.read_csv(input_filename, header=header_row)
        print(f"データ読み込み成功: {len(df)}行, 列: {list(df.columns)}")
    except Exception as e:
        print(f"ファイル読み込み中にエラーが発生しました: {e}")
        return

    # 列名を正しく設定（ヘッダー行を正しく読み込めなかった場合の対処）
    expected_columns = ['TIME', 'CH1', 'CH2', 'CH3']

    # 列数をチェックして適切な列名を設定
    if len(df.columns) >= 4:
        # 4列以上ある場合は最初の4列を使用
        df = df.iloc[:, :4]
        df.columns = expected_columns
        print(f"列名を修正しました: {list(df.columns)}")
    elif len(df.columns) == 3:
        # 3列の場合（CH3がない場合）
        df.columns = ['TIME', 'CH1', 'CH3']
        print(f"3列データとして処理: {list(df.columns)}")
    else:
        print(f"エラー: 予期しない列数です: {len(df.columns)}")
        return

    # 不要な列（Unnamedなど）があれば削除
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # NaNを含む行を削除
    df.dropna(inplace=True)

    # データ型を数値に変換（変換できないものはNaNになり、dropnaで消える）
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"データ前処理完了: {len(df)}行, 数値列: {list(df.columns)}")

    # --- トリガーポイントの検出 ---
    # CH3が3を初めて超えるインデックスを見つける
    try:
        # CH3列の存在確認
        if 'CH3' not in df.columns:
            print(f"警告: CH3列が見つかりません。利用可能な列: {list(df.columns)}")
            print("CH3列なしで処理を続行します。")
            # CH3がない場合は時間ベースでトリミング（-5秒から180秒まで）
            start_time = -5.0
            end_time = 180.0
            df = df[(df['TIME'] >= start_time) & (df['TIME'] <= end_time)].copy()
            df['TIME'] = df['TIME'] - start_time  # 時間を0からスタートに調整
            df.reset_index(drop=True, inplace=True)
            print(f"時間ベースでトリミング完了: {start_time}s～{end_time}s → {len(df)}行")
        else:
            print(f"CH3列の統計: 最小値={df['CH3'].min():.3f}, 最大値={df['CH3'].max():.3f}, 平均値={df['CH3'].mean():.3f}")

            # CH3が3を超える点を探す
            trigger_points = df[df['CH3'] >= 3]
            if trigger_points.empty:
                print("警告: CH3が3を超えるデータポイントが見つかりませんでした。")
                print("時間ベースでトリミングします。")
                # トリガーがない場合は時間ベースでトリミング
                start_time = -5.0
                end_time = 180.0
                df = df[(df['TIME'] >= start_time) & (df['TIME'] <= end_time)].copy()
                df['TIME'] = df['TIME'] - start_time
                df.reset_index(drop=True, inplace=True)
                print(f"時間ベースでトリミング完了: {len(df)}行")
            else:
                start_index = trigger_points.index[0]
                print(f"トリガーポイント発見: インデックス={start_index}, CH3値={df.loc[start_index, 'CH3']:.3f}")

                # --- 時間軸の更新と開始点のトリミング ---
                # トリガーポイントの時間を取得
                time_offset = df.loc[start_index, 'TIME']
                # TIME列を更新
                df['TIME'] = df['TIME'] - time_offset
                # 0秒より前のデータを削除
                df = df[df['TIME'] >= 0].copy()
                df.reset_index(drop=True, inplace=True)
                print(f"トリガーベースでトリミング完了: {len(df)}行")

                # --- 終了点の検出とトリミング ---
                # トリガー後、CH3が0.1未満に初めてなるインデックスを見つける
                try:
                    end_points = df[df['CH3'] < 0.1]
                    if not end_points.empty:
                        end_index = end_points.index[0]
                        print(f"終了ポイント発見: インデックス={end_index}, CH3値={df.loc[end_index, 'CH3']:.3f}")
                        # 終了インデックスまでのデータを抽出
                        df = df.loc[:end_index].copy()
                    else:
                        print("終了ポイントが見つかりませんでした。全データを出力します。")
                except Exception as e:
                    print(f"終了ポイント検出中にエラーが発生しました: {e}")

    except Exception as e:
        print(f"トリガーポイント検出中にエラーが発生しました: {e}")
        return
    # 処理後のデータをCSVに出力
    df.to_csv(output_filename, index=False)
    print(f"処理が完了しました。結果を'{output_filename}'に出力しました。")
    print(f"最終データ: {len(df)}行, 時間範囲: {df['TIME'].min():.3f}s - {df['TIME'].max():.3f}s")

def process_folder_recursively(input_path):
    """
    フォルダを再帰的に処理してオシロスコープファイルを変換

    Args:
        input_path (str): 処理するフォルダまたはファイルのパス
    """
    # オシロスコープファイルを検索
    scope_files = find_target_scope_files(input_path)

    if not scope_files:
        print(f"オシロスコープファイル (tek****ALL.csv) が見つかりませんでした: {input_path}")
        return

    print(f"見つかったオシロスコープファイル数: {len(scope_files)}")

    # フォルダ別にファイルをグループ化
    folders_with_files = {}
    for scope_file in scope_files:
        folder_path = os.path.dirname(scope_file)
        if folder_path not in folders_with_files:
            folders_with_files[folder_path] = []
        folders_with_files[folder_path].append(scope_file)

    # 各フォルダを処理
    processed_count = 0
    error_count = 0

    for folder_path, files_in_folder in folders_with_files.items():
        print(f"\n{'='*60}")
        print(f"処理中のフォルダ: {folder_path}")
        print(f"{'='*60}")

        # experiment_logファイルのタイムスタンプを取得
        timestamps = find_experiment_logs_in_folder(folder_path)

        if not timestamps:
            print(f"エラー: experiment_logファイルが見つかりません: {folder_path}")
            error_count += len(files_in_folder)
            continue

        print(f"見つかったexperiment_logファイル: {len(timestamps)}個")
        print(f"タイムスタンプ: {timestamps}")
        print(f"オシロスコープファイル: {len(files_in_folder)}個")

        # 各オシロスコープファイルを処理
        for scope_file in files_in_folder:
            output_path = generate_output_filename(scope_file, timestamps)

            if output_path is None:
                print(f"エラー: 出力ファイル名の生成に失敗しました: {scope_file}")
                error_count += 1
                continue

            try:
                print(f"\n--- 処理開始: {os.path.basename(scope_file)} ---")
                process_oscilloscope_csv(scope_file, output_path)
                processed_count += 1
                print(f"✓ 変換完了: {os.path.basename(scope_file)} → {os.path.basename(output_path)}")
            except Exception as e:
                print(f"✗ 変換エラー: {scope_file}")
                print(f"  エラー内容: {e}")
                import traceback
                print(f"  詳細なエラー情報:")
                traceback.print_exc()
                error_count += 1

    print(f"\n{'='*60}")
    print(f"処理結果:")
    print(f"  成功: {processed_count}ファイル")
    print(f"  エラー: {error_count}ファイル")
    print(f"{'='*60}")

if __name__ == '__main__':
    # コマンドライン引数の処理
    if len(sys.argv) == 2:
        # フォルダ指定モード（新機能）
        input_path = sys.argv[1]
        process_folder_recursively(input_path)
    elif len(sys.argv) == 3:
        # 従来の単一ファイル処理モード
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        process_oscilloscope_csv(input_csv, output_csv)
    else:
        print("使用法:")
        print("  フォルダ処理: python process_scope_data.py <フォルダパス>")
        print("  単一ファイル: python process_scope_data.py <入力ファイル名.csv> <出力ファイル名.csv>")
        print("\nフォルダ処理では以下のルールで変換されます:")
        print("  - tek****ALL.csvファイルを再帰的に検索")
        print("  - 同じフォルダ内の{timestamp}_experiment_log.csvと対応付け")
        print("  - 出力ファイル名: {timestamp}_scope_{フォルダ名}_trial_1.csv")
