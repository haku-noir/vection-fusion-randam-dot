import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, decimate
from scipy.fft import rfft, rfftfreq, next_fast_len
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from pathlib import Path

# =============================================================================
# Part I: データ読み込みと前処理モジュール
# =============================================================================

def find_experiment_folders(input_path):
    """
    実験データフォルダを再帰的に探索する
    タイムスタンプベースでセッションごとに認識

    Args:
        input_path (str): 検索する基準パス

    Returns:
        list: 実験データを含むフォルダのパスリスト
    """
    experiment_sessions = {}

    if os.path.isfile(input_path):
        # ファイルが指定された場合、そのディレクトリを返す
        return [os.path.dirname(input_path)]

    # フォルダを再帰的に検索
    for root, dirs, files in os.walk(input_path):
        # タイムスタンプ付きファイルからセッションを特定
        for file in files:
            if 'experiment_log.csv' in file:
                timestamp_match = re.match(r'(\d{8}_\d{6})_', file)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                    if timestamp not in experiment_sessions:
                        experiment_sessions[timestamp] = root
                elif 'experiment_log.csv' == file:  # タイムスタンプなしの場合
                    experiment_sessions['default'] = root

    # 特定されたセッションごとに仮想フォルダパスを作成
    session_folders = []
    for timestamp, folder_path in experiment_sessions.items():
        session_folders.append(f"{folder_path}#{timestamp}")  # セッションIDを付加

    return sorted(session_folders)

def get_folder_type(folder_path):
    """
    フォルダのタイプ（gvs, audio, vis）を判定する

    Args:
        folder_path (str): フォルダのパス

    Returns:
        str: フォルダタイプ（'gvs', 'audio', 'vis', 'unknown'）
    """
    folder_name = os.path.basename(folder_path).lower()
    parent_folder = os.path.basename(os.path.dirname(folder_path)).lower()

    if 'gvs' in folder_name or 'gvs' in parent_folder:
        return 'gvs'
    elif 'audio' in folder_name or 'audio' in parent_folder:
        return 'audio'
    elif 'vis' in folder_name or 'vis' in parent_folder:
        return 'vis'
    else:
        return 'unknown'

def get_condition_from_experiment_log(experiment_log_path, trial_number=1):
    """
    experiment_logファイルから条件（red/green）を取得する

    Args:
        experiment_log_path (str): experiment_logファイルのパス
        trial_number (int): 試行番号（デフォルト1）

    Returns:
        str: 条件（'red', 'green', 'unknown'）
    """
    try:
        df = pd.read_csv(experiment_log_path)
        if not df.empty and 'condition' in df.columns:
            trial_condition = df[df['trial'] == trial_number]['condition'].iloc[0] if len(df[df['trial'] == trial_number]) > 0 else df['condition'].iloc[0]
            return trial_condition.lower() if isinstance(trial_condition, str) else 'unknown'
        return 'unknown'
    except Exception as e:
        print(f"experiment_logの読み込みエラー: {e}")
        return 'unknown'

def find_data_files(folder_path, condition='red'):
    """
    フォルダ内から必要なデータファイルを探索する
    タイムスタンプ付きファイルパターンにも対応

    Args:
        folder_path (str): 検索するフォルダのパス
        condition (str): 条件（'red' or 'green'）

    Returns:
        dict: 見つかったファイルのパス辞書
    """
    files = os.listdir(folder_path)
    data_files = {}
    folder_type = get_folder_type(folder_path)

    # 共通ファイルの探索（タイムスタンプ付きパターンも含む）
    for file in files:
        if 'experiment_log.csv' in file or re.match(r'\d{8}_\d{6}_experiment_log\.csv$', file):
            data_files['experiment_log'] = os.path.join(folder_path, file)
        elif ('random_dot_data_trial_1.csv' in file or 'random_dot_trial_1.csv' in file or 
              re.match(r'\d{8}_\d{6}_random_dot.*trial_1\.csv$', file)):
            data_files['random_dot'] = os.path.join(folder_path, file)
        elif ('accel_log_serial_trial_1.csv' in file or 'accel_log_trial_1.csv' in file or 
              'accel_sensor_trial_1.csv' in file or 
              re.match(r'\d{8}_\d{6}_accel_log.*trial_1\.csv$', file)):
            data_files['accel_sensor'] = os.path.join(folder_path, file)

    # フォルダタイプ別の追加ファイル（タイムスタンプ付きパターンも含む）
    if folder_type == 'gvs':
        for file in files:
            if (f'dac_output_{condition}.csv' in file or 
                re.match(rf'\d{{8}}_\d{{6}}_dac_output_{condition}\.csv$', file)):
                data_files['dac_output'] = os.path.join(folder_path, file)

    elif folder_type == 'audio':
        for file in files:
            if (f'audio_{condition}_960Hz.csv' in file or f'audio_trial_1.csv' in file or 
                re.match(rf'\d{{8}}_\d{{6}}_audio.*{condition}.*\.csv$', file) or 
                re.match(r'\d{8}_\d{6}_audio.*trial_1\.csv$', file)):
                data_files['audio'] = os.path.join(folder_path, file)

    # visフォルダの場合は追加ファイルなし（random_dotとaccel_sensorのみ）

    return data_files

def load_integrated_data(session_path):
    """
    セッションパスから統合データを読み込む
    セッションIDを含むパスから適切なファイルを探索

    Args:
        session_path (str): セッションパス（folder_path#session_id形式）

    Returns:
        dict: 読み込まれたデータフレーム辞書
    """
    # セッションパスを分解
    if '#' in session_path:
        folder_path, session_id = session_path.split('#')
    else:
        folder_path = session_path
        session_id = 'default'

    print(f"\nセッションID: {session_id}")

    # experiment_logから条件を取得（セッション固有）
    experiment_log_file = None
    target_files = os.listdir(folder_path)

    for file in target_files:
        if session_id != 'default' and file.startswith(f"{session_id}_experiment_log.csv"):
            experiment_log_file = os.path.join(folder_path, file)
            break
        elif session_id == 'default' and file == 'experiment_log.csv':
            experiment_log_file = os.path.join(folder_path, file)
            break

    if not experiment_log_file:
        print(f"experiment_logファイルが見つかりません: {folder_path} (session: {session_id})")
        return {}

    condition = get_condition_from_experiment_log(experiment_log_file)
    print(f"フォルダ: {os.path.basename(folder_path)}, 条件: {condition}, タイプ: {get_folder_type(folder_path)}, セッション: {session_id}")

    # 必要なファイルを探索（セッション固有）
    data_files = find_session_data_files(folder_path, session_id, condition)

    # データを読み込み
    dataframes = {}
    for key, filepath in data_files.items():
        try:
            df = pd.read_csv(filepath)
            dataframes[key] = df
            print(f"  - {key}: {os.path.basename(filepath)} ({len(df)} rows)")
        except Exception as e:
            print(f"  - {key}読み込みエラー: {e}")

    return dataframes

def find_session_data_files(folder_path, session_id, condition='red'):
    """
    セッション固有のデータファイルを探索する

    Args:
        folder_path (str): 検索するフォルダのパス
        session_id (str): セッションID（タイムスタンプ）
        condition (str): 条件（'red' or 'green'）

    Returns:
        dict: 見つかったファイルのパス辞書
    """
    files = os.listdir(folder_path)
    data_files = {}
    folder_type = get_folder_type(folder_path)

    # セッション固有のファイルを探索
    for file in files:
        if session_id != 'default':
            # タイムスタンプ付きファイル
            if file.startswith(f"{session_id}_experiment_log.csv"):
                data_files['experiment_log'] = os.path.join(folder_path, file)
            elif file.startswith(f"{session_id}_random_dot_trial_1.csv") or file.startswith(f"{session_id}_random_dot_data_trial_1.csv"):
                data_files['random_dot'] = os.path.join(folder_path, file)
            elif (file.startswith(f"{session_id}_accel_sensor_trial_1.csv") or 
                  file.startswith(f"{session_id}_accel_log_trial_1.csv") or 
                  file.startswith(f"{session_id}_accel_log_serial_trial_1.csv")):
                data_files['accel_sensor'] = os.path.join(folder_path, file)
            elif folder_type == 'gvs' and file.startswith(f"{session_id}_dac_output_{condition}.csv"):
                data_files['dac_output'] = os.path.join(folder_path, file)
            elif folder_type == 'audio' and (file.startswith(f"{session_id}_audio_trial_1.csv") or 
                                             file.startswith(f"{session_id}_audio_{condition}")):
                data_files['audio'] = os.path.join(folder_path, file)
        else:
            # タイムスタンプなしファイル
            if file == 'experiment_log.csv':
                data_files['experiment_log'] = os.path.join(folder_path, file)
            elif 'random_dot_data_trial_1.csv' in file or 'random_dot_trial_1.csv' in file:
                data_files['random_dot'] = os.path.join(folder_path, file)
            elif ('accel_log_serial_trial_1.csv' in file or 'accel_log_trial_1.csv' in file or 
                  'accel_sensor_trial_1.csv' in file):
                data_files['accel_sensor'] = os.path.join(folder_path, file)

    # 共通ファイルをスクリプト実行ディレクトリから探索
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # GVSフォルダのdac_outputファイル
    if folder_type == 'gvs':
        dac_file = os.path.join(script_dir, f'dac_output_{condition}.csv')
        if os.path.exists(dac_file):
            data_files['dac_output'] = dac_file
            print(f"  共通ファイルを使用: dac_output_{condition}.csv")

    # オーディオフォルダのaudioファイル
    elif folder_type == 'audio':
        audio_file = os.path.join(script_dir, f'audio_{condition}_960Hz.csv')
        if os.path.exists(audio_file):
            data_files['audio'] = audio_file
            print(f"  共通ファイルを使用: audio_{condition}_960Hz.csv")

    return data_files

def find_target_csv_files(input_path, file_pattern=None):
    """
    指定されたパスから対象のCSVファイルを再帰的に検索
    analyze_acceleration.pyの機能を参考にしたタイムスタンプ対応版

    Args:
        input_path (str): ファイルまたはフォルダのパス
        file_pattern (str, optional): 検索するファイルパターン（正規表現）

    Returns:
        list: 対象CSVファイルのパスリスト
    """
    target_files = []

    # デフォルトのファイルパターン（タイムスタンプ付きファイル）
    if file_pattern is None:
        file_pattern = r'\d{8}_\d{6}_.*\.csv$'

    if os.path.isfile(input_path):
        # 単一ファイルが指定された場合
        if re.search(file_pattern, os.path.basename(input_path)):
            target_files.append(input_path)
    elif os.path.isdir(input_path):
        # ディレクトリが指定された場合、再帰的に検索
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if re.search(file_pattern, file):
                    target_files.append(os.path.join(root, file))

    return sorted(target_files)

def merge_experiment_data(dataframes, folder_type):
    """
    実験データを統合して一つのデータフレームを作成する

    Args:
        dataframes (dict): 読み込まれたデータフレーム辞書
        folder_type (str): フォルダタイプ

    Returns:
        pd.DataFrame: 統合されたデータフレーム
    """
    if 'accel_sensor' not in dataframes or 'random_dot' not in dataframes:
        print("基本データ（accel_sensor, random_dot）が不足しています")
        return None

    accel_df = dataframes['accel_sensor'].copy()
    dot_df = dataframes['random_dot'].copy()

    print(f"\nデータ統合処理開始:")
    print(f"  - 加速度データ: {len(accel_df)} samples")
    print(f"  - ランダムドットデータ: {len(dot_df)} samples")

    # 基本データの準備
    if 'accel_time' in accel_df.columns:
        accel_df['psychopy_time'] = accel_df['accel_time']

    # 角度変化を計算（オイラー角：ロール、ピッチ、ヨー）
    if 'accel_x' in accel_df.columns and 'accel_y' in accel_df.columns and 'accel_z' in accel_df.columns:
        # オイラー角を計算
        # ロール（Z軸回りの左右傾斜）: arctan2(accel_y, accel_z)
        accel_df['roll'] = np.arctan2(accel_df['accel_y'], accel_df['accel_z']) * 180 / np.pi

        # ピッチ（Y軸回りの前後傾斜）: arctan2(-accel_x, sqrt(accel_y^2 + accel_z^2))
        accel_df['pitch'] = np.arctan2(-accel_df['accel_x'], 
                                       np.sqrt(accel_df['accel_y']**2 + accel_df['accel_z']**2)) * 180 / np.pi

        # ヨー（X軸回りの左右回転）は磁力計が必要なため、加速度のみでは精度が低い
        # ここでは簡易的にロールをメインの角度変化として使用

        # 初期位置からの変化量を計算
        initial_roll = accel_df['roll'].iloc[0]
        initial_pitch = accel_df['pitch'].iloc[0]

        accel_df['roll_change'] = accel_df['roll'] - initial_roll
        accel_df['pitch_change'] = accel_df['pitch'] - initial_pitch

        # メインの角度変化としてロールを使用（左右傾斜が主な動きのことが多い）
        accel_df['angle_change'] = accel_df['roll_change']

        print(f"  - オイラー角を計算:")
        print(f"    ロール平均: {accel_df['roll'].mean():.3f}°")
        print(f"    ピッチ平均: {accel_df['pitch'].mean():.3f}°")
        print(f"    ロール変化範囲: {accel_df['roll_change'].min():.3f}° ~ {accel_df['roll_change'].max():.3f}°")

    # ドットデータの変化量を計算
    if 'red_dot_mean_x' in dot_df.columns and 'green_dot_mean_x' in dot_df.columns:
        dot_df['red_dot_x_change'] = dot_df['red_dot_mean_x'] - dot_df['red_dot_mean_x'].iloc[0]
        dot_df['green_dot_x_change'] = dot_df['green_dot_mean_x'] - dot_df['green_dot_mean_x'].iloc[0]
        print(f"  - ドットX座標変化量を計算")

    # 時間ベースでデータを結合
    # psychopy_timeをキーにして最近隣マッチング
    merged_data = []

    for _, accel_row in accel_df.iterrows():
        accel_time = accel_row['psychopy_time']

        # 最近隣のpsychopy_timeを探す
        closest_idx = (dot_df['psychopy_time'] - accel_time).abs().idxmin()
        closest_dot_row = dot_df.loc[closest_idx]

        # データを結合
        merged_row = {
            'psychopy_time': accel_time,
            'accel_x': accel_row['accel_x'],
            'accel_y': accel_row['accel_y'],
            'accel_z': accel_row['accel_z'],
            'angle_change': accel_row.get('angle_change', 0),
            'red_dot_mean_x': closest_dot_row['red_dot_mean_x'],
            'red_dot_mean_y': closest_dot_row['red_dot_mean_y'],
            'green_dot_mean_x': closest_dot_row['green_dot_mean_x'],
            'green_dot_mean_y': closest_dot_row['green_dot_mean_y'],
            'red_dot_x_change': closest_dot_row.get('red_dot_x_change', 0),
            'green_dot_x_change': closest_dot_row.get('green_dot_x_change', 0),
        }

        merged_data.append(merged_row)

    merged_df = pd.DataFrame(merged_data)

    # 追加データの統合
    if folder_type == 'gvs' and 'dac_output' in dataframes:
        dac_df = dataframes['dac_output']
        print(f"  - DACデータを統合: {len(dac_df)} samples")

        # DACデータを時間ベースでマッチング
        if 'time_sec' in dac_df.columns:
            for i, row in merged_df.iterrows():
                closest_dac_idx = (dac_df['time_sec'] - row['psychopy_time']).abs().idxmin()
                closest_dac_row = dac_df.loc[closest_dac_idx]

                merged_df.at[i, 'dac25_output'] = closest_dac_row.get('dac25_output', 0)
                merged_df.at[i, 'dac26_output'] = closest_dac_row.get('dac26_output', 0)
                merged_df.at[i, 'sine_value_internal'] = closest_dac_row.get('sine_value_internal', 0)

    elif folder_type == 'audio' and 'audio' in dataframes:
        # オーディオデータは形式が特殊なので、ここではスキップ
        print(f"  - オーディオデータはスキップ（特殊形式）")

    print(f"\n統合結果: {len(merged_df)} samples")
    return merged_df

def plot_integrated_data(df, session_id, folder_path, folder_type):
    """
    統合データのグラフを作成する

    Args:
        df (pd.DataFrame): 統合データフレーム
        session_id (str): セッションID
        folder_path (str): フォルダパス
        folder_type (str): フォルダタイプ
    """
    if df is None or df.empty:
        print("グラフ作成用のデータがありません")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

    # 4つのサブプロットを作成
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    fig.suptitle(f'統合データ解析 - {folder_type.upper()} Session: {session_id}', fontsize=16)

    # サブプロット1: 加速度データ
    axes[0].plot(df['psychopy_time'], df['accel_x'], label='X軸加速度', alpha=0.7)
    axes[0].plot(df['psychopy_time'], df['accel_y'], label='Y軸加速度', alpha=0.7)
    axes[0].plot(df['psychopy_time'], df['accel_z'], label='Z軸加速度', alpha=0.7)
    axes[0].set_title('加速度データ')
    axes[0].set_ylabel('加速度 (m/s²)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # サブプロット2: オイラー角（ロール、ピッチ）
    if 'roll' in df.columns and 'pitch' in df.columns:
        axes[1].plot(df['psychopy_time'], df['roll_change'], color='blue', linewidth=1.5, label='ロール変化（左右傾斜）')
        axes[1].plot(df['psychopy_time'], df['pitch_change'], color='red', linewidth=1.5, label='ピッチ変化（前後傾斜）', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1].set_title('オイラー角変化（初期位置からの変化）')
        axes[1].set_ylabel('角度変化 (度)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # サブプロット3: 視覚刺激（ドット位置）と角度変化の重ね合わせ
    if 'red_dot_mean_x' in df.columns and 'green_dot_mean_x' in df.columns and 'angle_change' in df.columns:
        ax3_1 = axes[2]
        # ドット位置（左軸）
        line1 = ax3_1.plot(df['psychopy_time'], df['red_dot_mean_x'], color='red', alpha=0.7, label='赤ドットX座標')
        line2 = ax3_1.plot(df['psychopy_time'], df['green_dot_mean_x'], color='green', alpha=0.7, label='緑ドットX座標')
        ax3_1.set_ylabel('X座標 (pixel)', color='black')
        ax3_1.tick_params(axis='y', labelcolor='black')

        # 角度変化（右軸）
        ax3_2 = ax3_1.twinx()
        line3 = ax3_2.plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=2, alpha=0.8, label='ロール変化')
        ax3_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
        ax3_2.set_ylabel('角度変化 (度)', color='orange')
        ax3_2.tick_params(axis='y', labelcolor='orange')

        # 凡例を結合
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax3_1.legend(lines, labels, loc='upper left')
        ax3_1.set_title('視覚刺激（ドット位置）と角度変化')
        ax3_1.grid(True, alpha=0.3)

    # サブプロット4: 刺激データと角度変化の重ね合わせ（フォルダタイプ別）
    if folder_type == 'gvs' and 'dac25_output' in df.columns and 'angle_change' in df.columns:
        ax4_1 = axes[3]
        # DAC出力（左軸）
        line1 = ax4_1.plot(df['psychopy_time'], df['dac25_output'], color='blue', alpha=0.7, label='DAC25出力')
        line2 = ax4_1.plot(df['psychopy_time'], df['dac26_output'], color='cyan', alpha=0.7, label='DAC26出力')
        if 'sine_value_internal' in df.columns:
            line3 = ax4_1.plot(df['psychopy_time'], df['sine_value_internal'], color='purple', alpha=0.5, label='内部sin値')
        ax4_1.set_ylabel('DAC出力値', color='blue')
        ax4_1.tick_params(axis='y', labelcolor='blue')

        # 角度変化（右軸）
        ax4_2 = ax4_1.twinx()
        line4 = ax4_2.plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=2, alpha=0.8, label='ロール変化')
        ax4_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
        ax4_2.set_ylabel('角度変化 (度)', color='orange')
        ax4_2.tick_params(axis='y', labelcolor='orange')

        # 凡例を結合
        all_lines = line1 + line2 + (['sine_value_internal' in df.columns and line3] or []) + line4
        all_lines = [l for l in all_lines if l]  # Noneを除去
        labels = [l.get_label() for l in all_lines]
        ax4_1.legend(all_lines, labels, loc='upper left')
        ax4_1.set_title('GVS刺激と角度変化の重ね合わせ')

    elif folder_type == 'audio' and 'angle_change' in df.columns:
        # オーディオデータの表示は実装が複雑なため、角度変化のみ表示
        axes[3].plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=2, label='ロール変化')
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].set_title('オーディオ刺激と角度変化\n（オーディオデータ表示未実装）')
        axes[3].set_ylabel('角度変化 (度)')
        axes[3].legend()
    else:
        # 視覚刺激のみの場合
        if 'angle_change' in df.columns:
            axes[3].plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=2, label='ロール変化')
            axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[3].set_ylabel('角度変化 (度)')
            axes[3].legend()
        else:
            axes[3].text(0.5, 0.5, '角度データなし', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('視覚刺激のみ')

    axes[3].set_xlabel('時間 (秒)')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    # グラフを保存
    if '#' in folder_path:
        actual_folder_path = folder_path.split('#')[0]
    else:
        actual_folder_path = folder_path

    output_file = os.path.join(actual_folder_path, f"{session_id}_integrated_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"グラフを保存しました: {output_file}")
    plt.close()

def save_angle_data_to_csv(df, output_file, has_dac_output=False, has_audio=False):
    """
    角度データをCSVファイルに保存

    Args:
        df (pd.DataFrame): 角度データを含むデータフレーム
        output_file (str): 出力CSVファイルのパス
        has_dac_output (bool): DAC出力データが含まれているかどうか
        has_audio (bool): オーディオデータが含まれているかどうか
    """
    # 基本的な列（オイラー角を含む）
    columns_to_save = [
        'psychopy_time', 'accel_x', 'accel_y', 'accel_z', 'angle_change',
        'roll', 'pitch', 'roll_change', 'pitch_change',
        'red_dot_mean_x', 'red_dot_mean_y', 'green_dot_mean_x', 'green_dot_mean_y',
        'red_dot_x_change', 'green_dot_x_change'
    ]

    # DAC出力データがある場合は追加
    if has_dac_output:
        dac_columns = ['dac25_output', 'dac26_output', 'sine_value_internal']
        for col in dac_columns:
            if col in df.columns:
                columns_to_save.append(col)

    # オーディオデータがある場合は追加
    if has_audio:
        audio_columns = ['audio_amplitude_l', 'audio_amplitude_r']
        for col in audio_columns:
            if col in df.columns:
                columns_to_save.append(col)

    # 存在する列のみを選択
    available_columns = [col for col in columns_to_save if col in df.columns]
    df_to_save = df[available_columns].copy()

    # CSVファイルに保存
    df_to_save.to_csv(output_file, index=False)
    print(f"統合角度データを保存しました: {output_file}")
    print(f"  - 保存列数: {len(available_columns)}")
    print(f"  - サンプル数: {len(df_to_save)}")
    if has_dac_output:
        print(f"  - DAC出力データを含む")
    if has_audio:
        print(f"  - オーディオデータを含む")

# =============================================================================
# Part II: 既存の関数群（簡略化版）
# =============================================================================

def load_data_efficiently(filepath, use_cols=None, dtype_map=None):
    """大規模CSVファイルをメモリ効率良く読み込む"""
    try:
        df = pd.read_csv(filepath, usecols=use_cols, dtype=dtype_map)
        print(f"'{filepath}' を正常に読み込みました。")
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        return df
    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
        return None

def synchronize_data(dataframes, target_freq, method='spline', order=3):
    """複数のデータフレームを共通の周波数にリサンプリングし、同期させる"""
    resampled_dfs = []
    for name, df in dataframes.items():
        resampled_df = df.resample(target_freq).interpolate(method=method, order=order)
        resampled_df.fillna(method='bfill', inplace=True)
        resampled_df.fillna(method='ffill', inplace=True)
        resampled_df.columns = [f"{name}_{col}" for col in resampled_df.columns]
        resampled_dfs.append(resampled_df)

    synchronized_df = pd.concat(resampled_dfs, axis=1)
    return synchronized_df

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    """バターワースフィルタを適用"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def calculate_postural_angles(df, acc_cols, fs, cutoff=2.0):
    """姿勢角を計算"""
    # 簡略化版実装
    df['roll'] = np.arctan2(df[acc_cols[1]], df[acc_cols[2]]) * 180 / np.pi
    df['pitch'] = np.arctan2(-df[acc_cols[0]], np.sqrt(df[acc_cols[1]]**2 + df[acc_cols[2]]**2)) * 180 / np.pi
    return df

def cross_correlation_analysis(signal1, signal2, fs, max_lag_sec=2):
    """相互相関分析"""
    # 簡略化版実装
    max_lag_samples = int(max_lag_sec * fs)
    correlation = np.correlate(signal1, signal2, mode='full')
    lags = np.arange(-len(signal2)+1, len(signal1))
    lags_ms = lags * 1000 / fs

    peak_idx = np.argmax(np.abs(correlation))
    peak_lag_ms = lags_ms[peak_idx]
    peak_corr = correlation[peak_idx]

    return lags_ms, correlation, peak_lag_ms, peak_corr

def rolling_correlation_analysis(signal1, signal2, fs, window_sec=5):
    """移動相関分析"""
    # 簡略化版実装
    window_samples = int(window_sec * fs)
    df1 = pd.Series(signal1)
    df2 = pd.Series(signal2)
    rolling_corr = df1.rolling(window=window_samples).corr(df2.rolling(window=window_samples))
    return rolling_corr

def plot_all_results(df, fs, stimulus_col, sway_col, window_sec=5.0):
    """全ての結果をプロット"""
    # 簡略化版実装 - 基本的なプロットのみ
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # 時系列プロット
    axes[0].plot(df.index, df[stimulus_col], label=stimulus_col)
    axes[0].plot(df.index, df[sway_col], label=sway_col)
    axes[0].set_title('Time Series Data')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()

    # 相関プロット
    rolling_corr = rolling_correlation_analysis(df[stimulus_col], df[sway_col], fs, window_sec)
    axes[1].plot(df.index, rolling_corr, label=f'Rolling Correlation (window={window_sec}s)')
    axes[1].set_title('Rolling Correlation')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Correlation')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# =============================================================================
# メイン実行ブロック
# =============================================================================

def process_experiment_folder(session_path):
    """
    実験セッションを処理する

    Args:
        session_path (str): セッションパス（folder_path#session_id形式）
    """
    print(f"\n{'='*60}")
    print(f"処理中のセッション: {session_path}")
    print(f"{'='*60}")

    # セッション情報を抽出
    if '#' in session_path:
        folder_path, session_id = session_path.split('#')
    else:
        folder_path = session_path
        session_id = 'default'

    # 統合データを読み込み
    dataframes = load_integrated_data(session_path)

    if not dataframes:
        print("データが見つかりませんでした。スキップします。")
        return

    # データの統合処理
    has_dac_output = 'dac_output' in dataframes
    has_audio = 'audio' in dataframes
    folder_type = get_folder_type(folder_path)

    # 統合データフレームの作成
    integrated_df = merge_experiment_data(dataframes, folder_type)

    if integrated_df is not None:
        # 出力ファイル名の生成
        output_file = os.path.join(folder_path, f"{session_id}_integrated_analysis.csv")

        # 統合データを保存
        save_angle_data_to_csv(integrated_df, output_file, has_dac_output, has_audio)

        # グラフを作成
        print("グラフを作成中...")
        plot_integrated_data(integrated_df, session_id, folder_path, folder_type)
    else:
        print("統合データの作成に失敗しました")

if __name__ == '__main__':
    import sys

    # コマンドライン引数の処理
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # デフォルトパス（現在のディレクトリ）
        input_path = '.'

    # 実験フォルダを探索
    experiment_folders = find_experiment_folders(input_path)

    if not experiment_folders:
        print("実験データセッションが見つかりませんでした。")
        sys.exit(1)

    print(f"見つかった実験セッション数: {len(experiment_folders)}")

    # 各セッションを処理
    for session_path in experiment_folders:
        try:
            process_experiment_folder(session_path)
        except Exception as e:
            print(f"セッション処理エラー {session_path}: {e}")

    print(f"\n全ての処理が完了しました。処理セッション数: {len(experiment_folders)}")
