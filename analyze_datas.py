import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, decimate, resample
from scipy.fft import rfft, rfftfreq, next_fast_len
from scipy.interpolate import interp1d
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
    フォルダのタイプ（gvs, audio, vis, all）を判定する

    Args:
        folder_path (str): フォルダのパス

    Returns:
        str: フォルダタイプ（'gvs', 'audio', 'vis', 'all', 'unknown'）
    """
    folder_name = os.path.basename(folder_path).lower()
    parent_folder = os.path.basename(os.path.dirname(folder_path)).lower()

    if 'all' in folder_name or 'all' in parent_folder:
        return 'all'
    elif 'gvs' in folder_name or 'gvs' in parent_folder:
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

def get_experiment_settings_from_log(experiment_log_path, trial_number=1):
    """
    experiment_logファイルから実験設定（audio_reverse, gvs_reverse, visual_reverse, single_color_dot）を取得する

    Args:
        experiment_log_path (str): experiment_logファイルのパス
        trial_number (int): 試行番号（デフォルト1）

    Returns:
        dict: 実験設定（'audio_reverse': bool, 'gvs_reverse': bool, 'visual_reverse': bool, 'single_color_dot': bool）
    """
    try:
        df = pd.read_csv(experiment_log_path)
        if not df.empty:
            # 指定された試行があるかチェック
            trial_data = df[df['trial'] == trial_number] if 'trial' in df.columns and len(df[df['trial'] == trial_number]) > 0 else df

            # 最初の行のデータを使用
            if not trial_data.empty:
                row = trial_data.iloc[0]
                audio_reverse = row.get('audio_reverse', False)
                gvs_reverse = row.get('gvs_reverse', False)
                visual_reverse = row.get('visual_reverse', False)
                single_color_dot = row.get('single_color_dot', False)

                # 文字列の場合をboolに変換
                if isinstance(audio_reverse, str):
                    audio_reverse = audio_reverse.lower() in ['true', '1', 'yes']
                if isinstance(gvs_reverse, str):
                    gvs_reverse = gvs_reverse.lower() in ['true', '1', 'yes']
                if isinstance(visual_reverse, str):
                    visual_reverse = visual_reverse.lower() in ['true', '1', 'yes']
                if isinstance(single_color_dot, str):
                    single_color_dot = single_color_dot.lower() in ['true', '1', 'yes']

                return {
                    'audio_reverse': bool(audio_reverse),
                    'gvs_reverse': bool(gvs_reverse),
                    'visual_reverse': bool(visual_reverse),
                    'single_color_dot': bool(single_color_dot)
                }

        return {'audio_reverse': False, 'gvs_reverse': False, 'visual_reverse': False, 'single_color_dot': False}
    except Exception as e:
        print(f"experiment_logの実験設定読み込みエラー: {e}")
        return {'audio_reverse': False, 'gvs_reverse': False, 'visual_reverse': False, 'single_color_dot': False}

def get_reverse_settings_from_experiment_log(experiment_log_path, trial_number=1):
    """
    後方互換性のための関数（get_experiment_settings_from_logのラッパー）
    """
    settings = get_experiment_settings_from_log(experiment_log_path, trial_number)
    return {
        'audio_reverse': settings['audio_reverse'],
        'gvs_reverse': settings['gvs_reverse'],
        'visual_reverse': settings['visual_reverse']
    }

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
    if folder_type == 'gvs' or folder_type == 'all':
        for file in files:
            if (f'dac_output_{condition}.csv' in file or
                re.match(rf'\d{{8}}_\d{{6}}_dac_output_{condition}\.csv$', file)):
                data_files['dac_output'] = os.path.join(folder_path, file)

    if folder_type == 'audio' or folder_type == 'all':
        for file in files:
            if (f'audio_{condition}_integrated_analysis.csv' in file or f'audio_trial_1.csv' in file or
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
    experiment_settings = get_experiment_settings_from_log(experiment_log_file)
    print(f"フォルダ: {os.path.basename(folder_path)}, 条件: {condition}, タイプ: {get_folder_type(folder_path)}, セッション: {session_id}")
    print(f"実験設定: audio_reverse={experiment_settings['audio_reverse']}, gvs_reverse={experiment_settings['gvs_reverse']}, visual_reverse={experiment_settings['visual_reverse']}, single_color_dot={experiment_settings['single_color_dot']}")

    # 必要なファイルを探索（セッション固有）
    data_files = find_session_data_files(folder_path, session_id, condition, experiment_settings)

    # データを読み込み
    dataframes = {}
    for key, filepath in data_files.items():
        try:
            df = pd.read_csv(filepath)
            dataframes[key] = df
            print(f"  - {key}: {os.path.basename(filepath)} ({len(df)} rows)")
        except Exception as e:
            print(f"  - {key}読み込みエラー: {e}")

    return dataframes, experiment_settings

def find_session_data_files(folder_path, session_id, condition='red', experiment_settings=None):
    """
    セッション固有のデータファイルを探索する

    Args:
        folder_path (str): 検索するフォルダのパス
        session_id (str): セッションID（タイムスタンプ）
        condition (str): 条件（'red' or 'green'）
        experiment_settings (dict): 実験設定（audio_reverse, gvs_reverse, single_color_dot）

    Returns:
        dict: 見つかったファイルのパス辞書
    """
    if experiment_settings is None:
        experiment_settings = {'audio_reverse': False, 'gvs_reverse': False, 'single_color_dot': False}

    files = os.listdir(folder_path)
    data_files = {}
    folder_type = get_folder_type(folder_path)

    # 反転設定に基づいて実際に読み込む条件を決定
    def get_effective_condition(original_condition, is_reversed):
        if is_reversed:
            return 'green' if original_condition == 'red' else 'red'
        return original_condition

    # 各データタイプに対する有効な条件を計算
    audio_condition = get_effective_condition(condition, experiment_settings.get('audio_reverse', False))
    gvs_condition = get_effective_condition(condition, experiment_settings.get('gvs_reverse', False))

    print(f"  オリジナル条件: {condition}")
    if experiment_settings.get('audio_reverse', False):
        print(f"  音響データ条件: {audio_condition} (反転)")
    if experiment_settings.get('gvs_reverse', False):
        print(f"  GVSデータ条件: {gvs_condition} (反転)")
    if experiment_settings.get('single_color_dot', False):
        print(f"  単色ドットモード: {condition}色のみ")    # セッション固有のファイルを探索
    for file in files:
        if session_id != 'default':
            # タイムスタンプ付きファイル
            if file.startswith(f"{session_id}_experiment_log.csv"):
                data_files['experiment_log'] = os.path.join(folder_path, file)
            if file.startswith(f"{session_id}_random_dot_trial_1.csv") or file.startswith(f"{session_id}_random_dot_data_trial_1.csv"):
                data_files['random_dot'] = os.path.join(folder_path, file)
            if (file.startswith(f"{session_id}_accel_sensor_trial_1.csv") or 
                  file.startswith(f"{session_id}_accel_log_trial_1.csv") or 
                  file.startswith(f"{session_id}_accel_log_serial_trial_1.csv")):
                data_files['accel_sensor'] = os.path.join(folder_path, file)
            if (folder_type == 'gvs' or folder_type == 'all') and file.startswith(f"{session_id}_dac_output_{gvs_condition}.csv"):
                data_files['dac_output'] = os.path.join(folder_path, file)
            if (folder_type == 'audio' or folder_type == 'all') and (file.startswith(f"{session_id}_audio_trial_1.csv") or 
                                             file.startswith(f"{session_id}_audio_{audio_condition}") or
                                             file == f"audio_{audio_condition}_integrated_analysis.csv"):
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
    if folder_type == 'gvs' or folder_type == 'all':
        dac_file = os.path.join(script_dir, f'dac_output_{gvs_condition}.csv')
        if os.path.exists(dac_file):
            data_files['dac_output'] = dac_file
            print(f"  共通ファイルを使用: dac_output_{gvs_condition}.csv")

    # オーディオフォルダのaudioファイル
    if folder_type == 'audio' or folder_type == 'all':
        audio_file = os.path.join(script_dir, f'audio_{audio_condition}_integrated_analysis.csv')
        if os.path.exists(audio_file):
            data_files['audio'] = audio_file
            print(f"  共通ファイルを使用: audio_{audio_condition}_integrated_analysis.csv")

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

def merge_experiment_data(dataframes, folder_type, experiment_settings=None, condition='red'):
    """
    実験データをリサンプリングして統合し、一つのデータフレームを作成する

    リサンプリング処理の詳細解説:
    1. 目標サンプリングレート: 60Hz (視覚刺激の最低レート)
    2. アンチエイリアシングフィルタ: ダウンサンプリング前に24Hzローパスフィルタ適用
    3. 補間方法: 3次スプライン補間で滑らかなデータを生成
    4. 時間軸統一: 全データを視覚刺激のpsychopy_timeに合わせる

    Args:
        dataframes (dict): 読み込まれたデータフレーム辞書
        folder_type (str): フォルダタイプ
        experiment_settings (dict): 実験設定（single_color_dot, visual_reverse等）
        condition (str): 実験条件（'red' or 'green'）

    Returns:
        pd.DataFrame: 統合されたデータフレーム
    """
    if experiment_settings is None:
        experiment_settings = {'single_color_dot': False, 'visual_reverse': False}
    if 'accel_sensor' not in dataframes or 'random_dot' not in dataframes:
        print("基本データ（accel_sensor, random_dot）が不足しています")
        return None

    accel_df = dataframes['accel_sensor'].copy()
    dot_df = dataframes['random_dot'].copy()

    print(f"\nリサンプリング統合処理開始:")
    print(f"  - 加速度データ: {len(accel_df)} samples")
    print(f"  - ランダムドットデータ: {len(dot_df)} samples")

    # 基本データの準備
    if 'accel_time' in accel_df.columns:
        accel_df['psychopy_time'] = accel_df['accel_time']

    # 目標サンプリングレートを60Hzに設定（視覚刺激の標準レート）
    TARGET_FS = 60.0  # Hz
    TARGET_NYQUIST = TARGET_FS / 2.0  # 30Hz
    ANTI_ALIAS_CUTOFF = TARGET_NYQUIST * 0.8  # 24Hz（安全マージン20%）

    print(f"  - 目標サンプリングレート: {TARGET_FS}Hz")
    print(f"  - ナイキスト周波数: {TARGET_NYQUIST}Hz")
    print(f"  - アンチエイリアシングカットオフ: {ANTI_ALIAS_CUTOFF}Hz")

    # 視覚刺激データ（ベースとなる時間軸）の準備
    dot_df_sorted = dot_df.sort_values('psychopy_time').reset_index(drop=True)

    # 統一時間軸の作成（視覚刺激の範囲で60Hz）
    time_start = dot_df_sorted['psychopy_time'].min()
    time_end = dot_df_sorted['psychopy_time'].max()
    target_time = np.arange(time_start, time_end, 1.0/TARGET_FS)

    print(f"  - 統一時間軸: {time_start:.3f}s ~ {time_end:.3f}s ({len(target_time)} samples)")

    # リサンプリング関数
    def resample_data_with_antialiasing(data_df, time_col, target_time_axis, original_fs_est):
        """
        データをリサンプリングする（アンチエイリアシングフィルタ適用）
        """
        # データをソートし、重複を削除
        df_clean = data_df.sort_values(time_col).drop_duplicates(subset=[time_col]).reset_index(drop=True)

        # 元のサンプリング周波数が目標周波数より高い場合のみアンチエイリアシングフィルタを適用
        resampled_data = {}

        for col in df_clean.columns:
            if col == time_col:
                continue

            # 数値データのみを処理
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                values = df_clean[col].values
                time_values = df_clean[time_col].values

                # NaNを除去
                valid_mask = ~np.isnan(values)
                if not np.any(valid_mask):
                    resampled_data[col] = np.zeros(len(target_time_axis))
                    continue

                values_clean = values[valid_mask]
                time_clean = time_values[valid_mask]

                # アンチエイリアシングフィルタの適用（元のFsが高い場合のみ）
                if original_fs_est > TARGET_FS * 1.2:  # 20%のマージンを持って判定
                    try:
                        # バターワースローパスフィルタ（4次、カットオフ24Hz）
                        nyquist = original_fs_est / 2.0
                        normalized_cutoff = ANTI_ALIAS_CUTOFF / nyquist
                        if normalized_cutoff < 1.0:  # ナイキスト周波数内の場合のみ適用
                            b, a = butter(4, normalized_cutoff, btype='low')
                            values_filtered = filtfilt(b, a, values_clean)
                            print(f"      {col}: アンチエイリアシングフィルタ適用 ({original_fs_est:.1f}Hz→{TARGET_FS}Hz)")
                        else:
                            values_filtered = values_clean
                    except Exception as e:
                        print(f"      {col}: フィルタ適用エラー - {e}")
                        values_filtered = values_clean
                else:
                    values_filtered = values_clean

                # 3次スプライン補間でリサンプリング
                try:
                    if len(time_clean) > 3:  # スプライン補間には最低4点必要
                        interp_func = interp1d(time_clean, values_filtered, 
                                             kind='cubic', bounds_error=False, 
                                             fill_value='extrapolate')
                    else:
                        interp_func = interp1d(time_clean, values_filtered, 
                                             kind='linear', bounds_error=False, 
                                             fill_value='extrapolate')

                    resampled_values = interp_func(target_time_axis)
                    resampled_data[col] = resampled_values

                except Exception as e:
                    print(f"      {col}: 補間エラー - {e}")
                    # フォールバック: 最近隣補間
                    resampled_values = np.interp(target_time_axis, time_clean, values_filtered)
                    resampled_data[col] = resampled_values
            else:
                # 非数値データは最近隣で補間
                resampled_data[col] = np.full(len(target_time_axis), df_clean[col].iloc[0])

        return resampled_data

    # 加速度データのリサンプリング
    accel_df_sorted = accel_df.sort_values('psychopy_time').reset_index(drop=True)
    accel_fs_est = len(accel_df) / (accel_df['psychopy_time'].max() - accel_df['psychopy_time'].min())
    print(f"\n  加速度データのリサンプリング (推定Fs: {accel_fs_est:.1f}Hz):")

    resampled_accel = resample_data_with_antialiasing(
        accel_df_sorted, 'psychopy_time', target_time, accel_fs_est
    )

    # 視覚刺激データのリサンプリング（既に60Hz程度のはず）
    dot_fs_est = len(dot_df) / (dot_df['psychopy_time'].max() - dot_df['psychopy_time'].min())
    print(f"\n  視覚刺激データのリサンプリング (推定Fs: {dot_fs_est:.1f}Hz):")

    resampled_dot = resample_data_with_antialiasing(
        dot_df_sorted, 'psychopy_time', target_time, dot_fs_est
    )

    # 統合データフレームの作成
    merged_df = pd.DataFrame({'psychopy_time': target_time})

    # リサンプリングされた加速度データを追加
    for col, values in resampled_accel.items():
        merged_df[col] = values

    # リサンプリングされた視覚刺激データを追加
    for col, values in resampled_dot.items():
        if col not in merged_df.columns:  # psychopy_timeは重複回避
            merged_df[col] = values

    # オイラー角の計算（リサンプリング後のデータで）
    if 'accel_x' in merged_df.columns and 'accel_y' in merged_df.columns and 'accel_z' in merged_df.columns:
        print(f"\n  オイラー角計算 (リサンプリング後):")

        # ロール（Z軸回りの左右傾斜）: arctan2(accel_y, accel_z)
        merged_df['roll'] = np.arctan2(merged_df['accel_y'], merged_df['accel_z']) * 180 / np.pi

        # ピッチ（Y軸回りの前後傾斜）: arctan2(-accel_x, sqrt(accel_y^2 + accel_z^2))
        merged_df['pitch'] = np.arctan2(-merged_df['accel_x'], 
                                       np.sqrt(merged_df['accel_y']**2 + merged_df['accel_z']**2)) * 180 / np.pi

        # 初期位置からの変化量を計算
        initial_roll = merged_df['roll'].iloc[0]
        initial_pitch = merged_df['pitch'].iloc[0]

        merged_df['roll_change'] = merged_df['roll'] - initial_roll
        merged_df['pitch_change'] = merged_df['pitch'] - initial_pitch

        # メインの角度変化としてロールを使用
        merged_df['angle_change'] = merged_df['roll_change']

        print(f"    ロール平均: {merged_df['roll'].mean():.3f}°")
        print(f"    ピッチ平均: {merged_df['pitch'].mean():.3f}°")
        print(f"    ロール変化範囲: {merged_df['roll_change'].min():.3f}° ~ {merged_df['roll_change'].max():.3f}°")

    # single_color_dotモードの処理
    # audiovisual_experimentに合わせたロジック:
    # - SINGLE_COLOR_DOT=Trueの場合、デフォルトで反対色のドットを表示
    # - SINGLE_COLOR_DOT=True かつ VISUAL_REVERSE=Trueの場合、元の条件色のドットを表示
    if experiment_settings.get('single_color_dot', False):
        visual_reverse = experiment_settings.get('visual_reverse', False)
        
        if visual_reverse:
            # VISUAL_REVERSE=Trueの場合、元の条件色を表示
            target_condition = condition
            print(f"    single_color_dotモード: 条件={condition} + visual_reverse=True → 表示条件={target_condition} (元の条件色)")
        else:
            # デフォルトでは反対色を表示
            target_condition = 'green' if condition == 'red' else 'red'
            print(f"    single_color_dotモード: 条件={condition} → 表示条件={target_condition} (反対色)")
        
        # 対象でない色のドットデータを削除
        if target_condition == 'red':
            # 赤ドットのみ表示 - 緑ドットデータを削除
            columns_to_remove = [col for col in merged_df.columns if col.startswith('green_dot')]
            for col in columns_to_remove:
                merged_df.drop(columns=[col], inplace=True)
                print(f"    {col}列を削除（single_color_dot: red）")
        else:
            # 緑ドットのみ表示 - 赤ドットデータを削除
            columns_to_remove = [col for col in merged_df.columns if col.startswith('red_dot')]
            for col in columns_to_remove:
                merged_df.drop(columns=[col], inplace=True)
                print(f"    {col}列を削除（single_color_dot: green）")    # ドットデータの変化量を計算（リサンプリング後）
    # single_color_dotモードの場合は片方のドットデータのみ存在する可能性がある
    if 'red_dot_mean_x' in merged_df.columns:
        merged_df['red_dot_x_change'] = merged_df['red_dot_mean_x'] - merged_df['red_dot_mean_x'].iloc[0]
        print(f"    赤ドットX座標変化量を計算")

    if 'green_dot_mean_x' in merged_df.columns:
        merged_df['green_dot_x_change'] = merged_df['green_dot_mean_x'] - merged_df['green_dot_mean_x'].iloc[0]
        print(f"    緑ドットX座標変化量を計算")

    # single_color_dotモードでない場合、両方のドットデータが必要
    if not experiment_settings.get('single_color_dot', False):
        if not ('red_dot_mean_x' in merged_df.columns and 'green_dot_mean_x' in merged_df.columns):
            print(f"    警告: 通常モードですが、ドットデータが不完全です")

    # 追加データのリサンプリング統合
    if (folder_type == 'gvs' or folder_type == 'all') and 'dac_output' in dataframes:
        dac_df = dataframes['dac_output']
        print(f"\n  GVS(DAC)データのリサンプリング (samples: {len(dac_df)}):")

        if 'time_sec' in dac_df.columns:
            # DACデータの推定サンプリング周波数
            dac_fs_est = len(dac_df) / (dac_df['time_sec'].max() - dac_df['time_sec'].min())
            print(f"    推定Fs: {dac_fs_est:.1f}Hz")

            # DACデータをリサンプリング
            resampled_dac = resample_data_with_antialiasing(
                dac_df, 'time_sec', target_time, dac_fs_est
            )

            # DACデータを統合（dac25_outputとdac26_outputからgvs_dac_outputを作成）
            dac25_values = None
            dac26_values = None

            for col, values in resampled_dac.items():
                if col == 'dac25_output':
                    dac25_values = values
                elif col == 'dac26_output':
                    dac26_values = values
                elif col != 'time_sec':
                    merged_df[col] = values

            # PIN25(+方向)とPIN26(-方向)を結合してGVS出力を作成
            if dac25_values is not None and dac26_values is not None:
                merged_df['gvs_dac_output'] = dac25_values - dac26_values
                print(f"    GVS出力を作成: PIN25(+) - PIN26(-)")
            elif dac25_values is not None:
                merged_df['gvs_dac_output'] = dac25_values
                print(f"    GVS出力を作成: PIN25のみ")
            elif dac26_values is not None:
                merged_df['gvs_dac_output'] = -dac26_values
                print(f"    GVS出力を作成: -PIN26のみ")
            else:
                print(f"    警告: DAC出力データが見つかりません")
        else:
            print(f"    警告: 'time_sec'列が見つかりません")

    if (folder_type == 'audio' or folder_type == 'all') and 'audio' in dataframes:
        audio_df = dataframes['audio']
        print(f"\n  音響データのリサンプリング (samples: {len(audio_df)}):")

        if 'psychopy_time' in audio_df.columns and 'angle_change' in audio_df.columns:
            # 音響データの推定サンプリング周波数
            audio_fs_est = len(audio_df) / (audio_df['psychopy_time'].max() - audio_df['psychopy_time'].min())
            print(f"    推定Fs: {audio_fs_est:.1f}Hz")
            print(f"    angle_changeを音響データとして使用")

            # 音響データをリサンプリング
            resampled_audio = resample_data_with_antialiasing(
                audio_df, 'psychopy_time', target_time, audio_fs_est
            )

            # 音響データを統合（angle_changeを使用）
            for col, values in resampled_audio.items():
                if col == 'angle_change':
                    merged_df['audio_angle_change'] = values
                elif col not in ['psychopy_time']:
                    merged_df[f'audio_{col}'] = values
        else:
            print(f"    警告: 'psychopy_time'または'angle_change'列が見つかりません")
            print(f"    利用可能な列: {list(audio_df.columns)}")

    print(f"\nリサンプリング統合結果:")
    print(f"  - 最終データサンプル数: {len(merged_df)}")
    print(f"  - サンプリングレート: {TARGET_FS}Hz")
    print(f"  - 時間範囲: {merged_df['psychopy_time'].min():.3f}s ~ {merged_df['psychopy_time'].max():.3f}s")
    print(f"  - 列数: {len(merged_df.columns)}")

    return merged_df

def plot_integrated_data(df, session_id, folder_path, folder_type, experiment_settings=None, condition='red'):
    """
    統合データのグラフを作成する

    Args:
        df (pd.DataFrame): 統合データフレーム
        session_id (str): セッションID
        folder_path (str): フォルダパス
        folder_type (str): フォルダタイプ
        experiment_settings (dict): 実験設定（single_color_dot, visual_reverse等）
        condition (str): 実験条件（'red' or 'green'）
    """
    if experiment_settings is None:
        experiment_settings = {'single_color_dot': False, 'visual_reverse': False, 'audio_reverse': False, 'gvs_reverse': False}
    if df is None or df.empty:
        print("グラフ作成用のデータがありません")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

    # フォルダタイプに応じてサブプロット数を決定
    is_vision_only = folder_type not in ['gvs', 'audio', 'all'] or (
        folder_type == 'gvs' and 'gvs_dac_output' not in df.columns
    ) or (
        folder_type == 'audio' and 'audio_angle_change' not in df.columns
    ) or (
        folder_type == 'all' and ('gvs_dac_output' not in df.columns or 'audio_angle_change' not in df.columns)
    )

    subplot_count = 2 if is_vision_only else 3
    fig, axes = plt.subplots(subplot_count, 1, figsize=(15, 8 if is_vision_only else 12))

    # axesを常にリストとして扱う
    if subplot_count == 2:
        axes = list(axes)

    # リバース設定の表示文字列を作成
    reverse_indicators = []
    if experiment_settings.get('visual_reverse', False):
        reverse_indicators.append('視覚反転')
    if experiment_settings.get('audio_reverse', False):
        reverse_indicators.append('音響反転')
    if experiment_settings.get('gvs_reverse', False):
        reverse_indicators.append('GVS反転')
    if experiment_settings.get('single_color_dot', False):
        # audiovisual_experimentと同じロジックでタイトル表示条件を決定
        visual_reverse = experiment_settings.get('visual_reverse', False)
        if visual_reverse:
            # VISUAL_REVERSE=Trueの場合、元の条件色を表示
            target_condition = condition
        else:
            # デフォルトでは反対色を表示
            target_condition = 'green' if condition == 'red' else 'red'
        reverse_indicators.append(f'単色ドット({target_condition})')

    # 元の条件と設定情報を含むタイトル
    condition_info = f"条件: {condition}"
    reverse_suffix = f" [{', '.join(reverse_indicators)}]" if reverse_indicators else ""
    fig.suptitle(f'統合データ解析 - {folder_type.upper()} ({condition_info}) Session: {session_id}{reverse_suffix}', fontsize=16)

    # サブプロット1: 加速度データ
    axes[0].plot(df['psychopy_time'], df['accel_x'], label='X軸加速度', alpha=0.7)
    axes[0].plot(df['psychopy_time'], df['accel_y'], label='Y軸加速度', alpha=0.7)
    axes[0].plot(df['psychopy_time'], df['accel_z'], label='Z軸加速度', alpha=0.7)
    axes[0].set_title('加速度データ')
    axes[0].set_ylabel('加速度 (m/s²)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)


    # サブプロット2: 視覚刺激（ドット位置）と角度変化の重ね合わせ
    # single_color_dotモードでは片方のドットデータのみ存在する可能性がある
    if ('red_dot_mean_x' in df.columns or 'green_dot_mean_x' in df.columns) and 'angle_change' in df.columns:
        ax2_1 = axes[1]
        # ドット位置（左軸）
        dot_lines = []
        if 'red_dot_mean_x' in df.columns:
            line1 = ax2_1.plot(df['psychopy_time'], df['red_dot_mean_x'], color='red', alpha=0.7, label='赤ドットX座標')
            dot_lines.extend(line1)
        if 'green_dot_mean_x' in df.columns:
            line2 = ax2_1.plot(df['psychopy_time'], df['green_dot_mean_x'], color='green', alpha=0.7, label='緑ドットX座標')
            dot_lines.extend(line2)
        ax2_1.set_ylabel('X座標 (pixel)', color='black')
        ax2_1.tick_params(axis='y', labelcolor='black')

        # 角度変化（右軸）
        ax2_2 = ax2_1.twinx()
        line3 = ax2_2.plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=2, alpha=0.8, label='ロール変化')
        ax2_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
        ax2_2.set_ylabel('角度変化 (度)', color='orange')
        ax2_2.tick_params(axis='y', labelcolor='orange')

        # 凡例を結合
        all_lines = dot_lines + line3
        labels = [l.get_label() for l in all_lines]
        ax2_1.legend(all_lines, labels, loc='upper left')

        # 視覚刺激のタイトルにリバース情報を追加
        visual_title_suffix = ""
        if experiment_settings.get('single_color_dot', False):
            # audiovisual_experimentと同じロジックでタイトル表示条件を決定
            visual_reverse = experiment_settings.get('visual_reverse', False)
            if visual_reverse:
                # VISUAL_REVERSE=Trueの場合、元の条件色を表示
                target_condition = condition
            else:
                # デフォルトでは反対色を表示
                target_condition = 'green' if condition == 'red' else 'red'
            visual_title_suffix += f" (単色: {target_condition}ドット)"
        if experiment_settings.get('visual_reverse', False):
            visual_title_suffix += " [視覚反転]"

        ax2_1.set_title(f'視覚刺激（ドット位置）と角度変化{visual_title_suffix}')
        ax2_1.grid(True, alpha=0.3)

        # 視覚刺激のみの場合はここでX軸ラベルを追加
        if is_vision_only:
            ax2_1.set_xlabel('時間 (秒)')

    # サブプロット3: 刺激データと角度変化の重ね合わせ（視覚刺激のみでない場合のみ）
    if not is_vision_only:
        if folder_type == 'all' and 'angle_change' in df.columns:
            ax3_1 = axes[2]
            all_lines = []

            # 両方のデータがある場合は3軸表示、そうでなければ2軸表示
            if 'gvs_dac_output' in df.columns and 'audio_angle_change' in df.columns:
                # GVS出力（左軸）
                line1 = ax3_1.plot(df['psychopy_time'], df['gvs_dac_output'], color='blue', alpha=0.7, label='GVS出力')
                ax3_1.set_ylabel('GVS出力値', color='blue')
                ax3_1.tick_params(axis='y', labelcolor='blue')
                all_lines.extend(line1)

                # 音響角度変化（中軸 - 右軸の左側に配置）
                ax3_2 = ax3_1.twinx()
                line2 = ax3_2.plot(df['psychopy_time'], df['audio_angle_change'], color='purple', alpha=0.7, label='音響ロール変化')
                ax3_2.set_ylabel('音響角度変化 (度)', color='purple')
                ax3_2.tick_params(axis='y', labelcolor='purple')
                all_lines.extend(line2)

                # 姿勢角度変化（右軸 - 最右側に配置）
                ax3_3 = ax3_1.twinx()
                # 右軸を右側にオフセット
                ax3_3.spines['right'].set_position(('outward', 60))
                line3 = ax3_3.plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=2, alpha=0.8, label='姿勢ロール変化')
                ax3_3.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
                ax3_3.set_ylabel('姿勢角度変化 (度)', color='orange')
                ax3_3.tick_params(axis='y', labelcolor='orange')
                all_lines.extend(line3)

                # 音響角度変化と姿勢角度変化の軸範囲を一致させる
                angle_min = min(df['audio_angle_change'].min(), df['angle_change'].min())
                angle_max = max(df['audio_angle_change'].max(), df['angle_change'].max())
                angle_range = max(abs(angle_min), abs(angle_max))
                axis_limit = angle_range * 1.1  # 10%のマージンを追加
                ax3_2.set_ylim(-axis_limit, axis_limit)
                ax3_3.set_ylim(-axis_limit, axis_limit)

                gvs_audio_indicators = []
                if experiment_settings.get('gvs_reverse', False):
                    gvs_audio_indicators.append('GVS反転')
                if experiment_settings.get('audio_reverse', False):
                    gvs_audio_indicators.append('音響反転')
                gvs_audio_suffix = f" [{', '.join(gvs_audio_indicators)}]" if gvs_audio_indicators else ""
                ax3_1.set_title(f'統合刺激（GVS + 音響）と姿勢角度変化{gvs_audio_suffix}')

            elif 'gvs_dac_output' in df.columns:
                # GVS出力のみ（左軸）
                line1 = ax3_1.plot(df['psychopy_time'], df['gvs_dac_output'], color='blue', alpha=0.7, label='GVS出力')
                ax3_1.set_ylabel('GVS出力値', color='blue')
                ax3_1.tick_params(axis='y', labelcolor='blue')
                all_lines.extend(line1)

                # 角度変化（右軸）
                ax3_2 = ax3_1.twinx()
                line2 = ax3_2.plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=2, alpha=0.8, label='姿勢ロール変化')
                ax3_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
                ax3_2.set_ylabel('姿勢角度変化 (度)', color='orange')
                ax3_2.tick_params(axis='y', labelcolor='orange')
                all_lines.extend(line2)

                gvs_suffix = " [GVS反転]" if experiment_settings.get('gvs_reverse', False) else ""
                ax3_1.set_title(f'GVS刺激と姿勢角度変化{gvs_suffix}')

            elif 'audio_angle_change' in df.columns:
                # 音響角度変化のみ（左軸）
                line1 = ax3_1.plot(df['psychopy_time'], df['audio_angle_change'], color='purple', alpha=0.7, label='音響ロール変化')
                ax3_1.set_ylabel('音響角度変化 (度)', color='purple')
                ax3_1.tick_params(axis='y', labelcolor='purple')
                all_lines.extend(line1)

                # 姿勢角度変化（右軸）
                ax3_2 = ax3_1.twinx()
                line2 = ax3_2.plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=2, alpha=0.8, label='姿勢ロール変化')
                ax3_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
                ax3_2.set_ylabel('姿勢角度変化 (度)', color='orange')
                ax3_2.tick_params(axis='y', labelcolor='orange')
                all_lines.extend(line2)

                # 音響角度変化と姿勢角度変化の軸範囲を一致させる
                angle_min = min(df['audio_angle_change'].min(), df['angle_change'].min())
                angle_max = max(df['audio_angle_change'].max(), df['angle_change'].max())
                angle_range = max(abs(angle_min), abs(angle_max))
                axis_limit = angle_range * 1.1  # 10%のマージンを追加
                ax3_1.set_ylim(-axis_limit, axis_limit)
                ax3_2.set_ylim(-axis_limit, axis_limit)

                audio_suffix = " [音響反転]" if experiment_settings.get('audio_reverse', False) else ""
                ax3_1.set_title(f'音響刺激と姿勢角度変化{audio_suffix}')

            # 凡例を結合
            labels = [l.get_label() for l in all_lines]
            ax3_1.legend(all_lines, labels, loc='upper left')

        elif folder_type == 'gvs' and 'gvs_dac_output' in df.columns and 'angle_change' in df.columns:
            ax3_1 = axes[2]
            # GVS出力（左軸）
            line1 = ax3_1.plot(df['psychopy_time'], df['gvs_dac_output'], color='blue', alpha=0.7, label='GVS出力')
            ax3_1.set_ylabel('GVS出力値', color='blue')
            ax3_1.tick_params(axis='y', labelcolor='blue')

            # 角度変化（右軸）
            ax3_2 = ax3_1.twinx()
            line2 = ax3_2.plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=2, alpha=0.8, label='ロール変化')
            ax3_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
            ax3_2.set_ylabel('角度変化 (度)', color='orange')
            ax3_2.tick_params(axis='y', labelcolor='orange')

            # 凡例を結合
            all_lines = line1 + line2
            labels = [l.get_label() for l in all_lines]
            ax3_1.legend(all_lines, labels, loc='upper left')
            gvs_suffix = " [GVS反転]" if experiment_settings.get('gvs_reverse', False) else ""
            ax3_1.set_title(f'GVS刺激と角度変化の重ね合わせ{gvs_suffix}')

        elif folder_type == 'audio' and 'audio_angle_change' in df.columns and 'angle_change' in df.columns:
            ax3_1 = axes[2]

            # 音響角度変化（左軸）
            line1 = ax3_1.plot(df['psychopy_time'], df['audio_angle_change'], color='blue', alpha=0.7, label='音響ロール変化')
            ax3_1.set_ylabel('音響角度変化 (度)', color='blue')
            ax3_1.tick_params(axis='y', labelcolor='blue')

            # 姿勢角度変化（右軸）
            ax3_2 = ax3_1.twinx()
            line2 = ax3_2.plot(df['psychopy_time'], df['angle_change'], color='orange', linewidth=2, alpha=0.8, label='姿勢ロール変化')
            ax3_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
            ax3_2.set_ylabel('姿勢角度変化 (度)', color='orange')
            ax3_2.tick_params(axis='y', labelcolor='orange')

            # 音響角度変化と姿勢角度変化の軸範囲を一致させる
            angle_min = min(df['audio_angle_change'].min(), df['angle_change'].min())
            angle_max = max(df['audio_angle_change'].max(), df['angle_change'].max())
            angle_range = max(abs(angle_min), abs(angle_max))
            axis_limit = angle_range * 1.1  # 10%のマージンを追加
            ax3_1.set_ylim(-axis_limit, axis_limit)
            ax3_2.set_ylim(-axis_limit, axis_limit)

            # 凡例を結合
            all_lines = line1 + line2
            labels = [l.get_label() for l in all_lines]
            ax3_1.legend(all_lines, labels, loc='upper left')
            audio_suffix = " [音響反転]" if experiment_settings.get('audio_reverse', False) else ""
            ax3_1.set_title(f'音響刺激（角度変化）と姿勢角度変化の比較{audio_suffix}')

        # 3番目のサブプロットがある場合のX軸ラベル設定
        axes[2].set_xlabel('時間 (秒)')
        axes[2].grid(True, alpha=0.3)

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
        dac_columns = ['gvs_dac_output']
        for col in dac_columns:
            if col in df.columns:
                columns_to_save.append(col)

    # オーディオデータがある場合は追加
    if has_audio:
        audio_columns = ['audio_angle_change']
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
    result = load_integrated_data(session_path)
    if len(result) == 2:
        dataframes, experiment_settings = result
    else:
        # 後方互換性のため
        dataframes = result
        experiment_settings = {'single_color_dot': False, 'visual_reverse': False}

    if not dataframes:
        print("データが見つかりませんでした。スキップします。")
        return

    # 条件を取得
    experiment_log_file = None
    target_files = os.listdir(folder_path)
    for file in target_files:
        if session_id != 'default' and file.startswith(f"{session_id}_experiment_log.csv"):
            experiment_log_file = os.path.join(folder_path, file)
            break
        elif session_id == 'default' and file == 'experiment_log.csv':
            experiment_log_file = os.path.join(folder_path, file)
            break

    condition = 'red'  # デフォルト
    if experiment_log_file:
        condition = get_condition_from_experiment_log(experiment_log_file)

    # データの統合処理
    folder_type = get_folder_type(folder_path)
    has_dac_output = 'dac_output' in dataframes and (folder_type == 'gvs' or folder_type == 'all')
    has_audio = 'audio' in dataframes and (folder_type == 'audio' or folder_type == 'all')

    # 統合データフレームの作成
    integrated_df = merge_experiment_data(dataframes, folder_type, experiment_settings, condition)

    if integrated_df is not None:
        # 出力ファイル名の生成
        output_file = os.path.join(folder_path, f"{session_id}_integrated_analysis.csv")

        # 統合データを保存
        save_angle_data_to_csv(integrated_df, output_file, has_dac_output, has_audio)

        # グラフを作成
        print("グラフを作成中...")
        plot_integrated_data(integrated_df, session_id, folder_path, folder_type, experiment_settings, condition)
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
