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


def find_integrated_analysis_files(input_path):
    """
    integrated_analysis.csvファイルを再帰的に検索
    
    Args:
        input_path (str): 検索開始パス
        
    Returns:
        list: 見つかったファイルパスのリスト
    """
    analysis_files = []
    
    if os.path.isfile(input_path) and 'integrated_analysis.csv' in input_path:
        return [input_path]
    
    # 再帰的にintegrated_analysis.csvファイルを検索
    search_pattern = '**/*integrated_analysis.csv'
    analysis_files = list(Path(input_path).glob(search_pattern))
    
    return [str(f) for f in sorted(analysis_files)]


def load_and_filter_integrated_data(filepath, cutoff_freq=3.0):
    """
    integrated_analysisファイルを読み込み、身体動揺成分を抽出
    
    Args:
        filepath (str): integrated_analysis.csvファイルのパス
        cutoff_freq (float): ローパスフィルタのカットオフ周波数
        
    Returns:
        pd.DataFrame: フィルタ処理されたデータフレーム
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
        gvs_cols = ['dac25_output', 'dac26_output', 'sine_value_internal']
        for col in gvs_cols:
            if col in df.columns:
                filtered_df[f'{col}_sway'] = apply_lowpass_filter(
                    df[col].values, cutoff_freq, estimated_fs
                )
                print(f"    {col} → {col}_sway")
        
        # 音響データのフィルタ処理
        audio_cols = ['audio_amplitude_l', 'audio_amplitude_r']
        for col in audio_cols:
            if col in df.columns:
                filtered_df[f'{col}_sway'] = apply_lowpass_filter(
                    df[col].values, cutoff_freq, estimated_fs
                )
                print(f"    {col} → {col}_sway")
        
        print(f"  - フィルタ処理完了: {len(filtered_df)} samples, {len(filtered_df.columns)} columns")
        
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
        sway_output_path = f"{base_name}_sway_{cutoff_freq}Hz.csv"
        
        # 身体動揺関連の列のみを選択
        sway_columns = ['psychopy_time']
        
        # _swayサフィックスがついた列を抽出
        for col in df.columns:
            if col.endswith('_sway'):
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
        
        return sway_output_path
        
    except Exception as e:
        print(f"エラー: ファイル保存に失敗: {e}")
        return None


def plot_sway_data(df, session_id, folder_path, folder_type, cutoff_freq=3.0):
    """
    身体動揺データのグラフを作成
    
    Args:
        df (pd.DataFrame): フィルタ処理済みデータフレーム
        session_id (str): セッションID
        folder_path (str): フォルダパス
        folder_type (str): フォルダタイプ
        cutoff_freq (float): 使用したカットオフ周波数
    """
    if df is None or df.empty:
        print("グラフ作成用のデータがありません")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

    # フォルダタイプに応じてサブプロット数を決定
    is_visual_only = folder_type not in ['gvs', 'audio'] or (
        folder_type == 'gvs' and 'dac25_output_sway' not in df.columns
    ) or (
        folder_type == 'audio' and 'audio_amplitude_l_sway' not in df.columns and 'audio_amplitude_r_sway' not in df.columns
    )

    subplot_count = 2 if is_visual_only else 3
    fig, axes = plt.subplots(subplot_count, 1, figsize=(15, 8 if is_visual_only else 12))

    # axesを常にリストとして扱う
    if subplot_count == 2:
        axes = list(axes)

    fig.suptitle(f'身体動揺解析 ({cutoff_freq}Hz LPF) - {folder_type.upper()} Session: {session_id}', fontsize=16)

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

        # 視覚刺激のみの場合はここでX軸ラベルを追加
        if is_visual_only:
            ax2_1.set_xlabel('時間 (秒)')

    # サブプロット3: 刺激データと角度変化の重ね合わせ（身体動揺成分）
    if not is_visual_only:
        if folder_type == 'gvs' and 'dac25_output_sway' in df.columns and 'angle_change_sway' in df.columns:
            ax3_1 = axes[2]
            # DAC出力（身体動揺成分、左軸）
            line1 = ax3_1.plot(df['psychopy_time'], df['dac25_output_sway'],
                              color='blue', alpha=0.7, label='PIN25出力(+方向)', linewidth=1.5)
            if 'dac26_output_sway' in df.columns:
                line2 = ax3_1.plot(df['psychopy_time'], -df['dac26_output_sway'],
                                  color='cyan', alpha=0.7, label='PIN26出力(-方向)', linewidth=1.5)
            else:
                line2 = []
            
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
            if 'audio_amplitude_l_sway' in df.columns or 'audio_amplitude_r_sway' in df.columns:
                lines_audio = []
                if 'audio_amplitude_l_sway' in df.columns:
                    line1 = ax3_1.plot(df['psychopy_time'], df['audio_amplitude_l_sway'], 
                                      color='blue', alpha=0.6, label='音響振幅L(身体動揺)', linewidth=1.5)
                    lines_audio.extend(line1)
                if 'audio_amplitude_r_sway' in df.columns:
                    line2 = ax3_1.plot(df['psychopy_time'], df['audio_amplitude_r_sway'], 
                                      color='cyan', alpha=0.6, label='音響振幅R(身体動揺)', linewidth=1.5)
                    lines_audio.extend(line2)

                ax3_1.set_ylabel('音響振幅 (身体動揺成分)', color='blue')
                ax3_1.tick_params(axis='y', labelcolor='blue')

                # 角度変化（身体動揺成分、右軸）
                ax3_2 = ax3_1.twinx()
                line_angle = ax3_2.plot(df['psychopy_time'], df['angle_change_sway'], 
                                       color='orange', linewidth=2, alpha=0.8, label='ロール変化(身体動揺)')
                ax3_2.axhline(y=0, color='orange', linestyle='--', alpha=0.5)
                ax3_2.set_ylabel('角度変化 (度)', color='orange')
                ax3_2.tick_params(axis='y', labelcolor='orange')

                # 凡例を結合
                all_lines = lines_audio + line_angle
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
        axes[2].set_xlabel('時間 (秒)')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # グラフを保存（周波数情報を含む）
    base_name = os.path.splitext(session_id)[0] if '.' in session_id else session_id
    output_file = os.path.join(folder_path, f"{base_name}_sway_analysis_{cutoff_freq}Hz.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"身体動揺解析グラフを保存: {os.path.basename(output_file)}")
    plt.close()


def process_integrated_analysis_file(filepath, cutoff_freq=3.0):
    """
    単一のintegrated_analysisファイルを処理
    
    Args:
        filepath (str): integrated_analysis.csvファイルのパス
        cutoff_freq (float): ローパスフィルタのカットオフ周波数
    """
    print(f"\n{'='*80}")
    print(f"処理中: {filepath}")
    print(f"{'='*80}")
    
    # ファイルパスからセッション情報を抽出
    folder_path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # セッションIDを抽出（ファイル名から）
    session_match = re.search(r'(\d{8}_\d{6})', filename)
    if session_match:
        session_id = session_match.group(1)
    else:
        session_id = os.path.splitext(filename)[0].replace('_integrated_analysis', '')
    
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
    filtered_df = load_and_filter_integrated_data(filepath, cutoff_freq)
    
    if filtered_df is not None:
        # 身体動揺データを保存
        sway_file_path = save_sway_data(filtered_df, filepath, cutoff_freq)
        
        # グラフを作成・保存
        if sway_file_path:
            print("身体動揺解析グラフを作成中...")
            plot_sway_data(filtered_df, session_id, folder_path, folder_type, cutoff_freq)
            
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
    
    print(f"入力パス: {input_path}")
    print(f"ローパスフィルタ周波数: {cutoff_freq}Hz")
    print()
    
    # integrated_analysis.csvファイルを検索
    analysis_files = find_integrated_analysis_files(input_path)
    
    if not analysis_files:
        print("integrated_analysis.csvファイルが見つかりませんでした。")
        print("まずanalyze_datasを実行して統合解析を行ってください。")
        return
    
    print(f"見つかったファイル数: {len(analysis_files)}")
    
    # 各ファイルを処理
    success_count = 0
    for filepath in analysis_files:
        try:
            if process_integrated_analysis_file(filepath, cutoff_freq):
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
