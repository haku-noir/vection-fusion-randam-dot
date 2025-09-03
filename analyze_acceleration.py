#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加速度データの解析プログラム
CSVファイルから加速度データを読み込み、加速度ベクトルの大きさを時間軸でプロット
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def calculate_acceleration_magnitude(csv_file_path):
    """
    CSVファイルから加速度データを読み込み、加速度ベクトルの大きさを計算

    Args:
        csv_file_path (str): CSVファイルのパス

    Returns:
        pd.DataFrame: 時間と加速度ベクトルの大きさを含むデータフレーム
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(csv_file_path)
        print(f"データを読み込みました: {len(df)} 行")

        # 必要な列が存在するかチェック
        required_columns = ['psychopy_time', 'accel_x', 'accel_y', 'accel_z']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"必要な列 '{col}' が見つかりません")

        # 加速度ベクトルの大きさを計算 (√(x² + y² + z²))
        df['accel_magnitude'] = np.sqrt(
            df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2
        )

        # 視覚刺激データも含めて返す（存在する場合）
        output_columns = ['psychopy_time', 'accel_x', 'accel_y', 'accel_z', 'accel_magnitude']

        # 視覚刺激のデータがある場合は追加
        visual_columns = ['red_dot_mean_x', 'red_dot_mean_y', 'green_dot_mean_x', 'green_dot_mean_y']
        for col in visual_columns:
            if col in df.columns:
                output_columns.append(col)

        return df[output_columns]

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def calculate_gravity_and_angles(df):
    """
    重力加速度ベクトルを推定し、各時刻での角度を計算

    Args:
        df (pd.DataFrame): 加速度データを含むデータフレーム

    Returns:
        pd.DataFrame: 角度情報を追加したデータフレーム
    """
    # 全ての時間での加速度ベクトルの平均を重力加速度として推定
    gravity_x = df['accel_x'].mean()
    gravity_y = df['accel_y'].mean()
    gravity_z = df['accel_z'].mean()
    gravity_vector = np.array([gravity_x, gravity_y, gravity_z])
    gravity_magnitude = np.linalg.norm(gravity_vector)

    print(f"\n=== 推定平均加速度ベクトル ===")
    print(f"平均X: {gravity_x:.3f} m/s²")
    print(f"平均Y: {gravity_y:.3f} m/s²")
    print(f"平均Z: {gravity_z:.3f} m/s²")
    print(f"平均の大きさ: {gravity_magnitude:.3f} m/s²")

    # 各時刻での加速度ベクトルと重力ベクトルの角度を計算
    angles = []
    angle_changes = []

    # 初期の加速度ベクトル（基準）
    if len(df) > 0:
        initial_vector = np.array([df.iloc[0]['accel_x'], df.iloc[0]['accel_y'], df.iloc[0]['accel_z']])
        initial_magnitude = np.linalg.norm(initial_vector)
    else:
        initial_vector = gravity_vector
        initial_magnitude = gravity_magnitude

    for idx, row in df.iterrows():
        # 現在の加速度ベクトル
        current_vector = np.array([row['accel_x'], row['accel_y'], row['accel_z']])
        current_magnitude = np.linalg.norm(current_vector)

        # 重力ベクトルとの角度を計算
        dot_product_gravity = np.dot(current_vector, gravity_vector)
        if current_magnitude > 0 and gravity_magnitude > 0:
            cos_angle_gravity = dot_product_gravity / (current_magnitude * gravity_magnitude)
            cos_angle_gravity = np.clip(cos_angle_gravity, -1.0, 1.0)
            angle_rad_gravity = np.arccos(cos_angle_gravity)
            angle_deg_gravity = np.degrees(angle_rad_gravity)
        else:
            angle_deg_gravity = 0.0

        # 初期ベクトルからの角度変化を計算（符号付き）
        dot_product_initial = np.dot(current_vector, initial_vector)
        if current_magnitude > 0 and initial_magnitude > 0:
            cos_angle_initial = dot_product_initial / (current_magnitude * initial_magnitude)
            cos_angle_initial = np.clip(cos_angle_initial, -1.0, 1.0)
            angle_rad_initial = np.arccos(cos_angle_initial)
            angle_deg_initial = np.degrees(angle_rad_initial)

            # 外積を使って回転方向を判定（簡易的な符号付き角度）
            cross_product = np.cross(initial_vector, current_vector)
            # Z成分の符号で回転方向を判定
            if len(cross_product) == 3:
                sign = np.sign(cross_product[2]) if abs(cross_product[2]) > 1e-10 else 1
            else:
                sign = 1
            angle_change = sign * angle_deg_initial
        else:
            angle_change = 0.0

        angles.append(angle_deg_gravity)
        angle_changes.append(angle_change)

    # 角度情報をデータフレームに追加
    df_with_angles = df.copy()
    df_with_angles['angle_to_gravity'] = angles
    df_with_angles['angle_change'] = angle_changes

    # 視覚刺激のx座標の初期値からの変化量を計算（データが存在する場合）
    if 'red_dot_mean_x' in df.columns and 'green_dot_mean_x' in df.columns:
        red_initial = df['red_dot_mean_x'].iloc[0] if len(df) > 0 else 0
        green_initial = df['green_dot_mean_x'].iloc[0] if len(df) > 0 else 0
        df_with_angles['red_dot_x_change'] = df['red_dot_mean_x'] - red_initial
        df_with_angles['green_dot_x_change'] = df['green_dot_mean_x'] - green_initial

    return df_with_angles, gravity_vector


def plot_angle_analysis(df, gravity_vector, output_file=None):
    """
    角度解析のグラフをプロット

    Args:
        df (pd.DataFrame): 角度データを含むデータフレーム
        gravity_vector (np.array): 重力ベクトル
        output_file (str): 保存するファイル名（Noneの場合は表示のみ）
    """
    plt.figure(figsize=(15, 12))

    # 日本語フォントの設定（macOS用）
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

    # サブプロット1: 加速度の各成分と平均の比較
    plt.subplot(3, 1, 1)
    plt.plot(df['psychopy_time'], df['accel_x'], label='X軸加速度', alpha=0.7)
    plt.plot(df['psychopy_time'], df['accel_y'], label='Y軸加速度', alpha=0.7)
    plt.plot(df['psychopy_time'], df['accel_z'], label='Z軸加速度', alpha=0.7)
    plt.axhline(y=gravity_vector[0], color='red', linestyle='--', alpha=0.8, label=f'平均X: {gravity_vector[0]:.2f}')
    plt.axhline(y=gravity_vector[1], color='green', linestyle='--', alpha=0.8, label=f'平均Y: {gravity_vector[1]:.2f}')
    plt.axhline(y=gravity_vector[2], color='blue', linestyle='--', alpha=0.8, label=f'平均Z: {gravity_vector[2]:.2f}')
    plt.xlabel('時間 (秒)')
    plt.ylabel('加速度 (m/s²)')
    plt.title('各軸の加速度変化と推定平均加速度')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # サブプロット2: 加速度ベクトルの大きさ
    plt.subplot(3, 1, 2)
    plt.plot(df['psychopy_time'], df['accel_magnitude'],
             color='purple', linewidth=1.5, label='加速度ベクトルの大きさ')
    gravity_magnitude = np.linalg.norm(gravity_vector)
    plt.axhline(y=gravity_magnitude, color='red', linestyle='--', alpha=0.8,
                label=f'平均の大きさ: {gravity_magnitude:.2f}')
    plt.xlabel('時間 (秒)')
    plt.ylabel('加速度の大きさ (m/s²)')
    plt.title('加速度ベクトルの大きさと平均の大きさ')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # サブプロット3: 角度変化 + 視覚刺激のx座標平均
    plt.subplot(3, 1, 3)

    # 主軸: 角度変化
    ax1 = plt.gca()
    line1 = ax1.plot(df['psychopy_time'], df['angle_change'],
                     color='orange', linewidth=1.5, label='初期位置からの角度変化')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_xlabel('時間 (秒)')
    ax1.set_ylabel('角度変化 (度)', color='orange')
    ax1.set_ylim(-5, 5)  # y軸の範囲を-5度から+5度に固定
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.grid(True, alpha=0.3)

    # 副軸: 視覚刺激のx座標変化量（データが存在する場合）
    if 'red_dot_x_change' in df.columns and 'green_dot_x_change' in df.columns:
        ax2 = ax1.twinx()
        line2 = ax2.plot(df['psychopy_time'], df['red_dot_x_change'],
                         color='red', alpha=0.7, linewidth=1.0, label='赤点X座標変化')
        line3 = ax2.plot(df['psychopy_time'], df['green_dot_x_change'],
                         color='green', alpha=0.7, linewidth=1.0, label='緑点X座標変化')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('X座標変化 (pixel)', color='black')
        ax2.set_ylim(-500, 500)  # y軸の範囲を-500から+500に固定
        ax2.tick_params(axis='y', labelcolor='black')

        # 凡例を結合
        lines = line1 + line2 + line3
        labels = ['初期位置からの角度変化', '赤点X座標変化', '緑点X座標変化']
        ax1.legend(lines, labels, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    plt.title('角度変化と視覚刺激X座標変化')

    plt.tight_layout()

    # 統計情報を表示
    print(f"\n=== 加速度ベクトルの統計情報 ===")
    print(f"平均値: {df['accel_magnitude'].mean():.3f} m/s²")
    print(f"標準偏差: {df['accel_magnitude'].std():.3f} m/s²")
    print(f"最大値: {df['accel_magnitude'].max():.3f} m/s²")
    print(f"最小値: {df['accel_magnitude'].min():.3f} m/s²")
    print(f"測定時間: {df['psychopy_time'].iloc[-1] - df['psychopy_time'].iloc[0]:.3f} 秒")

    # 角度の統計情報を表示
    print(f"\n=== 角度の統計情報 ===")
    print(f"平均との角度の平均値: {df['angle_to_gravity'].mean():.3f} 度")
    print(f"平均との角度の標準偏差: {df['angle_to_gravity'].std():.3f} 度")
    print(f"平均との角度の最大値: {df['angle_to_gravity'].max():.3f} 度")
    print(f"平均との角度の最小値: {df['angle_to_gravity'].min():.3f} 度")
    print(f"角度変化の平均値: {df['angle_change'].mean():.3f} 度")
    print(f"角度変化の標準偏差: {df['angle_change'].std():.3f} 度")
    print(f"角度変化の最大値: {df['angle_change'].max():.3f} 度")
    print(f"角度変化の最小値: {df['angle_change'].min():.3f} 度")

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"角度解析グラフを保存しました: {output_file}")

    plt.show()

def main():
    """メイン処理"""
    # デフォルトのCSVファイルパス
    default_csv = "both/20250829_154715_accel_log_trial_2.csv"

    # コマンドライン引数でCSVファイルが指定された場合はそれを使用
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = default_csv

    # ファイルの存在確認
    if not os.path.exists(csv_file):
        print(f"ファイルが見つかりません: {csv_file}")
        return

    print(f"解析対象ファイル: {csv_file}")

    # データの読み込みと処理
    df = calculate_acceleration_magnitude(csv_file)
    if df is None:
        return

    # 重力加速度と角度の計算
    df_with_angles, gravity_vector = calculate_gravity_and_angles(df)

    # 出力ファイル名を生成
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    angle_output = os.path.join(os.path.dirname(csv_file), f"{base_name}_angle_analysis.png")

    # グラフの作成と表示
    print("\n角度解析のグラフを表示します...")
    plot_angle_analysis(df_with_angles, gravity_vector, angle_output)

if __name__ == "__main__":
    main()
