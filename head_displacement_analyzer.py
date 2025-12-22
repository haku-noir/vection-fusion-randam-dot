#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
頭部変位解析プログラム (Head Displacement Analyzer)

倒立振子モデルを仮定し、加速度センサから計算された傾斜角度（roll, pitch）を用いて
頭部の前後・左右方向の変位を計算し、可視化するプログラム。

使用方法:
    # 単一ファイル
    python head_displacement_analyzer.py <integrated_analysis.csv>

    # ディレクトリ内を再帰的に処理
    python head_displacement_analyzer.py <directory>

オプション:
    --height HEIGHT     被験者の身長 [cm]（デフォルト: 170）
    --ankle ANKLE       足首高さ [cm]（デフォルト: 8）
    --output OUTPUT     出力ディレクトリ（デフォルト: 入力ファイルと同じディレクトリ）
    --start START       解析開始時間 [秒]（デフォルト: 0）
    --end END           解析終了時間 [秒]（デフォルト: データ終端）
    --use-change        roll_change, pitch_changeを使用（デフォルト: roll, pitchを使用）
    --pattern PATTERN   検索するファイルパターン（デフォルト: *integrated_analysis*.csv）
    --sway              ローパスフィルタ済み（_sway_）ファイルのみを対象
"""

import os
import sys
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from glob import glob

# =============================================================================
# 設定変数（ここで直接変更可能）
# =============================================================================

# 被験者パラメータ
DEFAULT_HEIGHT_CM = 165.0      # 被験者の身長 [cm]
DEFAULT_ANKLE_HEIGHT_CM = 20.0  # 足首高さ [cm]

# 解析時間範囲
DEFAULT_START_TIME = 20      # 解析開始時間 [秒]（Noneで最初から）
DEFAULT_END_TIME = None        # 解析終了時間 [秒]（Noneで最後まで）

# ファイル検索パターン
DEFAULT_FILE_PATTERN = "*integrated_analysis*.csv"  # 検索パターン
EXCLUDE_PATTERNS = ["_head_displacement", "_phase_analysis"]  # 除外パターン
DEFAULT_SWAY_ONLY = True

# 角度列の設定
USE_CHANGE_DEFAULT = False     # True: roll_change/pitch_change使用, False: roll/pitch使用

# =============================================================================
# 日本語フォント設定
# =============================================================================
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']


def find_csv_files(input_path, pattern="*integrated_analysis*.csv", sway_only=False):
    """
    ディレクトリ内からCSVファイルを再帰的に検索

    Args:
        input_path (str): ファイルまたはディレクトリのパス
        pattern (str): 検索パターン（glob形式）
        sway_only (bool): _sway_ファイルのみを対象にするか

    Returns:
        list: 見つかったCSVファイルのパスリスト
    """
    input_path = Path(input_path)

    if input_path.is_file():
        return [str(input_path)]

    if not input_path.is_dir():
        print(f"エラー: パスが見つかりません: {input_path}")
        return []

    # 再帰的に検索
    all_files = list(input_path.rglob(pattern))

    # 除外パターンでフィルタリング
    filtered_files = []
    for f in all_files:
        filename = f.name

        # 除外パターンに一致するものをスキップ
        if any(excl in filename for excl in EXCLUDE_PATTERNS):
            continue

        # sway_onlyの場合、_sway_を含むファイルのみ
        if sway_only and "_sway_" not in filename:
            continue

        # sway_onlyでない場合、_sway_を含まないファイルのみ
        if not sway_only and "_sway_" in filename:
            continue

        filtered_files.append(str(f))

    return sorted(filtered_files)


def detect_column_names(df):
    """
    CSVファイルの列名を検出し、適切な列名マッピングを返す

    Args:
        df (pd.DataFrame): データフレーム

    Returns:
        dict: 列名マッピング {'roll': 実際の列名, 'pitch': 実際の列名, ...}
    """
    columns = df.columns.tolist()
    mapping = {}

    # roll列の検出
    if 'roll_sway' in columns:
        mapping['roll'] = 'roll_sway'
        mapping['pitch'] = 'pitch_sway'
        mapping['roll_change'] = 'roll_change_sway'
        mapping['pitch_change'] = 'pitch_change_sway'
        mapping['file_type'] = 'sway'
    elif 'roll' in columns:
        mapping['roll'] = 'roll'
        mapping['pitch'] = 'pitch'
        mapping['roll_change'] = 'roll_change'
        mapping['pitch_change'] = 'pitch_change'
        mapping['file_type'] = 'normal'
    else:
        raise ValueError("roll または roll_sway 列が見つかりません")

    # 時間列の確認
    if 'psychopy_time' in columns:
        mapping['time'] = 'psychopy_time'
    else:
        raise ValueError("psychopy_time 列が見つかりません")

    return mapping


def load_data(filepath):
    """
    CSVファイルを読み込む

    Args:
        filepath (str): CSVファイルのパス

    Returns:
        tuple: (pd.DataFrame, dict) データフレームと列名マッピング
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

    df = pd.read_csv(filepath)

    # 列名マッピングを検出
    col_mapping = detect_column_names(df)

    print(f"データ読み込み完了: {filepath}")
    print(f"  - ファイルタイプ: {col_mapping['file_type']}")
    print(f"  - サンプル数: {len(df)}")
    print(f"  - 時間範囲: {df[col_mapping['time']].min():.2f}s ~ {df[col_mapping['time']].max():.2f}s")

    return df, col_mapping


def calculate_head_displacement(df, col_mapping, effective_length_cm, use_change=False):
    """
    倒立振子モデルを用いて頭部変位を計算

    Args:
        df (pd.DataFrame): 角度データを含むデータフレーム
        col_mapping (dict): 列名マッピング
        effective_length_cm (float): 足首から頭部センサまでの距離 [cm]
        use_change (bool): roll_change, pitch_changeを使用するか

    Returns:
        pd.DataFrame: 変位データを追加したデータフレーム
    """
    result_df = df.copy()

    # 使用する角度列を選択
    if use_change:
        roll_col = col_mapping['roll_change']
        pitch_col = col_mapping['pitch_change']
        print(f"  - 使用角度: {roll_col}, {pitch_col}（初期位置基準）")
    else:
        roll_col = col_mapping['roll']
        pitch_col = col_mapping['pitch']
        print(f"  - 使用角度: {roll_col}, {pitch_col}（重力方向基準）")

    # 列が存在するか確認
    if roll_col not in df.columns or pitch_col not in df.columns:
        raise ValueError(f"必要な列が見つかりません: {roll_col}, {pitch_col}")

    # 角度をラジアンに変換
    roll_rad = np.radians(df[roll_col])
    pitch_rad = np.radians(df[pitch_col])

    # 倒立振子モデルによる変位計算
    # 左右変位 (X方向): d_x = L × sin(roll)
    # 前後変位 (Y方向): d_y = L × sin(pitch)
    result_df['displacement_x_cm'] = effective_length_cm * np.sin(roll_rad)
    result_df['displacement_y_cm'] = effective_length_cm * np.sin(pitch_rad)

    # 初期位置からの相対変位も計算
    initial_x = result_df['displacement_x_cm'].iloc[0]
    initial_y = result_df['displacement_y_cm'].iloc[0]
    result_df['displacement_x_relative_cm'] = result_df['displacement_x_cm'] - initial_x
    result_df['displacement_y_relative_cm'] = result_df['displacement_y_cm'] - initial_y

    # 統計情報の表示
    print(f"\n変位計算結果:")
    print(f"  - 有効長（足首→センサ）: {effective_length_cm:.1f} cm")
    print(f"  - 左右変位 (X): {result_df['displacement_x_cm'].min():.2f} ~ {result_df['displacement_x_cm'].max():.2f} cm")
    print(f"  - 前後変位 (Y): {result_df['displacement_y_cm'].min():.2f} ~ {result_df['displacement_y_cm'].max():.2f} cm")
    print(f"  - 左右変位 標準偏差: {result_df['displacement_x_cm'].std():.3f} cm")
    print(f"  - 前後変位 標準偏差: {result_df['displacement_y_cm'].std():.3f} cm")

    return result_df


def plot_displacement_timeseries(df, col_mapping, output_path, title_suffix=""):
    """
    時系列での変位グラフを作成

    Args:
        df (pd.DataFrame): 変位データを含むデータフレーム
        col_mapping (dict): 列名マッピング
        output_path (str): 出力ファイルパス
        title_suffix (str): タイトルに追加する文字列
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    time = df[col_mapping['time']]

    # サブプロット1: 左右変位 (X)
    axes[0].plot(time, df['displacement_x_cm'], 'b-', linewidth=0.8, alpha=0.8)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('左右変位 [cm]')
    axes[0].set_title('左右方向の頭部変位（右が正）')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(['X変位'], loc='upper right')

    # サブプロット2: 前後変位 (Y)
    axes[1].plot(time, df['displacement_y_cm'], 'r-', linewidth=0.8, alpha=0.8)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('前後変位 [cm]')
    axes[1].set_title('前後方向の頭部変位（後ろが正）')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(['Y変位'], loc='upper right')

    # サブプロット3: 両方を重ねて表示
    axes[2].plot(time, df['displacement_x_cm'], 'b-', linewidth=0.8, alpha=0.8, label='左右 (X)')
    axes[2].plot(time, df['displacement_y_cm'], 'r-', linewidth=0.8, alpha=0.8, label='前後 (Y)')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('時間 [秒]')
    axes[2].set_ylabel('変位 [cm]')
    axes[2].set_title('頭部変位の時系列比較')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    fig.suptitle(f'頭部変位の時系列解析（倒立振子モデル）{title_suffix}', fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"時系列グラフを保存: {output_path}")
    plt.close()


def plot_2d_trajectory(df, col_mapping, output_path, title_suffix=""):
    """
    上から見た頭部位置の2Dプロットを作成

    Args:
        df (pd.DataFrame): 変位データを含むデータフレーム
        col_mapping (dict): 列名マッピング
        output_path (str): 出力ファイルパス
        title_suffix (str): タイトルに追加する文字列
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    x = df['displacement_x_cm']
    y = df['displacement_y_cm']
    time = df[col_mapping['time']]

    # 時間に基づいて色を変化させる
    norm = Normalize(vmin=time.min(), vmax=time.max())
    cmap = plt.cm.viridis

    # 軌跡をプロット（時間で色分け）
    for i in range(len(x) - 1):
        ax.plot([x.iloc[i], x.iloc[i+1]], [y.iloc[i], y.iloc[i+1]], 
                color=cmap(norm(time.iloc[i])), linewidth=0.5, alpha=0.7)

    # 開始点と終了点をマーク
    ax.scatter(x.iloc[0], y.iloc[0], c='green', s=100, marker='o', 
               label=f'開始 (t={time.iloc[0]:.1f}s)', zorder=5, edgecolors='black')
    ax.scatter(x.iloc[-1], y.iloc[-1], c='red', s=100, marker='s', 
               label=f'終了 (t={time.iloc[-1]:.1f}s)', zorder=5, edgecolors='black')

    # 平均位置をマーク
    ax.scatter(x.mean(), y.mean(), c='blue', s=150, marker='+', 
               label=f'平均位置', zorder=5, linewidths=3)

    # カラーバー
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='時間 [秒]')

    # 軸の設定
    ax.set_xlabel('左右変位 [cm]\n← 左     右 →', fontsize=12)
    ax.set_ylabel('前後変位 [cm]\n← 前     後 →', fontsize=12)
    ax.set_title(f'頭部位置の軌跡（上から見た図）{title_suffix}', fontsize=14)

    # 軸を等倍にして正方形に
    ax.set_aspect('equal')

    # グリッド
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

    # 凡例
    ax.legend(loc='upper left')

    # 統計情報をテキストで表示
    stats_text = (
        f"統計情報:\n"
        f"  X範囲: {x.min():.2f} ~ {x.max():.2f} cm\n"
        f"  Y範囲: {y.min():.2f} ~ {y.max():.2f} cm\n"
        f"  X標準偏差: {x.std():.3f} cm\n"
        f"  Y標準偏差: {y.std():.3f} cm\n"
        f"  計測時間: {time.max() - time.min():.1f} s"
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D軌跡グラフを保存: {output_path}")
    plt.close()


def plot_2d_density(df, col_mapping, output_path, title_suffix=""):
    """
    頭部位置の密度プロット（ヒートマップ）を作成

    Args:
        df (pd.DataFrame): 変位データを含むデータフレーム
        col_mapping (dict): 列名マッピング
        output_path (str): 出力ファイルパス
        title_suffix (str): タイトルに追加する文字列
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    x = df['displacement_x_cm']
    y = df['displacement_y_cm']

    # 2Dヒストグラム（ヒートマップ）
    h = ax.hist2d(x, y, bins=50, cmap='hot_r', cmin=1)
    plt.colorbar(h[3], ax=ax, label='滞在頻度')

    # 平均位置をマーク
    ax.scatter(x.mean(), y.mean(), c='blue', s=150, marker='+', 
               label=f'平均位置', zorder=5, linewidths=3)

    # 95%信頼楕円（簡易版）
    from matplotlib.patches import Ellipse
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # 95%信頼区間（カイ二乗分布の95%点 ≈ 5.991）
    chi2_val = 5.991
    width = 2 * np.sqrt(chi2_val * eigenvalues[0])
    height = 2 * np.sqrt(chi2_val * eigenvalues[1])

    ellipse = Ellipse(xy=(x.mean(), y.mean()), width=width, height=height,
                      angle=angle, fill=False, color='blue', linewidth=2, 
                      linestyle='--', label='95%信頼楕円')
    ax.add_patch(ellipse)

    # 軸の設定
    ax.set_xlabel('左右変位 [cm]\n← 左     右 →', fontsize=12)
    ax.set_ylabel('前後変位 [cm]\n← 前     後 →', fontsize=12)
    ax.set_title(f'頭部位置の滞在密度（上から見た図）{title_suffix}', fontsize=14)

    # 軸を等倍に
    ax.set_aspect('equal')

    # グリッド
    ax.axhline(y=0, color='white', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='white', linestyle='-', alpha=0.5)

    # 凡例
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"密度プロットを保存: {output_path}")
    plt.close()


def save_displacement_csv(df, col_mapping, output_path):
    """
    変位データをCSVファイルに保存

    Args:
        df (pd.DataFrame): 変位データを含むデータフレーム
        col_mapping (dict): 列名マッピング
        output_path (str): 出力ファイルパス
    """
    time_col = col_mapping['time']
    roll_col = col_mapping['roll']
    pitch_col = col_mapping['pitch']
    roll_change_col = col_mapping['roll_change']
    pitch_change_col = col_mapping['pitch_change']

    columns_to_save = [
        time_col,
        roll_col, pitch_col,
        'displacement_x_cm', 'displacement_y_cm',
        'displacement_x_relative_cm', 'displacement_y_relative_cm'
    ]

    # roll_change, pitch_changeがあれば追加
    if roll_change_col in df.columns:
        columns_to_save.insert(3, roll_change_col)
    if pitch_change_col in df.columns:
        columns_to_save.insert(4, pitch_change_col)

    available_columns = [col for col in columns_to_save if col in df.columns]
    df_to_save = df[available_columns].copy()

    df_to_save.to_csv(output_path, index=False)
    print(f"変位データCSVを保存: {output_path}")


def process_single_file(filepath, effective_length_cm, use_change=False, 
                        start_time=None, end_time=None, output_dir=None):
    """
    単一ファイルを処理

    Args:
        filepath (str): 入力CSVファイルのパス
        effective_length_cm (float): 有効長 [cm]
        use_change (bool): roll_change, pitch_changeを使用するか
        start_time (float): 解析開始時間 [秒]
        end_time (float): 解析終了時間 [秒]
        output_dir (str): 出力ディレクトリ

    Returns:
        bool: 成功したかどうか
    """
    try:
        input_path = Path(filepath)

        # 出力ディレクトリの設定
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # ファイル名のベース
        base_name = input_path.stem
        # _integrated_analysis を除去
        base_name = re.sub(r'_integrated_analysis.*', '', base_name)
        if not base_name:
            base_name = 'displacement'

        # swayファイルかどうかを判定してサフィックスを追加
        if '_sway_' in input_path.name:
            sway_match = re.search(r'_sway_(\d+\.?\d*)Hz', input_path.name)
            if sway_match:
                base_name += f'_sway_{sway_match.group(1)}Hz'

        print("\n" + "=" * 60)
        print(f"処理中: {filepath}")
        print("=" * 60)

        # データ読み込み
        df, col_mapping = load_data(filepath)

        time_col = col_mapping['time']

        # 時間範囲でフィルタリング
        if start_time is not None:
            df = df[df[time_col] >= start_time]
        if end_time is not None:
            df = df[df[time_col] <= end_time]

        if len(df) == 0:
            print("警告: 指定された時間範囲にデータがありません")
            return False

        print(f"\n解析対象:")
        print(f"  - 時間範囲: {df[time_col].min():.2f}s ~ {df[time_col].max():.2f}s")
        print(f"  - サンプル数: {len(df)}")

        # 変位計算
        df_with_displacement = calculate_head_displacement(
            df, col_mapping, effective_length_cm, use_change
        )

        # タイトル用のサフィックス
        title_suffix = f"\n(L={effective_length_cm}cm)"

        # グラフ出力
        print("\n" + "-" * 40)
        print("グラフ生成中...")

        # 時系列グラフ
        timeseries_path = out_dir / f"{base_name}_head_displacement_timeseries.png"
        plot_displacement_timeseries(df_with_displacement, col_mapping, 
                                     str(timeseries_path), title_suffix)

        # 2D軌跡グラフ
        trajectory_path = out_dir / f"{base_name}_head_trajectory_2d.png"
        plot_2d_trajectory(df_with_displacement, col_mapping, 
                          str(trajectory_path), title_suffix)

        # 密度プロット
        density_path = out_dir / f"{base_name}_head_density_2d.png"
        plot_2d_density(df_with_displacement, col_mapping, 
                       str(density_path), title_suffix)

        # CSVデータ保存
        csv_path = out_dir / f"{base_name}_head_displacement.csv"
        save_displacement_csv(df_with_displacement, col_mapping, str(csv_path))

        return True

    except Exception as e:
        print(f"エラー: {filepath} の処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='倒立振子モデルによる頭部変位解析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 単一ファイル
    python head_displacement_analyzer.py data/integrated_analysis.csv

    # ディレクトリ内を再帰的に処理
    python head_displacement_analyzer.py ./A/

    # オプション付き
    python head_displacement_analyzer.py ./A/ --height 175 --ankle 8
    python head_displacement_analyzer.py ./A/ --start 20 --end 300
    python head_displacement_analyzer.py ./A/ --use-change
    python head_displacement_analyzer.py ./A/ --sway  # ローパスフィルタ済みファイルのみ
        """
    )

    parser.add_argument('input_path', type=str, 
                        help='入力CSVファイルまたはディレクトリ')
    parser.add_argument('--height', type=float, default=DEFAULT_HEIGHT_CM, 
                        help=f'被験者の身長 [cm]（デフォルト: {DEFAULT_HEIGHT_CM}）')
    parser.add_argument('--ankle', type=float, default=DEFAULT_ANKLE_HEIGHT_CM, 
                        help=f'足首高さ [cm]（デフォルト: {DEFAULT_ANKLE_HEIGHT_CM}）')
    parser.add_argument('--output', type=str, default=None, 
                        help='出力ディレクトリ（デフォルト: 入力ファイルと同じ）')
    parser.add_argument('--start', type=float, default=DEFAULT_START_TIME, 
                        help='解析開始時間 [秒]')
    parser.add_argument('--end', type=float, default=DEFAULT_END_TIME, 
                        help='解析終了時間 [秒]')
    parser.add_argument('--use-change', action='store_true', default=USE_CHANGE_DEFAULT,
                        help='roll_change, pitch_changeを使用')
    parser.add_argument('--pattern', type=str, default=DEFAULT_FILE_PATTERN,
                        help=f'検索するファイルパターン（デフォルト: {DEFAULT_FILE_PATTERN}）')
    parser.add_argument('--sway', action='store_true', default=DEFAULT_SWAY_ONLY,
                        help='ローパスフィルタ済み（_sway_）ファイルのみを対象')

    args = parser.parse_args()

    print("=" * 60)
    print("頭部変位解析プログラム（倒立振子モデル）")
    print("=" * 60)

    # 有効長の計算
    effective_length_cm = args.height - args.ankle
    print(f"\n倒立振子モデルパラメータ:")
    print(f"  - 身長: {args.height} cm")
    print(f"  - 足首高さ: {args.ankle} cm")
    print(f"  - 有効長 L: {effective_length_cm} cm")

    # ファイル検索
    csv_files = find_csv_files(args.input_path, args.pattern, args.sway)

    if not csv_files:
        print(f"\nエラー: 対象ファイルが見つかりません: {args.input_path}")
        print(f"  パターン: {args.pattern}")
        print(f"  swayのみ: {args.sway}")
        sys.exit(1)

    print(f"\n対象ファイル数: {len(csv_files)}")
    for f in csv_files:
        print(f"  - {f}")

    # 各ファイルを処理
    success_count = 0
    fail_count = 0

    for filepath in csv_files:
        result = process_single_file(
            filepath,
            effective_length_cm,
            args.use_change,
            args.start,
            args.end,
            args.output
        )
        if result:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print("処理完了")
    print(f"  成功: {success_count} ファイル")
    print(f"  失敗: {fail_count} ファイル")
    print("=" * 60)


if __name__ == '__main__':
    main()
