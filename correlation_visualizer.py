#!/usr/bin/env python3
"""
相関係数可視化プログラム (correlation_visualizer.py)

postural_sway_analyzerで生成されたcorrelation_summary.csvを読み込み、
各セッションについて以下の3つのグラフを作成する：

1. 角度変化の箱ひげ図（平均・分散表示付き）
2. 窓相関の箱ひげ図（平均・分散表示付き）  
3. 全体相関の棒グラフ

機能:
- セッション別の3列サブプロット構成
- 実験設定情報の表示（single_color_dot, visual_reverse, audio_reverse, gvs_reverse）
- 統計情報の可視化（平均、分散、箱ひげ図の四分位数）
- 高解像度画像保存

使用例:
    python correlation_visualizer.py correlation_summary_3.0Hz.csv
    python correlation_visualizer.py all_hatano/correlation_summary_3.0Hz.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys
from pathlib import Path


def create_boxplot_data_from_stats(stats_dict, prefix):
    """
    統計情報から箱ひげ図用のデータを構築

    Args:
        stats_dict (dict): 統計情報辞書
        prefix (str): 統計カラムのプレフィックス（例: 'angle', 'audio_window_corr'）

    Returns:
        dict: 箱ひげ図用データ（統計情報が不足している場合は None）
    """
    try:
        # 必要な統計カラムが存在するかチェック
        required_stats = ['mean', 'std', 'var', 'q1', 'median', 'q3', 'min', 'max']
        missing_stats = [stat for stat in required_stats if f'{prefix}_{stat}' not in stats_dict]

        if missing_stats:
            print(f"統計情報不足 ({prefix}): {missing_stats} が見つかりません")
            return None

        return {
            'mean': stats_dict.get(f'{prefix}_mean', np.nan),
            'std': stats_dict.get(f'{prefix}_std', np.nan),
            'var': stats_dict.get(f'{prefix}_var', np.nan),
            'q1': stats_dict.get(f'{prefix}_q1', np.nan),
            'median': stats_dict.get(f'{prefix}_median', np.nan),
            'q3': stats_dict.get(f'{prefix}_q3', np.nan),
            'min': stats_dict.get(f'{prefix}_min', np.nan),
            'max': stats_dict.get(f'{prefix}_max', np.nan),
            'iqr': stats_dict.get(f'{prefix}_iqr', np.nan)
        }
    except Exception as e:
        print(f"箱ひげ図データ構築エラー ({prefix}): {e}")
        return None


def plot_custom_boxplot(ax, data_dict, label, color='blue'):
    """
    統計情報から箱ひげ図を描画

    Args:
        ax: matplotlib軸オブジェクト
        data_dict (dict): 箱ひげ図用統計データ
        label (str): データラベル
        color (str): 箱の色
    """
    if data_dict is None or np.isnan(data_dict['median']):
        ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
        return

    # 箱ひげ図の要素を手動で描画
    try:
        # 箱（Q1からQ3）
        box_height = data_dict['q3'] - data_dict['q1']
        box = mpatches.Rectangle((0.8, data_dict['q1']), 0.4, box_height, 
                               facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(box)

        # 中央値線
        ax.hlines(data_dict['median'], 0.8, 1.2, colors='black', linewidth=2)

        # ひげ（最小値と最大値）
        ax.vlines(1.0, data_dict['min'], data_dict['q1'], colors='black', linestyle='-')
        ax.vlines(1.0, data_dict['q3'], data_dict['max'], colors='black', linestyle='-')
        ax.hlines(data_dict['min'], 0.95, 1.05, colors='black')
        ax.hlines(data_dict['max'], 0.95, 1.05, colors='black')

        # 平均値をマーカーで表示
        ax.plot(1.0, data_dict['mean'], marker='o', color='red', markersize=8, label=f'平均: {data_dict["mean"]:.3f}')

        # 統計情報をテキストで表示
        stats_text = f'平均: {data_dict["mean"]:.3f}\n分散: {data_dict["var"]:.3f}\n標準偏差: {data_dict["std"]:.3f}'
        ax.text(1.3, data_dict['median'], stats_text, va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlim(0.5, 2.0)
        ax.set_xticks([1.0])
        ax.set_xticklabels([label])

    except Exception as e:
        print(f"箱ひげ図描画エラー: {e}")
        ax.text(0.5, 0.5, f'描画エラー: {str(e)}', ha='center', va='center', transform=ax.transAxes)


def create_session_visualization(session_data, output_dir):
    """
    1つのセッションについて3列のサブプロット可視化を作成

    Args:
        session_data (pd.Series): セッションデータ行
        output_dir (str): 出力ディレクトリ
    """
    try:
        # 日本語フォントを設定
        plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

        session_id = session_data['session_id']
        condition = session_data['condition']

        # postural_sway_analyzerと同じ色設定
        color_mapping = {
            'audio_corr_3hz': 'blue',
            'gvs_corr_3hz': 'darkblue', 
            'red_dot_corr_3hz': 'red',
            'green_dot_corr_3hz': 'green'
        }

        window_corr_color_mapping = {
            'audio_window_corr': 'blue',
            'gvs_window_corr': 'darkblue',
            'red_dot_window_corr': 'red',
            'green_dot_window_corr': 'green'
        }

        # 実験設定情報の構築
        settings_info = []
        if session_data.get('single_color_dot', False):
            # audiovisual_experiment.pyの新しいロジックに合わせた表示
            visual_reverse = session_data.get('visual_reverse', False)
            if visual_reverse:
                # VISUAL_REVERSE=Trueの場合、元の条件色を表示
                target_condition = condition
            else:
                # デフォルトでは反対色を表示
                target_condition = 'green' if condition == 'red' else 'red'
            settings_info.append(f'単色ドット({target_condition})')
        if session_data.get('visual_reverse', False):
            settings_info.append('視覚反転')
        if session_data.get('audio_reverse', False):
            settings_info.append('音響反転')
        if session_data.get('gvs_reverse', False):
            settings_info.append('GVS反転')

        settings_str = f" [{', '.join(settings_info)}]" if settings_info else ""

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'セッション解析: {session_id} - 条件: {condition}{settings_str}', fontsize=14, fontweight='bold')

        # 1. 角度変化の箱ひげ図
        angle_data = create_boxplot_data_from_stats(session_data, 'angle')
        if angle_data is not None:
            plot_custom_boxplot(axes[0], angle_data, '角度変化', color='lightblue')
        else:
            axes[0].text(0.5, 0.5, '角度変化統計データなし\n（新しいCSVが必要）', 
                        ha='center', va='center', transform=axes[0].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        axes[0].set_title('角度変化の分布')
        axes[0].set_ylabel('角度変化 (度)')
        axes[0].grid(True, alpha=0.3)

        # 2. 窓相関の箱ひげ図（複数のデータソース）
        correlation_types = []
        correlation_colors = ['blue', 'darkblue', 'red', 'green']  # postural_sway_analyzerと同じ色
        correlation_labels = ['音響', 'GVS', '赤ドット', '緑ドット']
        correlation_prefixes = ['audio_window_corr', 'gvs_window_corr', 'red_dot_window_corr', 'green_dot_window_corr']

        ax2_data = []
        ax2_labels = []
        ax2_colors = []

        for i, (prefix, label, color) in enumerate(zip(correlation_prefixes, correlation_labels, correlation_colors)):
            corr_data = create_boxplot_data_from_stats(session_data, prefix)
            if corr_data is not None and not np.isnan(corr_data['median']):
                ax2_data.append(corr_data)
                ax2_labels.append(label)
                ax2_colors.append(color)

        if ax2_data:
            # 複数の箱ひげ図を横並びで描画
            for j, (data_dict, label, color) in enumerate(zip(ax2_data, ax2_labels, ax2_colors)):
                x_pos = j + 1

                # 箱（Q1からQ3）
                box_height = data_dict['q3'] - data_dict['q1']
                box = mpatches.Rectangle((x_pos-0.2, data_dict['q1']), 0.4, box_height, 
                                       facecolor=color, alpha=0.7, edgecolor='black')
                axes[1].add_patch(box)

                # 中央値線
                axes[1].hlines(data_dict['median'], x_pos-0.2, x_pos+0.2, colors='black', linewidth=2)

                # ひげ
                axes[1].vlines(x_pos, data_dict['min'], data_dict['q1'], colors='black', linestyle='-')
                axes[1].vlines(x_pos, data_dict['q3'], data_dict['max'], colors='black', linestyle='-')
                axes[1].hlines(data_dict['min'], x_pos-0.05, x_pos+0.05, colors='black')
                axes[1].hlines(data_dict['max'], x_pos-0.05, x_pos+0.05, colors='black')

                # 平均値
                axes[1].plot(x_pos, data_dict['mean'], marker='o', color='red', markersize=6)

            axes[1].set_xlim(0.5, len(ax2_data) + 0.5)
            axes[1].set_xticks(range(1, len(ax2_data) + 1))
            axes[1].set_xticklabels(ax2_labels, rotation=45)
            axes[1].set_title('窓相関係数の分布')
            axes[1].set_ylabel('相関係数')
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1].set_ylim(-1, 1)
        else:
            axes[1].text(0.5, 0.5, '窓相関統計データなし\n（新しいCSVが必要）', 
                        ha='center', va='center', transform=axes[1].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            axes[1].set_title('窓相関係数の分布')

        axes[1].grid(True, alpha=0.3)

        # 3. 全体相関の棒グラフ
        overall_correlations = []
        overall_labels = []
        overall_colors = []

        corr_mapping = [
            ('audio_corr_3hz', '音響', 'blue'),
            ('gvs_corr_3hz', 'GVS', 'darkblue'),
            ('red_dot_corr_3hz', '赤ドット', 'red'),
            ('green_dot_corr_3hz', '緑ドット', 'green')
        ]

        for col, label, color in corr_mapping:
            if col in session_data and not pd.isna(session_data[col]):
                overall_correlations.append(session_data[col])
                overall_labels.append(label)
                overall_colors.append(color)

        if overall_correlations:
            bars = axes[2].bar(overall_labels, overall_correlations, color=overall_colors, alpha=0.7, edgecolor='black')

            # 棒グラフの上に数値を表示
            for bar, corr in zip(bars, overall_correlations):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.05),
                           f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        else:
            axes[2].text(0.5, 0.5, '全体相関データなし', ha='center', va='center', transform=axes[2].transAxes)

        axes[2].set_title('全体相関係数')
        axes[2].set_ylabel('相関係数')
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_ylim(-1, 1)
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # ファイル保存
        output_file = os.path.join(output_dir, f'{session_id}_correlation_visualization.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'可視化を保存: {os.path.basename(output_file)}')
        plt.close()

        return True

    except Exception as e:
        print(f'セッション {session_data.get("session_id", "unknown")} の可視化エラー: {e}')
        return False


def load_correlation_summary(filepath):
    """
    correlation_summaryファイルを読み込み

    Args:
        filepath (str): CSVファイルパス

    Returns:
        pd.DataFrame: 読み込まれたデータフレーム
    """
    try:
        df = pd.read_csv(filepath)
        print(f'相関サマリーを読み込み: {filepath}')
        print(f'  - セッション数: {len(df)}')
        print(f'  - カラム数: {len(df.columns)}')

        # 必須カラムの確認
        required_cols = ['session_id', 'condition']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f'警告: 必須カラムが不足: {missing_cols}')

        return df

    except Exception as e:
        print(f'ファイル読み込みエラー: {e}')
        return None


def create_overview_plot(df, output_dir):
    """
    全セッションの概要プロットを作成
    各セッションのグラフを1列ずつサブプロットで表示

    Args:
        df (pd.DataFrame): 相関サマリーデータ
        output_dir (str): 出力ディレクトリ
    """
    try:
        plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']

        # postural_sway_analyzerと同じ色設定
        color_mapping = {
            'audio_corr_3hz': 'blue',
            'gvs_corr_3hz': 'darkblue', 
            'red_dot_corr_3hz': 'red',
            'green_dot_corr_3hz': 'green'
        }

        label_mapping = {
            'audio_corr_3hz': '音響',
            'gvs_corr_3hz': 'GVS',
            'red_dot_corr_3hz': '赤ドット',
            'green_dot_corr_3hz': '緑ドット'
        }

        # セッション数に応じて行数を決定（3列固定）
        n_sessions = len(df)
        n_rows = n_sessions

        # 大きなキャンバスサイズを設定
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))
        fig.suptitle('全セッション相関係数概要', fontsize=16, y=0.995)

        # axesを2次元配列として扱う（セッションが1つの場合の対処）
        if n_sessions == 1:
            axes = axes.reshape(1, -1)

        for session_idx, (idx, session_data) in enumerate(df.iterrows()):
            session_id = session_data['session_id']
            condition = session_data['condition']

            # 実験設定情報の構築
            settings_info = []
            if session_data.get('single_color_dot', False):
                visual_reverse = session_data.get('visual_reverse', False)
                if visual_reverse:
                    target_condition = condition
                else:
                    target_condition = 'green' if condition == 'red' else 'red'
                settings_info.append(f'単色ドット({target_condition})')
            if session_data.get('visual_reverse', False):
                settings_info.append('視覚反転')
            if session_data.get('audio_reverse', False):
                settings_info.append('音響反転')
            if session_data.get('gvs_reverse', False):
                settings_info.append('GVS反転')

            settings_str = f" [{', '.join(settings_info)}]" if settings_info else ""

            # 現在の行の軸を取得
            current_axes = axes[session_idx]

            # 1列目: 角度変化の箱ひげ図
            angle_data = create_boxplot_data_from_stats(session_data, 'angle')
            if angle_data is not None and not np.isnan(angle_data['median']):
                # 箱ひげ図を描画
                box_data = [angle_data['q1'], angle_data['median'], angle_data['q3']]
                whisker_data = [angle_data['min'], angle_data['max']]

                # 箱
                box_height = angle_data['q3'] - angle_data['q1']
                box = mpatches.Rectangle((0.8, angle_data['q1']), 0.4, box_height, 
                                       facecolor='lightblue', alpha=0.7, edgecolor='black')
                current_axes[0].add_patch(box)

                # 中央値線
                current_axes[0].hlines(angle_data['median'], 0.8, 1.2, colors='black', linewidth=2)

                # ひげ
                current_axes[0].vlines(1.0, angle_data['min'], angle_data['q1'], colors='black', linestyle='-')
                current_axes[0].vlines(1.0, angle_data['q3'], angle_data['max'], colors='black', linestyle='-')
                current_axes[0].hlines(angle_data['min'], 0.95, 1.05, colors='black')
                current_axes[0].hlines(angle_data['max'], 0.95, 1.05, colors='black')

                # 平均値
                current_axes[0].plot(1.0, angle_data['mean'], marker='o', color='red', markersize=6)

                current_axes[0].set_xlim(0.5, 1.5)
                current_axes[0].set_xticks([1.0])
                current_axes[0].set_xticklabels(['角度変化'])

                # タイトルに平均と分散を表示
                title_text = f'角度変化分布\n平均: {angle_data["mean"]:.3f}, 分散: {angle_data["var"]:.3f}'
            else:
                current_axes[0].text(0.5, 0.5, '角度変化統計データなし', ha='center', va='center', transform=current_axes[0].transAxes)
                title_text = '角度変化分布\n（データなし）'

            current_axes[0].set_title(title_text, fontsize=10)
            current_axes[0].set_ylabel('角度変化 (度)', fontsize=9)
            current_axes[0].grid(True, alpha=0.3)

            # 2列目: 窓相関の箱ひげ図
            correlation_prefixes = ['audio_window_corr', 'gvs_window_corr', 'red_dot_window_corr', 'green_dot_window_corr']
            correlation_colors = ['blue', 'darkblue', 'red', 'green']
            correlation_labels = ['音響', 'GVS', '赤ドット', '緑ドット']

            window_corr_data = []
            window_corr_labels = []
            window_corr_colors = []
            window_stats = []

            for prefix, color, label in zip(correlation_prefixes, correlation_colors, correlation_labels):
                corr_data = create_boxplot_data_from_stats(session_data, prefix)
                if corr_data is not None and not np.isnan(corr_data['median']):
                    window_corr_data.append(corr_data)
                    window_corr_labels.append(label)
                    window_corr_colors.append(color)
                    window_stats.append(f'{label}: 平均{corr_data["mean"]:.3f}, 分散{corr_data["var"]:.3f}')

            if window_corr_data:
                for j, (data_dict, label, color) in enumerate(zip(window_corr_data, window_corr_labels, window_corr_colors)):
                    x_pos = j + 1

                    # 箱
                    box_height = data_dict['q3'] - data_dict['q1']
                    box = mpatches.Rectangle((x_pos-0.2, data_dict['q1']), 0.4, box_height, 
                                           facecolor=color, alpha=0.7, edgecolor='black')
                    current_axes[1].add_patch(box)

                    # 中央値線
                    current_axes[1].hlines(data_dict['median'], x_pos-0.2, x_pos+0.2, colors='black', linewidth=2)

                    # ひげ
                    current_axes[1].vlines(x_pos, data_dict['min'], data_dict['q1'], colors='black', linestyle='-')
                    current_axes[1].vlines(x_pos, data_dict['q3'], data_dict['max'], colors='black', linestyle='-')
                    current_axes[1].hlines(data_dict['min'], x_pos-0.05, x_pos+0.05, colors='black')
                    current_axes[1].hlines(data_dict['max'], x_pos-0.05, x_pos+0.05, colors='black')

                    # 平均値
                    current_axes[1].plot(x_pos, data_dict['mean'], marker='o', color='red', markersize=4)

                current_axes[1].set_xlim(0.5, len(window_corr_data) + 0.5)
                current_axes[1].set_xticks(range(1, len(window_corr_data) + 1))
                current_axes[1].set_xticklabels(window_corr_labels, rotation=45, fontsize=8)
                current_axes[1].set_ylim(-1, 1)
                current_axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

                # タイトルに平均と分散を追記
                title_text = '窓相関分布\n' + '\n'.join(window_stats)
            else:
                current_axes[1].text(0.5, 0.5, '窓相関統計データなし', ha='center', va='center', transform=current_axes[1].transAxes)
                title_text = '窓相関分布\n（データなし）'

            current_axes[1].set_title(title_text, fontsize=10)
            current_axes[1].set_ylabel('相関係数', fontsize=9)
            current_axes[1].grid(True, alpha=0.3)

            # 3列目: 全体相関の棒グラフ
            overall_correlations = []
            overall_labels = []
            overall_colors = []

            for corr_col, label in label_mapping.items():
                if corr_col in session_data and not pd.isna(session_data[corr_col]):
                    overall_correlations.append(session_data[corr_col])
                    overall_labels.append(label)
                    overall_colors.append(color_mapping[corr_col])

            if overall_correlations:
                bars = current_axes[2].bar(overall_labels, overall_correlations, color=overall_colors, alpha=0.7, edgecolor='black')

                # 数値を棒の上に表示
                for bar, corr in zip(bars, overall_correlations):
                    height = bar.get_height()
                    current_axes[2].text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.05),
                                       f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold', fontsize=8)
            else:
                current_axes[2].text(0.5, 0.5, '全体相関データなし', ha='center', va='center', transform=current_axes[2].transAxes)

            current_axes[2].set_title('全体相関係数', fontsize=10)
            current_axes[2].set_ylabel('相関係数', fontsize=9)
            current_axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            current_axes[2].set_ylim(-1, 1)
            current_axes[2].grid(True, alpha=0.3)
            current_axes[2].tick_params(axis='x', rotation=45, labelsize=8)

            # 各行の左端にセッション情報を表示
            session_info = f'{session_id}\n条件: {condition}{settings_str}'
            fig.text(0.02, 1 - (session_idx + 0.5) / n_rows, session_info, 
                    verticalalignment='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

        plt.tight_layout()
        plt.subplots_adjust(left=0.15, top=0.95)  # セッション情報のスペースを確保

        overview_file = os.path.join(output_dir, 'correlation_overview.png')
        plt.savefig(overview_file, dpi=300, bbox_inches='tight')
        print(f'概要プロットを保存: {os.path.basename(overview_file)}')
        plt.close()

    except Exception as e:
        print(f'概要プロット作成エラー: {e}')


def main():
    """メイン処理"""
    print("相関係数可視化プログラム")
    print("=" * 50)

    # コマンドライン引数の処理
    if len(sys.argv) < 2:
        print("使用法: python correlation_visualizer.py <correlation_summary.csv>")
        print("例: python correlation_visualizer.py correlation_summary_3.0Hz.csv")
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"エラー: ファイルが見つかりません: {input_file}")
        sys.exit(1)

    # データ読み込み
    df = load_correlation_summary(input_file)
    if df is None or df.empty:
        print("エラー: データの読み込みに失敗または空のデータです")
        sys.exit(1)

    # 出力ディレクトリの設定
    input_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else '.'
    output_dir = os.path.join(input_dir, 'correlation_visualizations')
    os.makedirs(output_dir, exist_ok=True)

    print(f"出力ディレクトリ: {output_dir}")
    print()

    # 各セッションの可視化を作成
    success_count = 0
    total_sessions = len(df)

    for idx, session_data in df.iterrows():
        print(f"処理中: {session_data['session_id']} ({idx+1}/{total_sessions})")
        if create_session_visualization(session_data, output_dir):
            success_count += 1

    # 概要プロットの作成
    print("\n概要プロットを作成中...")
    create_overview_plot(df, output_dir)

    print(f"\n{'='*50}")
    print(f"可視化完了")
    print(f"成功: {success_count}/{total_sessions} セッション")
    print(f"出力先: {output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
