#!/usr/bin/env python3
"""
頭部変位位相相関パターン別振幅解析プログラム - 被験者平均版 (displacement_pattern_average_analyzer.py)

displacement_pattern_analyzer.pyが生成したdisplacement_pattern_summary.csvを読み込み、
指定した被験者の平均を計算してグラフを出力する

機能:
1. 指定したフォルダ(複数可)からdisplacement_pattern_summary.csvを読み込み
2. 指定した被験者のデータを抽出
3. 条件別・パターン別に被験者平均を計算
4. 時間割合と平均振幅をそれぞれ平均
5. displacement_pattern_by_subjectと同じ形式でグラフを出力

使用例:
    python displacement_pattern_average_analyzer.py A/ B/ C/ --output result_averaged
    python displacement_pattern_average_analyzer.py B_results/ --subjects hatano omura saito
    python displacement_pattern_average_analyzer.py A_results/ B_results/ C_results/ --output averaged_results
"""

import argparse
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 相関閾値
CORRELATION_THRESHOLD = 0.5


def find_summary_file(directory):
    """
    ディレクトリ内からdisplacement_pattern_summary.csvを探す

    Args:
        directory (str): 検索するディレクトリ

    Returns:
        str or None: 見つかったCSVファイルのパス
    """
    # 直接指定されたパスがCSVファイルの場合
    if os.path.isfile(directory) and directory.endswith('.csv'):
        return directory

    # ディレクトリの場合、その中のdisplacement_pattern_summary.csvを探す
    if os.path.isdir(directory):
        # 直下を探す
        csv_path = os.path.join(directory, 'displacement_pattern_summary.csv')
        if os.path.exists(csv_path):
            return csv_path

        # _resultsサフィックスを付けて探す
        results_dir = directory.rstrip('/') + '_results'
        csv_path = os.path.join(results_dir, 'displacement_pattern_summary.csv')
        if os.path.exists(csv_path):
            return csv_path

        # 再帰的に探す
        pattern = os.path.join(directory, '**', 'displacement_pattern_summary.csv')
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]

    return None


def load_multiple_summary_data(directories):
    """
    複数のディレクトリからdisplacement_pattern_summary.csvを読み込んで結合する

    Args:
        directories (list): ディレクトリのリスト

    Returns:
        pandas.DataFrame: 結合した要約データ
    """
    all_data = []

    for directory in directories:
        csv_path = find_summary_file(directory)
        if csv_path is None:
            print(f"警告: {directory} 内にdisplacement_pattern_summary.csvが見つかりません")
            continue

        df = pd.read_csv(csv_path)
        print(f"データを読み込みました: {csv_path}")
        print(f"  被験者: {df['subject'].unique()}")
        all_data.append(df)

    if not all_data:
        print("エラー: 有効なデータが見つかりませんでした")
        return None

    # 全データを結合
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n結合後の全被験者: {combined['subject'].unique()}")
    print(f"刺激タイプ: {combined['stimulus_type'].unique()}")
    print(f"色条件: {combined['color_condition'].unique()}")

    return combined


def load_summary_data(summary_file):
    """
    displacement_pattern_summary.csvを読み込む

    Args:
        summary_file (str): CSVファイルパス

    Returns:
        pandas.DataFrame: 要約データ
    """
    if not os.path.exists(summary_file):
        print(f"エラー: ファイルが見つかりません: {summary_file}")
        return None

    df = pd.read_csv(summary_file)
    print(f"データを読み込みました: {summary_file}")
    print(f"被験者: {df['subject'].unique()}")
    print(f"刺激タイプ: {df['stimulus_type'].unique()}")
    print(f"色条件: {df['color_condition'].unique()}")
    return df


def calculate_subject_average(summary, subjects):
    """
    指定した被験者の平均を計算

    Args:
        summary (pandas.DataFrame): 要約データ
        subjects (list): 平均を取る被験者のリスト

    Returns:
        pandas.DataFrame: 被験者平均データ
    """
    # 指定した被験者のみフィルタ
    filtered = summary[summary['subject'].isin(subjects)]

    if len(filtered) == 0:
        print(f"警告: 指定した被験者のデータが見つかりません: {subjects}")
        return pd.DataFrame()

    print(f"\n平均を計算する被験者: {filtered['subject'].unique()}")

    # stimulus_typeとcolor_conditionでグループ化して平均を計算
    # 数値列のみを集計
    numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()

    # subject列を除外して集計
    agg_dict = {col: ['mean', 'std', 'count'] for col in numeric_cols}

    averaged = filtered.groupby(['stimulus_type', 'color_condition']).agg(agg_dict)

    # 列名を平坦化
    averaged.columns = ['_'.join(col).strip() for col in averaged.columns.values]
    averaged = averaged.reset_index()

    return averaged


def create_averaged_visualization(summary, subjects, output_dir):
    """
    被験者平均のグラフを作成

    Args:
        summary (pandas.DataFrame): 元の要約データ
        subjects (list): 平均を取る被験者のリスト
        output_dir (str): 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # 指定した被験者のみフィルタ
    filtered_all = summary[summary['subject'].isin(subjects)]

    if len(filtered_all) == 0:
        print(f"警告: 指定した被験者のデータが見つかりません: {subjects}")
        return

    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
    plt.rcParams["font.size"] = 10

    # stimulus_typeとcolor_conditionを使用（全行で共通の軸順を作る）
    stimulus_types = sorted(filtered_all['stimulus_type'].unique())
    color_conditions = sorted(filtered_all['color_condition'].unique())

    pattern_names = ['red_dominant', 'green_dominant', 'both_low']
    n_patterns = len(pattern_names)

    stim_display_names = {
        'vis': 'Vis. Only',
        'audio': 'Vis.+Aud.',
        'gvs': 'Vis.+Ves.',
        'only_audio': 'Aud. Only',
        'only_gvs': 'Ves. Only',
        'all': 'All'
    }

    # 条件リストを作成（visは平均のみ、他はred/greenに分ける）
    condition_labels = []
    condition_display_labels = []
    condition_label_colors = []
    for stim in stimulus_types:
        display_name = stim_display_names.get(stim, stim)
        if stim == 'vis':
            condition_labels.append('vis')
            condition_display_labels.append(display_name)
            condition_label_colors.append('black')
        else:
            condition_labels.append(f'{stim}_red')
            condition_display_labels.append(display_name)
            condition_label_colors.append('#d62728')
            condition_labels.append(f'{stim}_green')
            condition_display_labels.append(display_name)
            condition_label_colors.append('#2ca02c')

    n_conditions = len(condition_labels)
    x_pos = np.arange(n_conditions)
    width = 0.8 / n_patterns

    pattern_colors = {
        'red_dominant': '#ff6b6b',
        'green_dominant': '#51cf66',
        'both_low': '#868e96'
    }
    pattern_labels_short = ['赤ドット優勢', '緑ドット優勢', '両方低相関']

    def _compute_values_for_row(row_df, is_average: bool):
        """1行（平均 or 単一被験者）分の棒グラフ値を作る。

        row_df:
          - 平均行: 複数被験者のsummary行を含むDataFrame
          - 被験者行: subject=1名に絞ったsummary行を含むDataFrame
        is_average:
          - True: 被験者間のばらつきをerror barにする（np.std）
          - False: error barは0（そのまま表示）
        """
        values_by_pattern = []
        errors_by_pattern = []
        pct_by_pattern = []

        for pattern_name in pattern_names:
            values = []
            errors = []
            percentages = []

            for cond_label in condition_labels:
                if cond_label == 'vis':
                    all_vals = []
                    all_pcts = []
                    for color in color_conditions:
                        condition_data = row_df[
                            (row_df['stimulus_type'] == 'vis') &
                            (row_df['color_condition'] == color)
                        ]
                        if len(condition_data) > 0:
                            mean_col = f'{pattern_name}_mean_mean'
                            samples_col = f'{pattern_name}_samples_sum'
                            total_col = 'total_valid_samples_sum'

                            for _, row in condition_data.iterrows():
                                val = row[mean_col]
                                samples = row[samples_col]
                                total = row[total_col]
                                if not pd.isna(val):
                                    all_vals.append(val)
                                if total > 0:
                                    all_pcts.append((samples / total) * 100)

                    if all_vals:
                        values.append(float(np.mean(all_vals)))
                        errors.append(float(np.std(all_vals)) if is_average else 0.0)
                    else:
                        values.append(0.0)
                        errors.append(0.0)

                    if all_pcts:
                        percentages.append(float(np.mean(all_pcts)))
                    else:
                        percentages.append(0.0)

                else:
                    stim = cond_label.rsplit('_', 1)[0]
                    color = cond_label.rsplit('_', 1)[1]
                    condition_data = row_df[
                        (row_df['stimulus_type'] == stim) &
                        (row_df['color_condition'] == color)
                    ]

                    all_vals = []
                    all_pcts = []
                    if len(condition_data) > 0:
                        mean_col = f'{pattern_name}_mean_mean'
                        samples_col = f'{pattern_name}_samples_sum'
                        total_col = 'total_valid_samples_sum'

                        for _, row in condition_data.iterrows():
                            val = row[mean_col]
                            samples = row[samples_col]
                            total = row[total_col]
                            if not pd.isna(val):
                                all_vals.append(val)
                            if total > 0:
                                all_pcts.append((samples / total) * 100)

                    if all_vals:
                        values.append(float(np.mean(all_vals)))
                        errors.append(float(np.std(all_vals)) if is_average else 0.0)
                    else:
                        values.append(0.0)
                        errors.append(0.0)

                    if all_pcts:
                        percentages.append(float(np.mean(all_pcts)))
                    else:
                        percentages.append(0.0)

            values_by_pattern.append(values)
            errors_by_pattern.append(errors)
            pct_by_pattern.append(percentages)

        return values_by_pattern, errors_by_pattern, pct_by_pattern

    def _estimate_ymax_for_row(values_by_pattern, errors_by_pattern):
        ymax = 0.0
        for values, errors in zip(values_by_pattern, errors_by_pattern):
            for v, e in zip(values, errors):
                ymax = max(ymax, float(v) + float(e))
        return ymax

    # まず全行（平均 + 各被験者）で共通のy軸最大値を決める
    avg_vals, avg_errs, _ = _compute_values_for_row(filtered_all, is_average=True)
    y_max = _estimate_ymax_for_row(avg_vals, avg_errs)

    for subj in subjects:
        subj_df = filtered_all[filtered_all['subject'] == subj]
        subj_vals, subj_errs, _ = _compute_values_for_row(subj_df, is_average=False)
        y_max = max(y_max, _estimate_ymax_for_row(subj_vals, subj_errs))

    y_max = (y_max * 1.1) if y_max > 0 else 1.0

    # (被験者数 + 平均) 行の図を作成（平均は最終行）
    n_rows = len(subjects) + 1
    fig_height = 3.6 * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, fig_height), sharex=True)
    if n_rows == 1:
        axes = [axes]

    def _plot_row(ax, row_df, title, is_average: bool, show_legend: bool):
        values_by_pattern, errors_by_pattern, pct_by_pattern = _compute_values_for_row(
            row_df, is_average=is_average
        )

        bar_positions_all = []

        for pattern_idx, (pattern_name, pattern_label) in enumerate(
            zip(pattern_names, pattern_labels_short)
        ):
            values = values_by_pattern[pattern_idx]
            errors = errors_by_pattern[pattern_idx]
            percentages = pct_by_pattern[pattern_idx]

            bar_x = x_pos + pattern_idx * width - width * (n_patterns - 1) / 2
            ax.bar(
                bar_x,
                values,
                width,
                label=pattern_label,
                color=pattern_colors[pattern_name],
                alpha=0.8,
                yerr=errors,
                capsize=3,
            )

            for bx, by, err, pct in zip(bar_x, values, errors, percentages):
                bar_positions_all.append((bx, by, err, pct))

        # 数値表示（%は棒の中、cmは棒の上）
        for bx, by, err, pct in bar_positions_all:
            if by > 0:
                ax.text(
                    bx,
                    by / 2,
                    f'{pct:.1f}%',
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color='white',
                )
                text_y = by + err + y_max * 0.02
                ax.text(
                    bx,
                    text_y,
                    f'{by:.2f}cm',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold',
                )

        ax.set_ylabel('平均振幅 [cm]')
        ax.set_title(title)
        ax.set_ylim(0, y_max * 1.2)
        ax.grid(True, alpha=0.3, axis='y')
        if show_legend:
            ax.legend(loc='upper right', title='相関パターン')

    # 1行目〜: 各被験者（そのまま）
    for row_idx, subj in enumerate(subjects, start=0):
        subj_df = filtered_all[filtered_all['subject'] == subj]
        _plot_row(
            axes[row_idx],
            subj_df,
            title=f'{subj}',
            is_average=False,
            show_legend=False,
        )

    # 図全体の説明（最上部に表示）
    subject_list = ', '.join(subjects)
    fig.suptitle(
        f'相関パターン別 頭部変位平均振幅（閾値={CORRELATION_THRESHOLD}）\n'
        f'棒グラフ上: 平均振幅[cm]、棒グラフ内: 時間割合[%]　※軸ラベル色: 赤=red条件, 緑=green条件',
        y=0.995,
    )

    # 最終行: 被験者平均
    _plot_row(
        axes[-1],
        filtered_all,
        title=(
            f'平均 (n={len(subjects)}): {subject_list}'
        ),
        is_average=True,
        show_legend=True,
    )

    # 横軸（条件ラベル）は全行で表示
    for ax in axes:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(condition_display_labels)
        ax.tick_params(axis='x', labelbottom=True)
        for tick_label, color in zip(ax.get_xticklabels(), condition_label_colors):
            tick_label.set_color(color)
            tick_label.set_fontweight('bold')

    # 最終行のxlabelは「条件」
    axes[-1].set_xlabel('条件')

    plt.tight_layout(rect=[0, 0, 1, 0.965])

    output_file = os.path.join(output_dir, 'displacement_pattern_average.png')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n被験者平均+各被験者のグラフを保存: {output_file}")
    plt.close(fig)


def save_averaged_results(summary, subjects, output_dir):
    """
    被験者平均の結果をCSVに保存

    Args:
        summary (pandas.DataFrame): 元の要約データ
        subjects (list): 平均を取る被験者のリスト
        output_dir (str): 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # 指定した被験者のみフィルタ
    filtered = summary[summary['subject'].isin(subjects)]

    if len(filtered) == 0:
        return

    stimulus_types = sorted(filtered['stimulus_type'].unique())
    color_conditions = sorted(filtered['color_condition'].unique())

    pattern_names = ['red_dominant', 'green_dominant', 'both_low']

    # 結果を格納するリスト
    results = []

    for stim in stimulus_types:
        for color in color_conditions:
            condition_data = filtered[
                (filtered['stimulus_type'] == stim) & 
                (filtered['color_condition'] == color)
            ]

            if len(condition_data) == 0:
                continue

            row = {
                'stimulus_type': stim,
                'color_condition': color,
                'n_subjects': len(condition_data)
            }

            for pattern_name in pattern_names:
                mean_col = f'{pattern_name}_mean_mean'
                samples_col = f'{pattern_name}_samples_sum'
                total_col = 'total_valid_samples_sum'

                # 平均振幅の平均と標準偏差
                vals = condition_data[mean_col].dropna().values
                if len(vals) > 0:
                    row[f'{pattern_name}_amplitude_mean'] = np.mean(vals)
                    row[f'{pattern_name}_amplitude_std'] = np.std(vals)
                else:
                    row[f'{pattern_name}_amplitude_mean'] = np.nan
                    row[f'{pattern_name}_amplitude_std'] = np.nan

                # 時間割合の平均と標準偏差
                pcts = []
                for _, r in condition_data.iterrows():
                    samples = r[samples_col]
                    total = r[total_col]
                    if total > 0:
                        pcts.append((samples / total) * 100)

                if len(pcts) > 0:
                    row[f'{pattern_name}_percentage_mean'] = np.mean(pcts)
                    row[f'{pattern_name}_percentage_std'] = np.std(pcts)
                else:
                    row[f'{pattern_name}_percentage_mean'] = np.nan
                    row[f'{pattern_name}_percentage_std'] = np.nan

            results.append(row)

    # DataFrameに変換して保存
    df_results = pd.DataFrame(results)
    output_file = os.path.join(output_dir, 'displacement_pattern_average_summary.csv')
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"被験者平均サマリーを保存: {output_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='displacement_pattern_summary.csvから被験者平均を計算してグラフを出力'
    )
    parser.add_argument(
        'directories',
        nargs='+',
        help='displacement_pattern_summary.csvを含むディレクトリ（複数指定可）またはCSVファイルのパス'
    )
    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        help='平均を取る被験者のリスト（指定しない場合は全被験者）'
    )
    parser.add_argument(
        '--output', '-o',
        default='averaged_results',
        help='出力ディレクトリ（デフォルト: averaged_results）'
    )

    args = parser.parse_args()

    # データ読み込み（複数ディレクトリ対応）
    summary = load_multiple_summary_data(args.directories)
    if summary is None:
        sys.exit(1)

    # 被験者リストの決定
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = summary['subject'].unique().tolist()

    print(f"\n対象被験者: {subjects}")

    # 被験者平均のグラフ作成
    create_averaged_visualization(summary, subjects, args.output)

    # 被験者平均の結果をCSVに保存
    save_averaged_results(summary, subjects, args.output)

    print(f"\n処理完了")


if __name__ == '__main__':
    main()
