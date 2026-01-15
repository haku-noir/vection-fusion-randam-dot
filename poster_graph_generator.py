#!/usr/bin/env python3
"""
poster_graph_generator.py

実験結果（A, B, C）を集約し、ポスター用の2つの主要なグラフを作成する。
1. 図地判定の安定性比較 (Dominance Stability)
2. 身体動揺への寄与振幅 (Amplitude Contribution)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 日本語フォント設定
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans', 'DejaVu Sans']
# フォントサイズ設定 (ポスター用に大きめ)
plt.rcParams["font.size"] = 18

def load_data():
    """各被験者のデータを読み込む"""
    subjects = ['A', 'B', 'C']
    dfs = []
    
    for subj in subjects:
        path = f"{subj}_results/displacement_pattern_summary.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            # 被験者列がない場合は追加（本来はあるはずだが念のため）
            if 'subject' not in df.columns:
                df['subject'] = subj
            dfs.append(df)
        else:
            print(f"Warning: File not found {path}")
            
    if not dfs:
        return None
        
    return pd.concat(dfs, ignore_index=True)


def process_data_for_graphs(df):
    """
    グラフ用にデータを整理する
    """
    
    processed_rows = []
    
    subjects = df['subject'].unique()
    
    for subj in subjects:
        subj_data = df[df['subject'] == subj]
        
        # Helper to extract dominant metric
        def get_metric(sub_df, condition_type):
            # Red Condition
            df_red = sub_df[sub_df['color_condition'] == 'red']
            if not df_red.empty:
                r_row = df_red.iloc[0]
                r_dom = (r_row['red_dominant_samples_sum'] / r_row['total_valid_samples_sum']) * 100
                r_amp = r_row['red_dominant_mean_mean']
            else:
                r_dom = np.nan
                r_amp = np.nan
                
            # Green Condition
            df_green = sub_df[sub_df['color_condition'] == 'green']
            if not df_green.empty:
                g_row = df_green.iloc[0]
                g_dom = (g_row['green_dominant_samples_sum'] / g_row['total_valid_samples_sum']) * 100
                g_amp = g_row['green_dominant_mean_mean']
            else:
                g_dom = np.nan
                g_amp = np.nan
            
            return np.nanmean([r_dom, g_dom]), np.nanmean([r_amp, g_amp])

        # 1. Ves. Only (only_gvs)
        gvs_only_df = subj_data[subj_data['stimulus_type'] == 'only_gvs']
        if not gvs_only_df.empty:
            dom, amp = get_metric(gvs_only_df, 'only_gvs')
            processed_rows.append({
                'Subject': subj, 'Label': 'Ves. Only', 'MetricType': 'only_gvs',
                'Dominance_Percentage': dom, 'Dominant_Amplitude': amp
            })

        # 2. Vis. + Ves. (gvs)
        gvs_df = subj_data[subj_data['stimulus_type'] == 'gvs']
        if not gvs_df.empty:
            dom, amp = get_metric(gvs_df, 'gvs')
            processed_rows.append({
                'Subject': subj, 'Label': 'Vis. + Ves.', 'MetricType': 'gvs',
                'Dominance_Percentage': dom, 'Dominant_Amplitude': amp
            })

        # 3. Vis. Only (vis)
        vis_df = subj_data[subj_data['stimulus_type'] == 'vis']
        if not vis_df.empty:
            # VisはRed/Green条件の区別が少し特殊だが、summary上は red/green に分かれて格納されている場合とそうでない場合がある
            # 前回のコードでは red/green 分かれていたので同様に処理
            dom, amp = get_metric(vis_df, 'vis')
            processed_rows.append({
                'Subject': subj, 'Label': 'Vis. Only', 'MetricType': 'vis',
                'Dominance_Percentage': dom, 'Dominant_Amplitude': amp
            })

        # 4. Vis. + Aud. (audio)
        aud_df = subj_data[subj_data['stimulus_type'] == 'audio']
        if not aud_df.empty:
            dom, amp = get_metric(aud_df, 'audio')
            processed_rows.append({
                'Subject': subj, 'Label': 'Vis. + Aud.', 'MetricType': 'audio',
                'Dominance_Percentage': dom, 'Dominant_Amplitude': amp
            })

        # 5. Aud. Only (only_audio)
        aud_only_df = subj_data[subj_data['stimulus_type'] == 'only_audio']
        if not aud_only_df.empty:
            dom, amp = get_metric(aud_only_df, 'only_audio')
            processed_rows.append({
                'Subject': subj, 'Label': 'Aud. Only', 'MetricType': 'only_audio',
                'Dominance_Percentage': dom, 'Dominant_Amplitude': amp
            })
            
    return pd.DataFrame(processed_rows)

def get_palette():
    # User colors:
    # Aud: Blue, Vis: Red, Ves: Green
    # Mix: Vis+Aud (Red+Blue -> Purple), Vis+Ves (Red+Green -> Orange/Yellow)
    
    return {
        'Ves. Only': '#2ca02c',     # Green
        'Vis. + Ves.': '#ff7f0e',   # Orange (Red+Greenish)
        'Vis. Only': '#d62728',     # Red
        'Vis. + Aud.': '#9467bd',   # Purple (Red+Blueish)
        'Aud. Only': '#1f77b4'      # Blue
    }

def plot_stability(df, output_dir):
    """図地判定の安定性比較グラフを作成"""
    plt.figure(figsize=(12, 8))
    
    order = ['Ves. Only', 'Vis. + Ves.', 'Vis. Only', 'Vis. + Aud.', 'Aud. Only']
    palette = get_palette()
    
    sns.barplot(
        data=df, 
        x='Label', 
        y='Dominance_Percentage', 
        order=order,
        palette=palette,
        errorbar='se',
        capsize=0.1,
        alpha=0.8,
        zorder=1
    )
    
    sns.swarmplot(
        data=df,
        x='Label',
        y='Dominance_Percentage',
        order=order,
        color='black',
        size=8,
        zorder=2
    )

    plt.ylabel('手掛かりと一致した状態の割合 [%]\n(Dominance Percentage)')
    plt.xlabel('')
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.title('図地解釈の安定性 (Stability)', fontsize=20, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'poster_stability_comparison.png'), dpi=300)
    print("Stability graph saved.")
    plt.close()

def plot_amplitude(df, output_dir):
    """身体動揺への寄与振幅グラフを作成"""
    plt.figure(figsize=(12, 8))
    
    order = ['Ves. Only', 'Vis. + Ves.', 'Vis. Only', 'Vis. + Aud.', 'Aud. Only']
    palette = get_palette()
    
    sns.barplot(
        data=df, 
        x='Label', 
        y='Dominant_Amplitude', 
        order=order,
        palette=palette,
        errorbar='se',
        capsize=0.1,
        alpha=0.8,
        zorder=1
    )
    
    sns.swarmplot(
        data=df,
        x='Label',
        y='Dominant_Amplitude',
        order=order,
        color='black',
        size=8,
        zorder=2
    )

    plt.ylabel('優勢状態における平均振幅 [cm]\n(Mean Amplitude in Dominant State)')
    plt.xlabel('')
    plt.ylim(0, 9.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.title('身体動揺の駆動 (Amplitude)', fontsize=20, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'poster_amplitude_comparison.png'), dpi=300)
    print("Amplitude graph saved.")
    plt.close()


def main():
    print("Loading data...")
    df = load_data()
    if df is None:
        print("No data loaded.")
        return

    print("Processing data...")
    proc_df = process_data_for_graphs(df)
    print(proc_df)
    
    output_dir = 'docs/poster_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating graphs...")
    plot_stability(proc_df, output_dir)
    plot_amplitude(proc_df, output_dir)
    
    # CSV保存 (確認用)
    proc_df.to_csv(os.path.join(output_dir, 'poster_graph_source_data.csv'), index=False)
    print("Done.")

if __name__ == "__main__":
    main()
